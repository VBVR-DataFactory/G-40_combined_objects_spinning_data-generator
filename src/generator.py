"""
Task generator for Combined Objects Spinning.
"""

import math
import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt


class TaskGenerator(BaseGenerator):
    """Generate tasks with rotation-in-place followed by horizontal translation."""

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=config.image_size)

        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(fps=config.video_fps, output_format="mp4")

        self.color_palette = [
            (235, 120, 120),
            (120, 175, 235),
            (120, 200, 160),
            (230, 170, 95),
            (200, 140, 220),
            (180, 180, 180),
        ]
        self.shape_options = ["circle", "square", "triangle", "pentagon", "hexagon"]

    def generate_task_pair(self, task_id: str) -> TaskPair:
        task_data = self._generate_task_data()

        first_image = self._render_state(task_data, phase="start", show_targets=True)
        final_image = self._render_state(task_data, phase="end", show_targets=False)

        video_path = None
        if self.config.generate_videos and self.video_generator:
            video_path = self._generate_video(task_id, task_data)

        prompt = get_prompt()
        # Replace placeholders
        num_objects = len(task_data["objects"])
        prompt = prompt.replace("[num]", str(num_objects))

        # Build metadata (signature removed as it's redundant)
        metadata = self._build_metadata(task_id, task_data)
        
        
        
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=video_path,
            metadata=metadata
        )

    def _generate_task_data(self) -> dict:
        import time
        start_time = time.time()
        max_time = 10.0  # Maximum 10 seconds for layout generation
        
        for attempt in range(self.config.max_layout_attempts):
            # Check timeout
            if time.time() - start_time > max_time:
                print(f"⚠️  Layout generation timeout after {attempt} attempts")
                break
                
            objects = self._sample_objects()
            cluster_centers = self._sample_cluster_centers(objects)
            if cluster_centers is None:
                continue

            target_centers = self._sample_target_centers(objects, cluster_centers)
            if target_centers is None:
                continue

            for obj, center in zip(objects, cluster_centers):
                obj["start_center"] = center
            for obj, center in zip(objects, target_centers):
                obj["target_center"] = center

            start_centers = [obj["start_center"] for obj in objects]
            target_centers = [obj["target_center"] for obj in objects]
            start_angles = [obj["start_angle"] for obj in objects]
            target_angles = [obj["target_angle"] for obj in objects]

            if not self._layout_is_connected(objects, start_centers, start_angles):
                continue
            if not self._layout_is_separated(objects, target_centers, target_angles):
                continue

            # Clean up objects: remove derivable fields (extents, radius)
            # Only keep task-specific parameters: shape, color, size, start_angle, target_angle, start_center, target_center
            cleaned_objects = []
            for obj in objects:
                cleaned_objects.append({
                    "shape": obj["shape"],
                    "color": obj["color"],
                    "size": obj["size"],
                    "start_angle": obj["start_angle"],
                    "target_angle": obj["target_angle"],
                    "start_center": obj["start_center"],
                    "target_center": obj["target_center"],
                })

            # Removed derivable fields: start_centers, target_centers, start_angles, target_angles
            # These can be derived from objects list
            return {
                "objects": cleaned_objects,
            }

        # Fallback: Try with simpler constraints
        print("⚠️  Trying fallback with relaxed constraints...")
        for attempt in range(50):  # Increased fallback attempts
            objects = self._sample_objects()
            # Further reduce complexity for fallback
            if len(objects) > 2:
                objects = objects[:2]  # Limit to 2 objects in fallback
                
            cluster_centers = self._sample_cluster_centers(objects)
            if cluster_centers is None:
                continue

            target_centers = self._sample_target_centers(objects, cluster_centers)
            if target_centers is None:
                continue

            for obj, center in zip(objects, cluster_centers):
                obj["start_center"] = center
            for obj, center in zip(objects, target_centers):
                obj["target_center"] = center

            start_centers = [obj["start_center"] for obj in objects]
            target_centers = [obj["target_center"] for obj in objects]
            start_angles = [obj["start_angle"] for obj in objects]
            target_angles = [obj["target_angle"] for obj in objects]

            # Skip connectivity check for fallback (less strict)
            # Also skip separation check in final fallback to ensure we can always generate something
            if not self._layout_is_separated(objects, target_centers, target_angles):
                # In final attempts, skip separation check too
                if attempt < 40:
                    continue

            # Clean up objects: remove derivable fields (extents, radius)
            cleaned_objects = []
            for obj in objects:
                cleaned_objects.append({
                    "shape": obj["shape"],
                    "color": obj["color"],
                    "size": obj["size"],
                    "start_angle": obj["start_angle"],
                    "target_angle": obj["target_angle"],
                    "start_center": obj["start_center"],
                    "target_center": obj["target_center"],
                })

            return {
                "objects": cleaned_objects,
            }
        
        raise ValueError("Failed to generate a valid layout even with fallback.")

    def _sample_objects(self) -> list[dict]:
        count = random.randint(self.config.min_objects, self.config.max_objects)
        objects = []
        for _ in range(count):
            shape = random.choice(self.shape_options)
            size = random.randint(self.config.min_size, self.config.max_size)
            start_angle, target_angle = self._pick_rotation_pair()
            extents_start = self._shape_extents(shape, size, start_angle)
            extents_target = self._shape_extents(shape, size, target_angle)
            extents = (max(extents_start[0], extents_target[0]), max(extents_start[1], extents_target[1]))
            radius = max(extents)
            objects.append(
                {
                    "shape": shape,
                    "size": size,
                    "color": random.choice(self.color_palette),
                    "start_angle": start_angle,
                    "target_angle": target_angle,
                    "extents": extents,
                    "radius": radius,
                }
            )
        return objects

    def _pick_rotation_pair(self) -> tuple[int, int]:
        min_deg = self.config.rotation_min_deg
        max_deg = self.config.rotation_max_deg
        if min_deg > max_deg:
            min_deg, max_deg = max_deg, min_deg
        start = random.randint(min_deg, max_deg)
        target = random.randint(min_deg, max_deg)
        min_delta = max(0, self.config.min_rotation_delta)
        for _ in range(40):
            if abs(target - start) >= min_delta:
                break
            target = random.randint(min_deg, max_deg)
        return start, target

    def _sample_cluster_centers(self, objects: list[dict]) -> Optional[list[tuple[float, float]]]:
        count = len(objects)
        if count == 0:
            return None

        for _ in range(self.config.cluster_attempts):
            centers = [(0.0, 0.0)]
            for obj_idx in range(1, count):
                placed = False
                for _ in range(self.config.place_attempts):
                    anchor_idx = random.randrange(len(centers))
                    anchor = objects[anchor_idx]
                    obj = objects[obj_idx]
                    direction = self._random_direction()
                    dist = (
                        self._support_distance(anchor, direction, anchor["start_angle"])
                        + self._support_distance(obj, (-direction[0], -direction[1]), obj["start_angle"])
                        + self.config.contact_gap
                    )
                    cx = centers[anchor_idx][0] + direction[0] * dist
                    cy = centers[anchor_idx][1] + direction[1] * dist
                    candidate = (cx, cy)
                    if self._bbox_overlaps(candidate, obj["extents"], centers, objects, gap=0):
                        continue
                    centers.append(candidate)
                    placed = True
                    break
                if not placed:
                    break
            if len(centers) != count:
                continue
            shifted = self._shift_cluster_left(objects, centers)
            if shifted is None:
                continue
            return shifted

        return None

    def _sample_target_centers(
        self,
        objects: list[dict],
        start_centers: list[tuple[float, float]],
    ) -> Optional[list[tuple[float, float]]]:
        width, _ = self.config.image_size
        margin = self.config.canvas_margin
        right_min = width * self.config.right_region_ratio
        right_max = width - margin
        indices = list(range(len(objects)))
        random.shuffle(indices)

        for _ in range(self.config.place_attempts):
            centers: list[Optional[tuple[float, float]]] = [None] * len(objects)
            placed_centers: list[tuple[float, float]] = []
            placed_extents: list[tuple[float, float]] = []
            for idx in indices:
                obj = objects[idx]
                ext_x, _ = obj["extents"]
                start_y = start_centers[idx][1]
                min_x = right_min + ext_x
                max_x = right_max - ext_x
                if min_x >= max_x:
                    break
                placed = False
                for _ in range(self.config.place_attempts):
                    x = random.uniform(min_x, max_x)
                    candidate = (x, start_y)
                    if not self._bbox_non_overlapping(
                        candidate,
                        obj["extents"],
                        placed_centers,
                        placed_extents,
                        self.config.target_min_spacing,
                    ):
                        continue
                    centers[idx] = candidate
                    placed_centers.append(candidate)
                    placed_extents.append(obj["extents"])
                    placed = True
                    break
                if not placed:
                    break
            if all(center is not None for center in centers):
                return [center for center in centers if center is not None]

        return None

    def _shift_cluster_left(
        self,
        objects: list[dict],
        centers: list[tuple[float, float]],
    ) -> Optional[list[tuple[float, float]]]:
        width, height = self.config.image_size
        margin = self.config.canvas_margin
        left_max = width * self.config.left_region_ratio

        min_x, min_y, max_x, max_y = self._layout_bounds(centers, objects)
        cluster_width = max_x - min_x
        cluster_height = max_y - min_y

        left_limit = margin
        right_limit = left_max - cluster_width
        if right_limit <= left_limit:
            return None

        shift_x = random.uniform(left_limit, right_limit) - min_x
        shift_y = (height - cluster_height) / 2 - min_y

        shifted = [(cx + shift_x, cy + shift_y) for cx, cy in centers]
        if not self._cluster_within_bounds(shifted, objects):
            return None
        return shifted

    def _layout_bounds(
        self,
        centers: list[tuple[float, float]],
        objects: list[dict],
    ) -> tuple[float, float, float, float]:
        min_x = min(cx - obj["extents"][0] for (cx, _), obj in zip(centers, objects))
        max_x = max(cx + obj["extents"][0] for (cx, _), obj in zip(centers, objects))
        min_y = min(cy - obj["extents"][1] for (_, cy), obj in zip(centers, objects))
        max_y = max(cy + obj["extents"][1] for (_, cy), obj in zip(centers, objects))
        return min_x, min_y, max_x, max_y

    def _cluster_within_bounds(self, centers: list[tuple[float, float]], objects: list[dict]) -> bool:
        width, height = self.config.image_size
        margin = self.config.canvas_margin
        min_x, min_y, max_x, max_y = self._layout_bounds(centers, objects)
        return (
            min_x >= margin
            and min_y >= margin
            and max_x <= width - margin
            and max_y <= height - margin
        )

    def _random_direction(self) -> tuple[float, float]:
        angle = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        rad = math.radians(angle)
        return math.cos(rad), math.sin(rad)

    def _support_distance(self, obj: dict, direction: tuple[float, float], angle: int) -> float:
        dx, dy = direction
        if obj["shape"] == "circle":
            return float(obj["size"])
        points = self._shape_points(obj["shape"], (0.0, 0.0), obj["size"], angle)
        return max(px * dx + py * dy for px, py in points)

    def _bbox_overlaps(
        self,
        center: tuple[float, float],
        extents: tuple[float, float],
        other_centers: list[tuple[float, float]],
        objects: list[dict],
        gap: float,
    ) -> bool:
        cx, cy = center
        ex, ey = extents
        for (ox, oy), other in zip(other_centers, objects):
            oex, oey = other["extents"]
            if abs(cx - ox) < ex + oex + gap and abs(cy - oy) < ey + oey + gap:
                return True
        return False

    def _bbox_non_overlapping(
        self,
        center: tuple[float, float],
        extents: tuple[float, float],
        other_centers: list[tuple[float, float]],
        other_extents: list[tuple[float, float]],
        gap: float,
    ) -> bool:
        cx, cy = center
        ex, ey = extents
        for (ox, oy), (oex, oey) in zip(other_centers, other_extents):
            if abs(cx - ox) < ex + oex + gap and abs(cy - oy) < ey + oey + gap:
                return False
        return True

    def _layout_is_connected(
        self,
        objects: list[dict],
        centers: list[tuple[float, float]],
        angles: list[int],
    ) -> bool:
        masks = [self._shape_mask(obj, center, angle) for obj, center, angle in zip(objects, centers, angles)]
        overlaps = self._mask_overlap_matrix(masks)
        if any(overlaps):
            return False
        adjacency = self._mask_touch_matrix(masks)
        visited = {0}
        stack = [0]
        while stack:
            idx = stack.pop()
            for j in range(len(objects)):
                if adjacency[idx][j] and j not in visited:
                    visited.add(j)
                    stack.append(j)
        return len(visited) == len(objects)

    def _layout_is_separated(
        self,
        objects: list[dict],
        centers: list[tuple[float, float]],
        angles: list[int],
    ) -> bool:
        masks = [self._shape_mask(obj, center, angle) for obj, center, angle in zip(objects, centers, angles)]
        overlaps = self._mask_overlap_matrix(masks)
        if any(overlaps):
            return False
        adjacency = self._mask_touch_matrix(masks)
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if adjacency[i][j]:
                    return False
        return True

    def _mask_overlap_matrix(self, masks: list[np.ndarray]) -> list[bool]:
        overlaps = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                overlaps.append(bool((masks[i] & masks[j]).any()))
        return overlaps

    def _mask_touch_matrix(self, masks: list[np.ndarray]) -> list[list[bool]]:
        count = len(masks)
        adjacency = [[False for _ in range(count)] for _ in range(count)]
        dilated = [self._dilate_mask(mask) for mask in masks]
        for i in range(count):
            for j in range(i + 1, count):
                touching = bool((dilated[i] & masks[j]).any()) or bool((dilated[j] & masks[i]).any())
                adjacency[i][j] = touching
                adjacency[j][i] = touching
        return adjacency

    def _shape_mask(self, obj: dict, center: tuple[float, float], angle: int) -> np.ndarray:
        width, height = self.config.image_size
        mask = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        if obj["shape"] == "circle":
            size = obj["size"]
            cx, cy = center
            bbox = [cx - size, cy - size, cx + size, cy + size]
            draw.ellipse(bbox, fill=1)
        else:
            points = self._shape_points(obj["shape"], center, obj["size"], angle)
            draw.polygon(points, fill=1)
        return np.array(mask, dtype=bool)

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        neighbors = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
        return neighbors

    def _render_state(self, task_data: dict, phase: str, show_targets: bool) -> Image.Image:
        image = self.renderer.create_blank_image(self.config.background_color)
        draw = ImageDraw.Draw(image)

        objects = task_data["objects"]
        if show_targets:
            for obj in objects:
                self._draw_target_outline(draw, obj["target_center"], obj)

        # Extract centers and angles from objects
        if phase == "start":
            centers = [obj["start_center"] for obj in objects]
            angles = [obj["start_angle"] for obj in objects]
        else:
            centers = [obj["target_center"] for obj in objects]
            angles = [obj["target_angle"] for obj in objects]

        for obj, center, angle in zip(objects, centers, angles):
            self._draw_object(draw, obj, center, angle)
        return image

    def _generate_video(self, task_id: str, task_data: dict) -> str:
        frames = []
        objects = task_data["objects"]
        # Extract centers and angles from objects
        start_centers = [obj["start_center"] for obj in objects]
        target_centers = [obj["target_center"] for obj in objects]
        start_angles = [obj["start_angle"] for obj in objects]
        target_angles = [obj["target_angle"] for obj in objects]

        for _ in range(self.config.hold_frames):
            frames.append(self._render_frame(objects, start_centers, start_angles, show_targets=True))

        for step in range(1, self.config.rotation_frames + 1):
            t = step / max(1, self.config.rotation_frames)
            angles = [self._lerp_angle(sa, ta, t) for sa, ta in zip(start_angles, target_angles)]
            frames.append(self._render_frame(objects, start_centers, angles, show_targets=True))

        for step in range(1, self.config.move_frames + 1):
            t = step / max(1, self.config.move_frames)
            centers = [
                (self._lerp(sc[0], tc[0], t), sc[1])
                for sc, tc in zip(start_centers, target_centers)
            ]
            frames.append(self._render_frame(objects, centers, target_angles, show_targets=True))

        for _ in range(self.config.end_hold_frames):
            frames.append(self._render_frame(objects, target_centers, target_angles, show_targets=False))

        temp_dir = Path(tempfile.mkdtemp())
        output_path = temp_dir / f"{task_id}.mp4"
        return str(self.video_generator.create_video_from_frames(frames, output_path))

    def _render_frame(
        self,
        objects: list[dict],
        centers: list[tuple[float, float]],
        angles: list[int],
        show_targets: bool,
    ) -> Image.Image:
        image = self.renderer.create_blank_image(self.config.background_color)
        draw = ImageDraw.Draw(image)
        if show_targets:
            for obj in objects:
                self._draw_target_outline(draw, obj["target_center"], obj)
        for obj, center, angle in zip(objects, centers, angles):
            self._draw_object(draw, obj, center, angle)
        return image

    def _draw_object(
        self,
        draw: ImageDraw.ImageDraw,
        obj: dict,
        center: tuple[float, float],
        angle: int,
    ) -> None:
        if obj["shape"] == "circle":
            self._draw_two_tone_circle(draw, center, obj["size"], angle, obj["color"])
            return
        points = self._shape_points(obj["shape"], center, obj["size"], angle)
        draw.polygon(points, fill=obj["color"], outline=self.config.outline_color)

    def _draw_two_tone_circle(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[float, float],
        radius: int,
        angle: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        base = color
        accent = self._adjust_color(color, 1.2)
        start = angle % 360
        draw.ellipse(bbox, fill=base, outline=self.config.outline_color, width=self.config.outline_width)
        draw.pieslice(bbox, start=start, end=(start + 180) % 360, fill=accent)
        draw.ellipse(bbox, outline=self.config.outline_color, width=self.config.outline_width)

    def _draw_target_outline(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[float, float],
        obj: dict,
    ) -> None:
        color = self.config.target_outline_color
        width = self.config.target_outline_width
        if obj["shape"] == "circle":
            self._draw_dashed_circle(draw, center, obj["size"], color, width)
            return
        points = self._shape_points(obj["shape"], center, obj["size"], obj["target_angle"])
        self._draw_dashed_polygon(draw, points, color, width)

    def _draw_dashed_circle(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[float, float],
        radius: int,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        cx, cy = center
        circumference = 2 * math.pi * radius
        dash_len = self.config.target_dash_length
        gap = self.config.target_dash_gap
        step = dash_len + gap
        segments = max(6, int(circumference / step))
        angle_step = 360 / segments
        for i in range(segments):
            start = i * angle_step
            end = start + (dash_len / step) * angle_step
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.arc(bbox, start=start, end=end, fill=color, width=width)

    def _draw_dashed_polygon(
        self,
        draw: ImageDraw.ImageDraw,
        points: list[tuple[float, float]],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        for idx in range(len(points)):
            p1 = points[idx]
            p2 = points[(idx + 1) % len(points)]
            self._draw_dashed_line(draw, p1, p2, color, width)

    def _draw_dashed_line(
        self,
        draw: ImageDraw.ImageDraw,
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            return
        dash_len = self.config.target_dash_length
        gap = self.config.target_dash_gap
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        pos = 0.0
        while pos < length:
            start = pos
            end = min(pos + dash_len, length)
            sx = x1 + dx * start
            sy = y1 + dy * start
            ex = x1 + dx * end
            ey = y1 + dy * end
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos += dash_len + gap

    def _shape_points(
        self,
        shape: str,
        center: tuple[float, float],
        size: int,
        angle: int,
    ) -> list[tuple[float, float]]:
        cx, cy = center
        angle_rad = math.radians(angle)

        def rotate(px: float, py: float) -> tuple[float, float]:
            rx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
            ry = px * math.sin(angle_rad) + py * math.cos(angle_rad)
            return rx + cx, ry + cy

        if shape == "square":
            half = size
            pts = [(-half, -half), (half, -half), (half, half), (-half, half)]
        elif shape == "triangle":
            pts = [
                (0, -size),
                (size * math.cos(math.radians(210)), size * math.sin(math.radians(210))),
                (size * math.cos(math.radians(330)), size * math.sin(math.radians(330))),
            ]
        elif shape == "pentagon":
            pts = [
                (size * math.cos(math.radians(theta)), size * math.sin(math.radians(theta)))
                for theta in range(-90, 270, 72)
            ]
        elif shape == "hexagon":
            pts = [
                (size * math.cos(math.radians(theta)), size * math.sin(math.radians(theta)))
                for theta in range(0, 360, 60)
            ]
        else:
            pts = [(0, 0)]

        return [rotate(px, py) for px, py in pts]

    def _shape_extents(self, shape: str, size: int, angle: int) -> tuple[float, float]:
        if shape == "circle":
            return float(size), float(size)
        points = self._shape_points(shape, (0, 0), size, angle)
        max_x = max(abs(px) for px, _ in points)
        max_y = max(abs(py) for _, py in points)
        return max_x, max_y

    def _adjust_color(self, color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return tuple(min(255, max(0, int(c * factor))) for c in color)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _lerp_angle(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t
