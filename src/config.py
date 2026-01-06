"""Configuration for Combined Objects Spinning."""

from core.base_generator import GenerationConfig


class TaskConfig(GenerationConfig):
    """Task-specific configuration."""

    domain: str = "combined_objects_spinning"
    num_samples: int = 50
    image_size: tuple[int, int] = (512, 512)

    generate_videos: bool = True
    video_fps: int = 10
    hold_frames: int = 4
    rotation_frames: int = 12
    move_frames: int = 18
    end_hold_frames: int = 1

    min_objects: int = 2
    max_objects: int = 4
    min_size: int = 28
    max_size: int = 40

    rotation_min_deg: int = -90
    rotation_max_deg: int = 90
    min_rotation_delta: int = 30

    canvas_margin: int = 30
    left_region_ratio: float = 0.5
    right_region_ratio: float = 0.5
    target_min_spacing: int = 8
    contact_gap: int = 0

    cluster_attempts: int = 80
    place_attempts: int = 300
    max_layout_attempts: int = 120

    background_color: tuple[int, int, int] = (255, 255, 255)
    outline_color: tuple[int, int, int] = (60, 60, 60)
    outline_width: int = 2

    target_outline_color: tuple[int, int, int] = (120, 120, 120)
    target_outline_width: int = 2
    target_dash_length: int = 8
    target_dash_gap: int = 6
