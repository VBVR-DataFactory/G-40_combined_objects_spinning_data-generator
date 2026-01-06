"""Entry point for generating combined objects spinning tasks."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core import OutputWriter
from src import TaskConfig, TaskGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined objects spinning tasks.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/questions"),
        help="Output directory for generated tasks.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable ground-truth video generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TaskConfig(
        num_samples=args.num_samples,
        random_seed=args.seed,
        output_dir=args.output_dir,
        generate_videos=not args.no_video,
    )
    generator = TaskGenerator(config)
    task_pairs = generator.generate_dataset()

    writer = OutputWriter(config.output_dir)
    writer.write_dataset(task_pairs)
    print(f"Wrote {len(task_pairs)} tasks to {config.output_dir}")


if __name__ == "__main__":
    main()
