"""CLI tien xu ly du lieu VAI cho ImprovedGS."""
from __future__ import annotations

import argparse
import json

from vai.preprocessing import preprocess_dataset, validate_processed_scene


def main() -> int:
    """Doc tham so va preprocess hoac validate cac scene duoc chon."""
    parser = argparse.ArgumentParser(description="Preprocess VAI data for ImprovedGS")
    parser.add_argument("--input", required=True, help="Thu muc public_set hoac private_set")
    parser.add_argument("--output", required=True, help="Thu muc scene da chuan hoa")
    parser.add_argument("--subset", nargs="*", default=[])
    parser.add_argument("--colmap_executable", default="colmap")
    parser.add_argument("--blank_pixels", type=float, default=1.0)
    parser.add_argument("--min_scale", type=float, default=1.0)
    parser.add_argument("--max_scale", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force_undistort",
        action="store_true",
        help="Ep chay COLMAP undistort ngay ca khi camera da PINHOLE/SIMPLE_PINHOLE",
    )
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        scene_names = args.subset
        if not scene_names:
            raise ValueError("--validate_only yeu cau it nhat mot scene trong --subset")
        results = [validate_processed_scene(f"{args.output}/{name}") for name in scene_names]
    else:
        results = preprocess_dataset(
            input_root=args.input,
            output_root=args.output,
            subset=args.subset,
            colmap_executable=args.colmap_executable,
            blank_pixels=args.blank_pixels,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            overwrite=args.overwrite,
            force_undistort=args.force_undistort,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
