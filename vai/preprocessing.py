"""Tien xu ly scene VAI thanh layout COLMAP ma ImprovedGS doc truc tiep."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vai import VAI_METADATA_FILENAME
from vai.colmap_io import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    write_extrinsics_binary,
)
from vai.common import camera_to_dict, read_pose_rows, save_json


def _single_camera(sparse_dir: Path) -> Any:
    """Doc camera duy nhat trong sparse model cua scene VAI."""
    cameras = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    if len(cameras) != 1:
        raise ValueError(f"VAI yeu cau dung 1 camera, nhan duoc {len(cameras)} tai {sparse_dir}")
    return next(iter(cameras.values()))


def _check_colmap_executable(executable: str) -> None:
    """Bao loi som neu COLMAP CLI chua san sang."""
    explicit_path = Path(executable)
    if explicit_path.is_file() or shutil.which(executable):
        return
    raise FileNotFoundError(
        f"Khong tim thay COLMAP CLI '{executable}'. Hay cai COLMAP hoac truyen --colmap_executable."
    )


def _run_colmap_undistorter(
    executable: str,
    image_path: Path,
    sparse_path: Path,
    output_path: Path,
    blank_pixels: float,
    min_scale: float,
    max_scale: float,
) -> None:
    """Chay image_undistorter voi tham so da kiem chung tu baseline."""
    command = [
        executable,
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(sparse_path),
        "--output_path",
        str(output_path),
        "--output_type",
        "COLMAP",
        "--blank_pixels",
        str(blank_pixels),
        "--min_scale",
        str(min_scale),
        "--max_scale",
        str(max_scale),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"COLMAP image_undistorter that bai:\n{details}") from error


def _files_by_stem(folder: Path) -> dict[str, Path]:
    """Lap bang tra file theo stem va chan ten trung lap."""
    result: dict[str, Path] = {}
    for path in sorted(item for item in folder.iterdir() if item.is_file()):
        if path.stem in result:
            raise ValueError(f"Trung stem anh trong {folder}: {path.stem}")
        result[path.stem] = path
    return result


def _embed_alpha_masks(undistorted_images: Path, mask_images: Path) -> int:
    """Ghep vung hop le cua anh undistort vao kenh alpha PNG."""
    image_by_stem = _files_by_stem(undistorted_images)
    mask_by_stem = _files_by_stem(mask_images)
    if set(image_by_stem) != set(mask_by_stem):
        missing_masks = sorted(set(image_by_stem) - set(mask_by_stem))
        missing_images = sorted(set(mask_by_stem) - set(image_by_stem))
        raise ValueError(
            "Anh va mask undistort khong khop. thieu_mask={} thieu_anh={}".format(
                missing_masks[:5], missing_images[:5]
            )
        )

    for stem, image_path in image_by_stem.items():
        mask = np.asarray(Image.open(mask_by_stem[stem]).convert("L"))
        alpha = (mask > 127).astype(np.uint8) * 255
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        if rgb.shape[:2] != alpha.shape:
            raise ValueError(f"Kich thuoc anh va mask khong khop: {image_path.name}")
        output_path = image_path.with_suffix(".png")
        Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(output_path)
        if output_path != image_path:
            image_path.unlink()
    return len(image_by_stem)


def _embed_full_alpha_masks(image_dir: Path) -> int:
    """Gan alpha=255 toan bo cho anh da la PINHOLE (khong can COLMAP mask)."""
    count = 0
    for path in sorted(item for item in image_dir.iterdir() if item.is_file()):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        output_path = path.with_suffix(".png")
        Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(output_path)
        if output_path != path:
            path.unlink()
        count += 1
    return count

def _build_white_images(source_images: Path, white_images: Path) -> None:
    """Tao anh trang de COLMAP sinh mask hinh hoc chinh xac."""
    white_images.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(item for item in source_images.iterdir() if item.is_file()):
        with Image.open(source_path) as image:
            Image.new("RGB", image.size, (255, 255, 255)).save(white_images / source_path.name)


def _synchronize_and_filter_images(sparse_dir: Path, image_dir: Path) -> int:
    """Dong bo duoi PNG va loai camera khong co anh train tren dia."""
    images_path = sparse_dir / "images.bin"
    images = read_extrinsics_binary(str(images_path))
    files_by_stem = _files_by_stem(image_dir)
    filtered: dict[int, Any] = {}
    for image_id, image in images.items():
        disk_path = files_by_stem.get(Path(image.name).stem)
        if disk_path is None:
            continue
        filtered[image_id] = image._replace(name=disk_path.name)
    if not filtered:
        raise ValueError(f"Khong con camera train nao sau khi loc {images_path}")
    write_extrinsics_binary(filtered, images_path)
    return len(filtered)


def _copy_raw_scene(source_scene: Path, work_scene: Path) -> None:
    """Sao chep dung cac thanh phan can cho pipeline VAI."""
    train_dir = source_scene / "train"
    source_images = train_dir / "images"
    source_sparse = train_dir / "sparse"
    pose_csv = source_scene / "test" / "test_poses.csv"
    for required_path in (source_images, source_sparse / "0", pose_csv):
        if not required_path.exists():
            raise FileNotFoundError(f"Scene VAI thieu du lieu bat buoc: {required_path}")

    shutil.copytree(source_images, work_scene / "images")
    shutil.copytree(source_sparse, work_scene / "sparse")
    shutil.copytree(source_scene / "test", work_scene / "test")
    readme_path = source_scene / "README.txt"
    if readme_path.is_file():
        shutil.copy2(readme_path, work_scene / readme_path.name)


def _replace_with_undistorted_scene(
    work_scene: Path,
    colmap_executable: str,
    blank_pixels: float,
    min_scale: float,
    max_scale: float,
) -> tuple[int, int]:
    """Undistort anh that, sinh alpha mask va thay sparse model bang PINHOLE."""
    source_images = work_scene / "images"
    source_sparse = work_scene / "sparse" / "0"
    temp_root = Path(tempfile.mkdtemp(prefix="vai-undistort-"))
    try:
        image_output = temp_root / "image_output"
        _run_colmap_undistorter(
            colmap_executable,
            source_images,
            source_sparse,
            image_output,
            blank_pixels,
            min_scale,
            max_scale,
        )

        white_images = temp_root / "white_images"
        mask_output = temp_root / "mask_output"
        _build_white_images(source_images, white_images)
        _run_colmap_undistorter(
            colmap_executable,
            white_images,
            source_sparse,
            mask_output,
            blank_pixels,
            min_scale,
            max_scale,
        )

        embedded_count = _embed_alpha_masks(
            image_output / "images",
            mask_output / "images",
        )
        registered_count = _synchronize_and_filter_images(
            image_output / "sparse",
            image_output / "images",
        )

        shutil.rmtree(work_scene / "images")
        shutil.rmtree(work_scene / "sparse")
        shutil.move(str(image_output / "images"), str(work_scene / "images"))
        (work_scene / "sparse").mkdir()
        shutil.move(str(image_output / "sparse"), str(work_scene / "sparse" / "0"))
        return embedded_count, registered_count
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _finalize_already_pinhole_scene(work_scene: Path) -> tuple[int, int]:
    """Chuan hoa scene da PINHOLE/SIMPLE_PINHOLE: embed alpha full, sync ten PNG."""
    images_dir = work_scene / "images"
    sparse_dir = work_scene / "sparse" / "0"
    embedded_count = _embed_full_alpha_masks(images_dir)
    registered_count = _synchronize_and_filter_images(sparse_dir, images_dir)
    return embedded_count, registered_count

def validate_processed_scene(scene_path: str | Path) -> dict[str, Any]:
    """Kiem tra scene da san sang cho train va render VAI."""
    scene_path = Path(scene_path)
    image_dir = scene_path / "images"
    sparse_dir = scene_path / "sparse" / "0"
    pose_csv = scene_path / "test" / "test_poses.csv"
    metadata_path = scene_path / VAI_METADATA_FILENAME
    for required_path in (image_dir, sparse_dir, pose_csv, metadata_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Scene preprocess thieu: {required_path}")

    camera = _single_camera(sparse_dir)
    if camera.model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(f"Camera sau preprocess phai la PINHOLE, nhan duoc {camera.model}")

    files_by_stem = _files_by_stem(image_dir)
    images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
    registered_names = {image.name for image in images.values()}
    disk_names = {path.name for path in files_by_stem.values()}
    if registered_names != disk_names:
        raise ValueError(
            "images.bin va thu muc images khong khop: registered={} disk={}".format(
                len(registered_names), len(disk_names)
            )
        )
    non_rgba = []
    for image_path in files_by_stem.values():
        with Image.open(image_path) as image:
            if image.mode != "RGBA":
                non_rgba.append(image_path.name)
    if non_rgba:
        raise ValueError(f"Anh train chua co alpha mask: {non_rgba[:5]}")

    pose_count = len(read_pose_rows(pose_csv))
    return {
        "scene_name": scene_path.name,
        "train_images": len(disk_names),
        "registered_images": len(registered_names),
        "test_poses": pose_count,
        "camera": camera_to_dict(camera),
    }


def _safe_remove_scene(scene_path: Path, output_root: Path) -> None:
    """Chi xoa scene con nam truc tiep ben trong output root."""
    resolved_scene = scene_path.resolve()
    resolved_root = output_root.resolve()
    if resolved_scene == resolved_root or resolved_scene.parent != resolved_root:
        raise ValueError(f"Tu choi xoa duong dan ngoai output root: {resolved_scene}")
    shutil.rmtree(resolved_scene)


def preprocess_scene(
    source_scene: str | Path,
    output_root: str | Path,
    colmap_executable: str = "colmap",
    blank_pixels: float = 1.0,
    min_scale: float = 1.0,
    max_scale: float = 2.0,
    overwrite: bool = False,
    force_undistort: bool = False,
) -> dict[str, Any]:
    """Chuyen mot scene raw VAI thanh scene ImprovedGS co metadata distortion."""
    source_scene = Path(source_scene)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _check_colmap_executable(colmap_executable)

    scene_name = source_scene.name
    output_scene = output_root / scene_name
    if output_scene.exists() and not overwrite:
        raise FileExistsError(
            f"Output scene da ton tai: {output_scene}. Dung --overwrite de tao lai."
        )

    work_scene = Path(tempfile.mkdtemp(prefix=f".{scene_name}-", dir=output_root))
    try:
        _copy_raw_scene(source_scene, work_scene)
        original_camera = _single_camera(work_scene / "sparse" / "0")
        needs_undistort = force_undistort or original_camera.model == "SIMPLE_RADIAL"

        if needs_undistort:
            if original_camera.model not in {"SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE"}:
                raise ValueError(
                    f"Camera raw VAI khong ho tro, nhan duoc {original_camera.model}"
                )
            embedded_count, registered_count = _replace_with_undistorted_scene(
                work_scene,
                colmap_executable,
                blank_pixels,
                min_scale,
                max_scale,
            )
        elif original_camera.model in {"PINHOLE", "SIMPLE_PINHOLE"}:
            embedded_count, registered_count = _finalize_already_pinhole_scene(work_scene)
        else:
            raise ValueError(
                f"Camera raw VAI phai la SIMPLE_RADIAL, PINHOLE hoac SIMPLE_PINHOLE, "
                f"nhan duoc {original_camera.model}"
            )
        undistorted_camera = _single_camera(work_scene / "sparse" / "0")
        pose_count = len(read_pose_rows(work_scene / "test" / "test_poses.csv"))
        metadata = {
            "format_version": 1,
            "scene_name": scene_name,
            "original_camera": camera_to_dict(original_camera),
            "undistorted_camera": camera_to_dict(undistorted_camera),
            "train_image_count": embedded_count,
            "registered_image_count": registered_count,
            "test_pose_count": pose_count,
            "test_poses": "test/test_poses.csv",
            "test_images": "test/images",
            "undistort": (
                {
                    "blank_pixels": float(blank_pixels),
                    "min_scale": float(min_scale),
                    "max_scale": float(max_scale),
                }
                if needs_undistort
                else None
            ),
        }
        save_json(work_scene / VAI_METADATA_FILENAME, metadata)
        validation = validate_processed_scene(work_scene)

        if output_scene.exists():
            _safe_remove_scene(output_scene, output_root)
        os.replace(work_scene, output_scene)
        validation["scene_name"] = scene_name
        validation["output_path"] = str(output_scene)
        return validation
    except Exception:
        shutil.rmtree(work_scene, ignore_errors=True)
        raise


def preprocess_dataset(
    input_root: str | Path,
    output_root: str | Path,
    subset: list[str] | None = None,
    **options: Any,
) -> list[dict[str, Any]]:
    """Tien xu ly cac scene duoc chon trong public hoac private set."""
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Khong tim thay VAI set: {input_root}")
    requested = set(subset or [])
    scene_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    available = {path.name for path in scene_dirs}
    missing = sorted(requested - available)
    if missing:
        raise ValueError("Khong tim thay scene: {}".format(", ".join(missing)))
    selected = [path for path in scene_dirs if not requested or path.name in requested]
    if not selected:
        raise ValueError(f"Khong co scene nao trong {input_root}")

    results = []
    for scene_dir in selected:
        print(f"Preprocessing VAI scene {scene_dir.name}...")
        result = preprocess_scene(scene_dir, output_root, **options)
        results.append(result)
        print(
            "  OK: train={} poses={} camera={} -> {}".format(
                result["train_images"],
                result["test_poses"],
                result["camera"]["model"],
                result["output_path"],
            )
        )
    return results
