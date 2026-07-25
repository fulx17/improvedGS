"""Render test_poses.csv bang ImprovedGS va khoi phuc radial distortion VAI."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene import Scene
from scene.cameras import MiniCam
from scene.colmap_loader import qvec2rotmat, read_intrinsics_binary
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
from vai.common import (
    load_vai_metadata,
    output_name_for_pose,
    read_pose_rows,
    save_json,
)
from vai.evaluation import evaluate_rendered_scene
from vai.distortion import redistort_and_crop
from vai.image_processing import save_render_image, sharpen_image


def _single_undistorted_camera(source_path: Path) -> Any:
    """Doc camera PINHOLE tren canvas da preprocess."""
    cameras = read_intrinsics_binary(str(source_path / "sparse" / "0" / "cameras.bin"))
    if len(cameras) != 1:
        raise ValueError(f"VAI yeu cau dung 1 camera, nhan duoc {len(cameras)}")
    camera = next(iter(cameras.values()))
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = [float(value) for value in camera.params]
    elif camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = [float(value) for value in camera.params]
        fx = fy = focal
    else:
        raise ValueError(f"Camera render phai la PINHOLE, nhan duoc {camera.model}")
    if abs(fx - fy) > 1e-3:
        raise ValueError(f"VAI redistort yeu cau fx gan bang fy, nhan duoc {fx} va {fy}")
    return {
        "model": camera.model,
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def _needs_redistort(metadata: dict[str, Any]) -> bool:
    """Xac dinh scene co tung undistort (can redistort lai) hay khong."""
    return metadata.get("original_camera", {}).get("model") == "SIMPLE_RADIAL"


def _original_radial_camera(metadata: dict[str, Any]) -> dict[str, float | int]:
    """Lay intrinsics va he so k tu camera SIMPLE_RADIAL goc."""
    camera = metadata.get("original_camera", {})
    params = camera.get("params", [])
    if camera.get("model") != "SIMPLE_RADIAL" or len(params) != 4:
        raise ValueError("VAI metadata khong chua camera SIMPLE_RADIAL hop le")
    return {
        "focal": float(params[0]),
        "cx": float(params[1]),
        "cy": float(params[2]),
        "radial_k": float(params[3]),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
    }


def _validate_pose_intrinsics(
    pose_rows: list[dict[str, str]],
    reference_camera: dict[str, float | int],
) -> None:
    """Chan scene co intrinsics CSV khong khop camera tham chieu da luu."""
    expected_fx = float(reference_camera.get("focal", reference_camera.get("fx", 0.0)))
    expected_fy = float(reference_camera.get("focal", reference_camera.get("fy", 0.0)))
    for row in pose_rows:
        values = {
            "fx": float(row["fx"]),
            "fy": float(row["fy"]),
            "cx": float(row["cx"]),
            "cy": float(row["cy"]),
        }
        expected = {
            "fx": expected_fx,
            "fy": expected_fy,
            "cx": float(reference_camera["cx"]),
            "cy": float(reference_camera["cy"]),
        }
        if any(abs(values[key] - expected[key]) > 1e-3 for key in expected):
            raise ValueError(f"Intrinsics CSV khong khop camera tham chieu tai {row['image_name']}")
        size = (int(float(row["width"])), int(float(row["height"])))
        expected_size = (int(reference_camera["width"]), int(reference_camera["height"]))
        if size != expected_size:
            raise ValueError(
                f"Kich thuoc CSV khong khop camera tham chieu tai {row['image_name']}: {size}!={expected_size}"
            )

def camera_from_pose_row(row: dict[str, str], camera: dict[str, Any]) -> MiniCam:
    """Tao camera render nhe tu qvec va tvec COLMAP trong CSV."""
    qvec = np.array(
        [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
        dtype=np.float64,
    )
    tvec = np.array(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        dtype=np.float64,
    )
    rotation = qvec2rotmat(qvec).T
    fov_x = focal2fov(float(camera["fx"]), int(camera["width"]))
    fov_y = focal2fov(float(camera["fy"]), int(camera["height"]))
    znear = 0.01
    zfar = 100.0
    world_view = torch.tensor(getWorld2View2(rotation, tvec)).transpose(0, 1).cuda()
    projection = getProjectionMatrix(
        znear=znear,
        zfar=zfar,
        fovX=fov_x,
        fovY=fov_y,
    ).transpose(0, 1).cuda()
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    result = MiniCam(
        width=int(camera["width"]),
        height=int(camera["height"]),
        fovy=fov_y,
        fovx=fov_x,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view,
        full_proj_transform=full_projection,
    )
    result.image_name = Path(row["image_name"]).name
    return result


def _prepare_scene_output(output_root: Path, scene_name: str, overwrite: bool) -> Path:
    """Tao thu muc render scene va chi xoa dung scene khi duoc cho phep."""
    output_root.mkdir(parents=True, exist_ok=True)
    scene_output = output_root / scene_name
    if scene_output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Thu muc render da ton tai: {scene_output}. Dung --overwrite de render lai."
            )
        resolved_root = output_root.resolve()
        resolved_scene = scene_output.resolve()
        if resolved_scene.parent != resolved_root or resolved_scene == resolved_root:
            raise ValueError(f"Tu choi xoa thu muc render khong an toan: {resolved_scene}")
        shutil.rmtree(resolved_scene)
    scene_output.mkdir(parents=True)
    return scene_output


def _training_time_minutes(model_path: Path) -> float | str:
    """Doc thoi gian train de giu result_test.json tuong thich batch runner."""
    parameters_path = model_path / "training_parameters.json"
    if not parameters_path.is_file():
        return ""
    import json

    with open(parameters_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("elapsed_minutes", "")
    return "" if value in (None, "") else float(value)


def render_vai_scene(
    dataset: Any,
    pipeline: Any,
    iteration: int = -1,
    scene_name: str = "",
    output_root: str = "",
    eval_root: str = "",
    output_extension: str = "csv",
    save_png: bool = False,
    png_root: str = "",
    redistort_interpolation: str = "bicubic",
    sharpen_amount: float = 1.0,
    sharpen_sigma: float = 0.6,
    jpeg_quality: int = 95,
    jpeg_subsampling: int = 2,
    evaluate: bool = True,
    require_gt: bool = False,
    overwrite: bool = False,
    psnr_max: float = 40.0,
    lpips_net: str = "alex",
) -> dict[str, Any]:
    """Load model, render toan bo test pose, danh gia neu co GT va luu ket qua."""
    source_path = Path(dataset.source_path)
    model_path = Path(dataset.model_path)
    metadata = load_vai_metadata(source_path)
    resolved_scene_name = scene_name or str(metadata.get("scene_name") or source_path.name)
    pose_rows = read_pose_rows(source_path / metadata.get("test_poses", "test/test_poses.csv"))
    undistorted_camera = _single_undistorted_camera(source_path)
    apply_redistort = _needs_redistort(metadata)
    if apply_redistort:
        original_camera = _original_radial_camera(metadata)
        _validate_pose_intrinsics(pose_rows, original_camera)
        radial_k = float(original_camera["radial_k"])
    else:
        _validate_pose_intrinsics(pose_rows, undistorted_camera)
        radial_k = 0.0

    render_root = Path(output_root) if output_root else model_path / "vai_submission"
    png_render_root = Path(png_root) if png_root else model_path / "vai_png"
    if save_png and png_render_root.resolve() == render_root.resolve():
        raise ValueError("png_root phai khac output_root de dong goi JPEG va PNG rieng")
    scene_output = _prepare_scene_output(render_root, resolved_scene_name, overwrite)
    png_output = (
        _prepare_scene_output(png_render_root, resolved_scene_name, overwrite)
        if save_png
        else None
    )
    eval_output_root = Path(eval_root) if eval_root else model_path / "vai_eval"

    with torch.no_grad():
        gaussians = GaussianModel3DGS(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(background_color, dtype=torch.float32, device="cuda")

        for row in tqdm(pose_rows, desc=f"Rendering VAI {resolved_scene_name}", dynamic_ncols=True):
            camera = camera_from_pose_row(row, undistorted_camera)
            rendering = render(
                camera,
                gaussians,
                pipeline,
                background,
                use_trained_exp=dataset.train_test_exp,
                track_gradients=False,
                inference_only=True,
            )["render"]
            if apply_redistort:
                rendering = redistort_and_crop(
                    rendering,
                    focal=float(undistorted_camera["fx"]),
                    render_cx=float(undistorted_camera["cx"]),
                    render_cy=float(undistorted_camera["cy"]),
                    radial_k=radial_k,
                    target_cx=float(row["cx"]),
                    target_cy=float(row["cy"]),
                    target_width=int(float(row["width"])),
                    target_height=int(float(row["height"])),
                    interpolation=redistort_interpolation,
                )
            rendering = sharpen_image(
                rendering,
                amount=sharpen_amount,
                sigma=sharpen_sigma,
            )
            output_name = output_name_for_pose(row["image_name"], output_extension)
            save_render_image(
                rendering,
                scene_output / output_name,
                jpeg_quality=jpeg_quality,
                jpeg_subsampling=jpeg_subsampling,
            )
            if png_output is not None:
                png_name = output_name_for_pose(row["image_name"], "png")
                save_render_image(rendering, png_output / png_name)
            del camera, rendering

        manifest = {
            "scene_name": resolved_scene_name,
            "iteration": int(scene.loaded_iter),
            "image_count": len(pose_rows),
            "render_dir": str(scene_output),
            "output_extension": output_extension,
            "save_png": bool(save_png),
            "png_dir": str(png_output) if png_output is not None else "",
            "redistort_interpolation": redistort_interpolation,
            "sharpen_amount": float(sharpen_amount),
            "sharpen_sigma": float(sharpen_sigma),
            "jpeg_quality": int(jpeg_quality),
            "jpeg_subsampling": int(jpeg_subsampling),
            "radial_k": radial_k,
            "undistorted_camera": undistorted_camera,
            "applied_redistort": apply_redistort,
        }
        save_json(model_path / "vai_render.json", manifest)

        gt_dir = source_path / metadata.get("test_images", "test/images")
        if not evaluate:
            return manifest
        if not gt_dir.is_dir():
            if require_gt:
                raise FileNotFoundError(f"Khong tim thay public ground truth: {gt_dir}")
            print(f"Khong co ground truth, bo qua VAI evaluation: {gt_dir}")
            return manifest

        summary, per_view = evaluate_rendered_scene(
            gt_dir=gt_dir,
            render_dir=scene_output,
            pose_rows=pose_rows,
            output_extension=output_extension,
            psnr_max=psnr_max,
            lpips_net=lpips_net,
            device="cuda",
        )
        full_result = {**manifest, **summary, "per_view": per_view}
        save_json(eval_output_root / f"{resolved_scene_name}.json", full_result)
        save_json(model_path / "vai_per_view.json", per_view)
        batch_summary = {
            "method": f"ours_{scene.loaded_iter}",
            "iteration": int(scene.loaded_iter),
            **summary,
            "FPS": "",
            "NUM": int(gaussians.get_opacity.shape[0]),
            "Training_time": _training_time_minutes(model_path),
            "render_dir": str(scene_output),
        }
        save_json(model_path / "result_test.json", batch_summary)
        print(
            "VAI result {}: SSIM={:.5f} PSNR={:.3f} LPIPS={:.5f} score={:.6f}".format(
                resolved_scene_name,
                summary["SSIM"],
                summary["PSNR"],
                summary["LPIPS"],
                summary["weighted_score"],
            )
        )
        return full_result
