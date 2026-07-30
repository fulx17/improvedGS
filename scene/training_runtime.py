"""
Runtime helpers used by the training script.

This module contains code that is not tied to one training method: output folder
setup, checkpoint I/O, argument parsing, shared loop state, camera sampling, and
periodic evaluation reports.
"""
import os
import time
import uuid
from argparse import ArgumentParser, Namespace
from typing import Any, Optional, TypedDict

import torch
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.general_utils import get_expon_lr_func
from utils.experiment_utils import append_log_lines, archive_existing_output, write_training_parameters
from utils.loss_utils import psnr
from utils.loss_utils import ssim
from scene.training_context import TrainingContext, activate_training_resolution
from utils.coarse_to_fine import resolve_training_resolution_scale
from utils.pose_aware_sampling import (
    build_pose_sampling_plan,
    build_repeated_camera_pool,
    load_test_camera_poses,
    resolve_test_pose_path,
)


class TrainingLoopState(TypedDict):
    """
        State reused by many training-loop stages.

        Keeping these values in one typed dictionary makes call sites shorter
        while still showing which runtime objects are expected.
    """
    background: torch.Tensor
    log_path: str
    depth_l1_weight: Any
    report_lpips_model: Optional[LPIPS]
    viewpoint_stack: list[Any]
    testing_iterations: list[int]
    saving_iterations: list[int]
    checkpoint_iterations: list[int]
    training_resolution_scale: float
    viewpoint_repeat_counts: dict[int, int]


def synchronized_timestamp() -> float:
    """
        Return a timer value after pending CUDA work has finished.

        CUDA kernels are asynchronous, so synchronization is needed before timing
        sections that include GPU work.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _ensure_model_path(args: Any) -> None:
    """
        Fill `args.model_path` with a unique output folder when it is empty.
    """
    if args.model_path:
        return
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    args.model_path = os.path.join("./output/", unique_str[0:10])


def _build_training_background(dataset: Any) -> torch.Tensor:
    """
        Build the fixed white or black CUDA background tensor for training renders.
    """
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    return torch.tensor(bg_color, dtype=torch.float32, device="cuda")


def prepare_output_and_logger(args: Any, runtime_args: Optional[Any] = None) -> None:
    """
        Prepare the experiment folder before training starts.

        Existing output is archived unless this run resumes in place. The current
        arguments are then written so the run can be reproduced later.
    """
    _ensure_model_path(args)

    should_archive_existing = True
    if runtime_args is not None:
        should_archive_existing = not is_inplace_checkpoint_resume(runtime_args)
    if should_archive_existing:
        archived_path = archive_existing_output(args.model_path)
        if archived_path is not None:
            print("Archived existing output to {}".format(archived_path))

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    if runtime_args is not None:
        write_training_parameters(args.model_path, runtime_args)
        cfg_args = Namespace(**vars(runtime_args))
        cfg_args.source_path = args.source_path
        cfg_args.model_path = args.model_path
    else:
        cfg_args = Namespace(**vars(args))
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(cfg_args))


def build_training_loop_state(
    context: TrainingContext,
    report_lpips_test: bool,
    report_lpips_train: bool,
) -> TrainingLoopState:
    """
        Build shared loop state after scene creation.

        This attaches scene-related objects into `context.runtime_state`, creates
        the background tensor, prepares evaluation settings, and initializes the
        shuffled training-camera pool.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene before building loop state.")

    dataset = context.dataset
    opt = context.opt
    runtime_args = context.runtime_args
    background = _build_training_background(dataset)
    runtime_state = context.runtime_state
    runtime_state["scene"] = scene
    runtime_state["pipe"] = context.pipe
    runtime_state["dataset_sh_degree"] = int(dataset.sh_degree)
    runtime_state["minigs_background"] = background

    training_resolution_scale = float(runtime_state.get("training_resolution_scale", 1.0))
    train_cameras = list(runtime_state.get("train_cameras") or scene.getTrainCameras(training_resolution_scale))
    viewpoint_repeat_counts: dict[int, int] = {}
    if bool(getattr(opt, "pose_aware_sampling", False)):
        test_pose_path = resolve_test_pose_path(
            dataset.source_path,
            str(getattr(opt, "pose_aware_test_poses", "")),
        )
        test_poses = load_test_camera_poses(test_pose_path)
        sampling_plan = build_pose_sampling_plan(
            train_cameras,
            test_poses,
            neighbor_count=int(getattr(opt, "pose_aware_k", 3)),
            extra_fraction=float(getattr(opt, "pose_aware_extra_fraction", 0.25)),
            max_repeat=int(getattr(opt, "pose_aware_max_repeat", 2)),
            angle_weight=float(getattr(opt, "pose_aware_angle_weight", 0.25)),
        )
        viewpoint_repeat_counts = sampling_plan.repeat_counts
        print(
            "Pose-aware sampling: {} train, {} test, +{} luot/pool {}, "
            "median spacing {:.3f}, max gap {:.2f}x spacing".format(
                sampling_plan.train_count,
                sampling_plan.test_count,
                sampling_plan.extra_count,
                sampling_plan.pool_size,
                sampling_plan.median_train_spacing,
                sampling_plan.max_test_gap,
            )
        )
    return {
        "background": background,
        "log_path": os.path.join(scene.model_path, "log.txt"),
        "depth_l1_weight": get_expon_lr_func(
            opt.depth_l1_weight_init,
            opt.depth_l1_weight_final,
            max_steps=opt.iterations,
        ),
        "report_lpips_model": LPIPS("vgg").to("cuda").eval() if (report_lpips_test or report_lpips_train) else None,
        "viewpoint_stack": build_repeated_camera_pool(train_cameras, viewpoint_repeat_counts),
        "testing_iterations": list(runtime_args.test_iterations),
        "saving_iterations": list(runtime_args.save_iterations),
        "checkpoint_iterations": list(runtime_args.checkpoint_iterations),
        "training_resolution_scale": training_resolution_scale,
        "viewpoint_repeat_counts": viewpoint_repeat_counts,
    }


def update_training_resolution(
    context: TrainingContext,
    loop_state: TrainingLoopState,
    iteration: int,
) -> None:
    """Doi camera pool khi iteration di qua mot moc coarse-to-fine."""
    target_scale = resolve_training_resolution_scale(iteration, context.opt)
    current_scale = float(loop_state.get("training_resolution_scale", 1.0))
    if target_scale == current_scale:
        return

    train_cameras = activate_training_resolution(context, target_scale)
    loop_state["viewpoint_stack"].clear()
    loop_state["viewpoint_stack"].extend(
        build_repeated_camera_pool(train_cameras, loop_state["viewpoint_repeat_counts"])
    )
    loop_state["training_resolution_scale"] = target_scale
    print("Coarse-to-fine: switched training resolution to 1/{} at iteration {}".format(int(target_scale), iteration))


def sample_training_viewpoint(
    scene: Scene,
    viewpoint_stack: list[Any],
    resolution_scale: float = 1.0,
    repeat_counts: Optional[dict[int, int]] = None,
) -> Any:
    """
        Lay ngau nhien mot train camera va xoa khoi pool hien tai.

        Khi pool het, nap lai moi camera it nhat mot lan va them cac ban sao
        da duoc pose-aware sampling chon.
    """
    if not viewpoint_stack:
        viewpoint_stack.extend(
            build_repeated_camera_pool(
                scene.getTrainCameras(float(resolution_scale)),
                repeat_counts or {},
            )
        )
    random_index = torch.randint(0, len(viewpoint_stack), (1,)).item()
    return viewpoint_stack.pop(random_index)


def parse_training_arguments(argv: list[str]) -> tuple[Any, Any, Any, Namespace]:
    """
        Build the training CLI parser, register all option groups, and parse argv.
    """
    from arguments import ModelParams, OptimizationParams, PipelineParams, parse_bool_arg

    parser = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * (i + 1) for i in range(1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--progress_bar_width", type=int, default=90)
    parser.add_argument("--report_lpips_test", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--report_lpips_train", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint_dir", type=str, default="")
    parser.add_argument("--start_checkpoint_file", type=str, default="")
    parser.add_argument("--empty_cache_interval", type=int, default=200)
    parser.add_argument("--bts_scores_csv", type=str, default="")
    args = parser.parse_args(argv)
    return model_params, optimization_params, pipeline_params, args


def resolve_start_checkpoint_path(model_path: str, runtime_args: Any) -> Optional[str]:
    """
        Return the checkpoint path to load, or None when resume is disabled.

        If the checkpoint directory is omitted, the current model output folder
        is used as the default location.
    """
    checkpoint_file = str(getattr(runtime_args, "start_checkpoint_file", "") or "").strip()
    if checkpoint_file == "":
        return None
    checkpoint_dir = str(getattr(runtime_args, "start_checkpoint_dir", "") or "").strip()
    if checkpoint_dir == "":
        checkpoint_dir = model_path
    return os.path.join(checkpoint_dir, checkpoint_file)


def save_training_checkpoint(
    checkpoint_path: str,
    gaussians: GaussianModel3DGS,
    iteration: int,
    runtime_state: dict[str, Any] | None = None,
) -> None:
    """
        Save Gaussian model state and iteration in the current checkpoint format.
    """
    torch.save(
        {
            "model_params": gaussians.capture(),
            "iteration": int(iteration),
        },
        checkpoint_path,
    )


def load_training_checkpoint(checkpoint_path: str) -> tuple[Any, int]:
    """
        Load checkpoint payload and validate that it matches the supported model type.

        The loader accepts both the current dictionary format and the older
        two-item tuple format, then returns model parameters plus start iteration.
    """
    payload = torch.load(checkpoint_path)
    model_params: Any
    iteration: int

    if isinstance(payload, dict):
        model_params = payload.get("model_params")
        iteration = int(payload.get("iteration", 0))
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        model_params = payload[0]
        iteration = int(payload[1])
    else:
        raise ValueError("Unsupported checkpoint payload format: {}".format(type(payload).__name__))

    checkpoint_model_type = ""
    if isinstance(model_params, tuple):
        scaling_index: Optional[int] = None
        if len(model_params) >= 15:
            scaling_index = 5
        elif len(model_params) in {12, 13, 14}:
            scaling_index = 4
        if scaling_index is not None and scaling_index < len(model_params):
            scaling_tensor = model_params[scaling_index]
            if isinstance(scaling_tensor, torch.Tensor) and scaling_tensor.ndim >= 2:
                scaling_dim = int(scaling_tensor.shape[-1])
                if scaling_dim == 2:
                    checkpoint_model_type = "2dgs"
                elif scaling_dim == 3:
                    checkpoint_model_type = "3dgs"

    if checkpoint_model_type != "" and checkpoint_model_type != "3dgs":
        print(
            "Checkpoint type mismatch. The open-source version only supports `3dgs`, but checkpoint `{}` belongs to `{}`. Training stopped.".format(
                checkpoint_path,
                checkpoint_model_type,
            )
        )
        raise SystemExit(1)

    return model_params, iteration


def is_inplace_checkpoint_resume(runtime_args: Any) -> bool:
    """
        Check whether resume should reuse the current output folder.

        An empty checkpoint directory means the checkpoint is expected inside
        `model_path`, so archiving that folder would remove the resume source.
    """
    checkpoint_file = str(getattr(runtime_args, "start_checkpoint_file", "") or "").strip()
    checkpoint_dir = str(getattr(runtime_args, "start_checkpoint_dir", "") or "").strip()
    return checkpoint_file != "" and checkpoint_dir == ""


def _evaluate_report_view(
    viewpoint: Any,
    scene: Scene,
    render_func: Any,
    render_args: tuple[Any, ...],
    train_test_exp: bool,
    l1_loss_fn: Any,
    lpips_model: Optional[LPIPS],
    report_lpips: bool,
    runtime_state: dict[str, Any] | None,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
        Render one validation view and compute all requested image metrics.

        The train/test exposure mode compares only the held-out half of the
        image, matching the exposure-split training setup.
    """
    image = torch.clamp(
        render_func(viewpoint, scene.gaussians, *render_args)["render"],
        0.0,
        1.0,
    )
    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
    if train_test_exp:
        image = image[..., image.shape[-1] // 2:]
        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
    image_batch = image.unsqueeze(0)
    gt_image_batch = gt_image.unsqueeze(0)
    l1_value = l1_loss_fn(image, gt_image).mean().double()
    psnr_value = psnr(image, gt_image).mean().double()
    ssim_value = ssim(image_batch, gt_image_batch).mean().double()
    lpips_value = None
    if lpips_model is not None and report_lpips:
        lpips_value = lpips_model(image_batch, gt_image_batch).mean().double()
    return l1_value, psnr_value, ssim_value, lpips_value


def training_report(
    iteration: int,
    l1_loss_fn: Any,
    scene: Scene,
    render_func: Any,
    render_args: tuple[Any, ...],
    train_test_exp: bool,
    lpips_model: Optional[LPIPS],
    report_lpips_test: bool,
    report_lpips_train: bool,
    log_path: Optional[str],
    is_first_report_iteration: bool,
    runtime_state: dict[str, Any] | None = None,
) -> None:
    """
        Evaluate train and test cameras and write a compact report.

        The function averages metrics over each split, prints them through tqdm,
        and mirrors the same lines into the experiment log file.
    """
    torch.cuda.empty_cache()
    any_split_reports_lpips = report_lpips_test or report_lpips_train
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras(), "report_lpips": report_lpips_test},
        {"name": "train", "cameras": scene.getTrainCameras(), "report_lpips": report_lpips_train},
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0 if config["report_lpips"] and lpips_model is not None else None
            for viewpoint in config["cameras"]:
                l1_value, psnr_value, ssim_value, lpips_value = _evaluate_report_view(
                    viewpoint,
                    scene,
                    render_func,
                    render_args,
                    train_test_exp,
                    l1_loss_fn,
                    lpips_model,
                    config["report_lpips"],
                    runtime_state,
                    config["name"],
                )
                l1_test += l1_value
                psnr_test += psnr_value
                ssim_test += ssim_value
                if lpips_test is not None and lpips_value is not None:
                    lpips_test += lpips_value
            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            ssim_test /= len(config["cameras"])
            if lpips_test is not None:
                lpips_test /= len(config["cameras"])
            split_label = "Test " if config["name"] == "test" else "Train"
            message = "[{}] {}: [L1 {:.5f}] [PSNR {:.3f}] [SSIM {:.5f}]".format(
                iteration,
                split_label,
                float(l1_test),
                float(psnr_test),
                float(ssim_test),
            )
            if config["report_lpips"] and lpips_test is not None:
                message += " [LPIPS {:.5f}]".format(float(lpips_test))
            elif any_split_reports_lpips:
                message += " [LPIPS -------]"
            message += " [N_GS {}]".format(scene.gaussians.get_xyz.shape[0])
            log_lines = [message]
            if config["name"] == "train":
                log_lines.append("")
                tqdm.write(message)
            else:
                terminal_message = message if is_first_report_iteration else "\n" + message
                tqdm.write(terminal_message)
            if log_path is not None:
                append_log_lines(log_path, log_lines)
    torch.cuda.empty_cache()
