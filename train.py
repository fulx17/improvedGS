"""
Main training script for 3DGS experiments.

The script builds the experiment output folder, creates the Gaussian model and
scene, then runs the iteration loop. Each loop iteration renders one training
view, computes losses, updates model parameters, and optionally changes the
Gaussian structure.
"""
import sys
import os
from argparse import Namespace
import torch
from tqdm import tqdm
from arguments import GroupParams
from utils.experiment_utils import finalize_training_parameters
from utils.general_utils import safe_state
from scene import Scene
from scene.methods.initialization_3dgs import build_gaussian_model_3dgs
from scene.methods.optimization_methods import run_3dgs_optimization_method
from scene.methods.pruning_methods import run_3dgs_pruning_method
from scene.methods.densification_stage import run_3dgs_densification_method
from scene.training_loop import (
    run_3dgs_parameter_update_method,
    run_log_and_save_stage,
    update_3dgs_training_schedule,
    update_training_progress_bar,
)
from scene.training_context import attach_scene_and_gaussians_to_context, build_training_context
from scene.training_runtime import (
    build_training_loop_state,
    load_training_checkpoint,
    parse_training_arguments,
    prepare_output_and_logger,
    resolve_start_checkpoint_path,
    sample_training_viewpoint,
    synchronized_timestamp,
    update_training_resolution,
)
from utils.coarse_to_fine import build_training_resolution_scales


def training(dataset: GroupParams, opt: GroupParams, pipe: GroupParams, runtime_args: Namespace) -> float:
    """
        Run one complete training job and return pure training time in seconds.

        The function keeps the high-level training order in one place: prepare
        files, build shared context, restore checkpoint if needed, then repeat
        optimization, logging, densification, and pruning until the final
        iteration.
    """
    # Nap truoc cac muc do phan giai can dung de Scene tao dung camera.
    resolution_scales = build_training_resolution_scales(opt)

    # Prepare the output folder before any model state is created, so config and logs match this run.
    prepare_output_and_logger(dataset, runtime_args)

    # Build shared objects in dependency order: config context, Gaussian model, scene, then optimizer state.
    training_context = build_training_context(dataset, opt, pipe, runtime_args)
    gaussians = build_gaussian_model_3dgs(dataset, opt)
    scene = Scene(dataset, gaussians, load_iteration=30000, resolution_scales=resolution_scales)
    

    training_context = attach_scene_and_gaussians_to_context(training_context, scene, gaussians)
    gaussians.training_setup(opt)

    # BTS visibility labels: gan ngay sau khi Gaussian da duoc load tu checkpoint 30k
    from vai.bts_visibility import compute_gaussian_labels
    bts_labels = compute_gaussian_labels(csv_path=runtime_args.bts_scores_csv)
    gaussians.set_bts_labels(bts_labels)

    # Restore checkpoint state before loop state is built, so later stages see the resumed model.
    first_iter = 0
    checkpoint_path = resolve_start_checkpoint_path(dataset.model_path, runtime_args)
    if checkpoint_path is not None:
        model_params, first_iter = load_training_checkpoint(checkpoint_path)
        gaussians.restore(model_params, opt)

    loop_state = build_training_loop_state(
        training_context,
        bool(runtime_args.report_lpips_test),
        bool(runtime_args.report_lpips_train),
    )

    ema_loss_for_log = 0.0
    report_time_seconds = 0.0
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Training progress",
        ncols=runtime_args.progress_bar_width,
        dynamic_ncols=False,
    )
    active_training_start = synchronized_timestamp()

    try:
        first_iter += 1
        empty_cache_interval = int(getattr(runtime_args, "empty_cache_interval", 0))
        for iteration in range(first_iter, opt.iterations + 1):
            if empty_cache_interval > 0 and iteration % empty_cache_interval == 0:
                torch.cuda.empty_cache()
            # Update time-based settings, such as SH degree and method-specific training cadence.
            update_3dgs_training_schedule(training_context, gaussians, iteration)

            # Doi camera pool dung tai cac moc 1/4, 1/2 va full resolution.
            update_training_resolution(training_context, loop_state, iteration)

            # Pick one training camera and build gradients from rendering plus all active loss terms.
            viewpoint_cam = sample_training_viewpoint(
                scene,
                loop_state["viewpoint_stack"],
                loop_state["training_resolution_scale"],
                loop_state["viewpoint_repeat_counts"],
            )
            optimization_outputs = run_3dgs_optimization_method(
                training_context,
                gaussians,
                loop_state,
                iteration,
                viewpoint_cam,
            )
            loss = optimization_outputs["loss"]
            render_state = optimization_outputs["render_state"]
            radii = render_state["radii"]

            with torch.no_grad():
                # After gradients are ready, run side-effect stages that should not be tracked by autograd.
                ema_loss_for_log = update_training_progress_bar(
                    progress_bar,
                    iteration,
                    ema_loss_for_log,
                    loss.item(),
                    gaussians.get_xyz.shape[0],
                )
                # Apply optimizer steps only on iterations allowed by the current method schedule.
                run_3dgs_parameter_update_method(training_context, gaussians, iteration, radii)
                # Save images, checkpoints, and evaluation logs at configured milestones.
                report_time_seconds += run_log_and_save_stage(
                    training_context,
                    gaussians,
                    loop_state,
                    iteration,
                )
                # Grow, split, relocate, or rebuild Gaussians according to the selected method.
                run_3dgs_densification_method(training_context, gaussians, iteration, render_state)
                # Remove low-importance Gaussians after densification so the model stays within budget.
                run_3dgs_pruning_method(training_context, gaussians, iteration)
    finally:
        progress_bar.close()

    pure_training_seconds = synchronized_timestamp() - active_training_start - report_time_seconds
    return max(pure_training_seconds, 0.0)


if __name__ == "__main__":
    lp, op, pp, args = parse_training_arguments(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Split the flat argparse namespace into model, optimization, and renderer parameter groups.
    model_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)
    try:
        print("Optimizing " + args.model_path)
        pure_training_seconds = training(model_args, opt_args, pipe_args, args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise SystemExit(130)
    finalize_training_parameters(model_args.model_path, pure_training_seconds)

    print("\nTraining complete.")
