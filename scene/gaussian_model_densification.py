"""
GaussianModel structure-changing operations.

These methods modify the number of Gaussians or reset their trainable tensors.
They also keep optimizer state aligned with the new tensors, so training can
continue after clone, split, prune, relocation, or MiniGS reinitialization.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from simple_knn._C import distCUDA2
from scene.methods.mcmc_ops import add_new_gs as _mcmc_add_new_gs
from scene.methods.mcmc_ops import apply_mcmc_noise as _mcmc_apply_noise
from scene.methods.mcmc_ops import relocate_gs as _mcmc_relocate_gs
from utils.general_utils import RGB2SH, build_rotation, inverse_sigmoid


class GaussianModelDensificationMixin:
    """
        Mixin that adds structure updates to GaussianModel.
    """

    def reset_opacity(self, min_opacity: float = 0.01) -> None:
        """
            Clamp all activated opacities to a maximum value and update optimizer state.
        """
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        reset_params: bool = True,
        new_bts_label: torch.Tensor | None = None,
    ) -> None:
        """
            Append newly created Gaussians to all parameter tensors.

            The optimizer receives the new tensors first, then the model fields
            are rebound to the optimizer-owned parameters. Gradient accumulators
            are reset because the Gaussian count has changed.
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # VAR
        if self._bts_label.numel() > 0:                                          # << thêm
            if new_bts_label is None:                                            # << thêm
                new_bts_label = torch.zeros(new_xyz.shape[0], dtype=torch.int8, device="cuda")  # << thêm (fallback: coi Gaussian mới không có label là background)
            self._bts_label = torch.cat((self._bts_label, new_bts_label))        # << thêm


        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def densify_and_split(self, grads: torch.Tensor, grad_threshold: float, scene_extent: float, N: int = 2) -> None:
        """
            Split large high-gradient Gaussians using the original 3DGS rule.

            Selected parents generate `N` children sampled from their local
            Gaussian shape, then the original parents are pruned away.
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_bts_label = (self._bts_label[selected_pts_mask].repeat(N)                     # << thêm
                          if self._bts_label.numel() > 0 else None)                        # << thêm

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_bts_label=new_bts_label,                                                  # << thêm
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float, scene_extent: float) -> None:
        """
            Clone small high-gradient Gaussians using the original 3DGS rule.

            Unlike split, cloning keeps the original Gaussian and appends an
            identical copy for compact regions that still need more detail.
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_bts_label = (self._bts_label[selected_pts_mask]                                # << thêm
                          if self._bts_label.numel() > 0 else None)                        # << thêm

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_bts_label=new_bts_label,                                                  # << thêm
        )

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
    ) -> None:
        """
            Run original 3DGS densification, then prune Gaussians that meet the pruning rules.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_prune_split(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        mask: torch.Tensor,
    ) -> None:
        """
            Reuse the original clone branch, then merge the MiniGS blur mask during split to match the reference densification behavior.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_mask(grads, max_grad, extent, mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_split_mask(self, grads: torch.Tensor, grad_threshold: float, scene_extent: float, mask: torch.Tensor, N: int = 2) -> None:
        """
            Add the MiniGS blur mask to the original split candidates, so highly blurred areas can split even when gradients are weak.
        """
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device="cuda")
        padded_mask[:mask.shape[0]] = mask
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_bts_label = (self._bts_label[selected_pts_mask].repeat(N)                      # << thêm
                          if self._bts_label.numel() > 0 else None)                        # << thêm

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_bts_label=new_bts_label,                                                  # << thêm
        )
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)

    def add_densification_stats(self, viewspace_point_tensor: torch.Tensor, update_filter: torch.Tensor) -> None:
        """
            Accumulate normal screen-space gradient magnitudes for visible Gaussians.
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_abs(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
        sync_default_accumulator: bool = False,
    ) -> None:
        """
            Accumulate `abs(dx)` and `abs(dy)` written by CUDA backward for ImprovedGS densification scoring. When `sync_default_accumulator=True`, also update `xyz_gradient_accum` so ABSGS can reuse the baseline `densify_and_prune()` path.
        """
        grad_values = torch.norm(viewspace_point_tensor.grad[update_filter, 2:4], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += grad_values
        if sync_default_accumulator:
            self.xyz_gradient_accum[update_filter] += grad_values
        self.denom[update_filter] += 1

    def relocate_gs(self, dead_mask: torch.Tensor | None = None) -> None:
        """
            Delegate MCMC relocation of low-opacity Gaussians to the method helper.
        """
        _mcmc_relocate_gs(self, dead_mask)

    def add_new_gs(self, cap_max: int) -> int:
        """
            Delegate MCMC Gaussian insertion and return the number of added points.
        """
        return _mcmc_add_new_gs(self, cap_max)

    def apply_mcmc_noise(self, xyz_lr: float) -> None:
        """
            Add MCMC SGLD noise to positions through the method helper.
        """
        _mcmc_apply_noise(self, xyz_lr)

    def only_prune(self, min_opacity: float, percent: bool = False) -> None:
        """
            Provide a lightweight pruning entry point that RAP and GNS can call without running densification.
        """
        if self.get_opacity.numel() == 0:
            return
        if percent:
            opacity_array = self.get_opacity.detach().flatten()
            threshold = torch.quantile(opacity_array, float(min_opacity))
            prune_mask = (self.get_opacity < threshold).squeeze()
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        if prune_mask.any():
            self.prune_points(prune_mask)

    def densify_and_prune_improved(
        self,
        scores: torch.Tensor | None,
        min_opacity: float,
        budget: int,
        opt: Any,
        iteration: int,
        scene_extent: float,
    ) -> None:
        """
            Select candidate Gaussians with absolute screen-space gradients, then run long-axis split under the budget. In late training, fall back to gradient scores and relax the threshold while the budget is not full, matching the reference ImprovedGS behavior.
        """
        grad_values = self.xyz_gradient_accum_abs / self.denom
        grad_values[grad_values.isnan()] = 0.0

        min_grad = float(opt.densify_grad_threshold)
        late_densify_iter = max(int(opt.densify_until_iter) - 100, int(opt.densify_from_iter))
        if scores is None or iteration > late_densify_iter:
            scores = grad_values.squeeze(-1)
            if self.get_opacity.shape[0] < budget and iteration > late_densify_iter:
                min_grad = min_grad / 1.5

        grad_qualifiers = torch.where(torch.norm(grad_values, dim=-1) >= min_grad, True, False)
        total_candidates = int(grad_qualifiers.sum().item())
        current_points = int(self.get_xyz.shape[0])
        current_budget = min(int(budget), total_candidates + current_points)
        split_budget = current_budget - current_points
        if split_budget > 0:
            if bool(getattr(opt, "use_las", True)):
                self.long_axis_split(scores, split_budget, grad_qualifiers, opt.split_distance, opt.opacity_reduction)
            else:
                self.densify_and_split(grad_values, min_grad, float(scene_extent))

        if iteration < late_densify_iter:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
            if prune_mask.any():
                self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def long_axis_split(
        self,
        scores: torch.Tensor,
        budget: int,
        filter_mask: torch.Tensor,
        split_distance: float,
        opacity_reduction: float,
    ) -> int:
        """
            Sample candidates by importance, create two child Gaussians by moving along the longest axis in both directions, then adjust scale and opacity for the ImprovedGS split.
        """
        if budget <= 0 or scores.numel() == 0 or not torch.any(filter_mask):
            return 0

        padded_importance = torch.zeros((self.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
        padded_importance[:scores.shape[0]] = scores.detach().float().clamp_min(0)
        padded_importance[~filter_mask] = 0
        positive_count = int((padded_importance > 0).sum().item())
        if positive_count == 0:
            return 0

        budget = min(int(budget), positive_count)
        selected_indices = torch.multinomial(padded_importance, budget, replacement=False)
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=torch.bool)
        selected_pts_mask[selected_indices] = True

        stds = self.get_scaling[selected_pts_mask]
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        axis_mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)
        axis_offsets = stds * axis_mask * 3.0 * float(split_distance)
        axis_offsets = torch.cat([axis_offsets, -axis_offsets], dim=0)

        rotation_mats = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        parent_xyz = self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_xyz = torch.bmm(rotation_mats, axis_offsets.unsqueeze(-1)).squeeze(-1) + parent_xyz

        split_distance_sq = float(split_distance) * float(split_distance)
        rate_w = max(1.0 - float(split_distance), 1e-6)
        rate_h = math.sqrt(max(1.0 - split_distance_sq, 1e-6))
        new_scales = stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h
        new_scaling = self.scaling_inverse_activation(new_scales)
        new_opacity = inverse_sigmoid(self.get_opacity[selected_pts_mask] * float(opacity_reduction)).repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_bts_label = (self._bts_label[selected_pts_mask].repeat(2)                      # << thêm
                          if self._bts_label.numel() > 0 else None)                        # << thêm

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation,
            new_bts_label=new_bts_label,                                                  # << thêm
        )
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(2 * int(selected_pts_mask.sum().item()), device="cuda", dtype=torch.bool))
        )
        self.prune_points(prune_filter)
        return budget

    def reinitial_pts(self, pts: torch.Tensor, rgb: torch.Tensor) -> None:
        """
            Rebuild the Gaussian set from MiniGS periodic back-sampling results. Keep the current SH settings and exposure parameters so `training_setup()` can be called again later.
        """
        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device="cuda")
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
