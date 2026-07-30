"""
Main 3DGS Gaussian model class.

The class stores trainable Gaussian tensors and combines mixins for optimizer
management, PLY I/O, SH scheduling, and structure changes. Public properties
return activated values such as positive scales, normalized rotations, and
sigmoid opacity while the raw tensors remain optimizable parameters.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from scene.gaussian_model_densification import GaussianModelDensificationMixin
from scene.gaussian_model_io import GaussianModelIOMixin
from scene.gaussian_model_optimizer import GaussianModelOptimizerMixin
from scene.gaussian_model_sh import GaussianModelSHMixin
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import BasicPointCloud


class GaussianModel(
    GaussianModelOptimizerMixin,
    GaussianModelSHMixin,
    GaussianModelIOMixin,
    GaussianModelDensificationMixin,
):
    """
        Trainable Gaussian collection used by rendering, training, and saving code.
    """

    def setup_functions(self) -> None:
        """
            Register activation and inverse-activation functions for raw parameters.
        """
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree: int, optimizer_type: str = "default") -> None:
        """
            Create an empty Gaussian model with the requested SH capacity.
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        normalized_optimizer_type = str(optimizer_type).lower()
        if normalized_optimizer_type not in {"default", "ours_adam"}:
            raise ValueError(
                "3DGS currently supports only optimizer_type in {default, ours_adam}; got {}".format(optimizer_type)
            )
        self.optimizer_type = normalized_optimizer_type
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.noise_lr = 0.0
        self.mcmc_noise_opacity_sharpness = 100.0
        self.use_mcmc_initialization = False
        self.initial_opacity = 0.1
        self.spatial_lr_scale = 0
        #VAR
        self._bts_label = torch.empty(0)   # << thêm: int8, 0=background 1=boundary 2=BTS
        ###
        self.setup_functions()

    def capture(self) -> tuple[Any, ...]:
        """
            Pack model tensors and optimizer state into the checkpoint tuple format.
        """
        return (
            self.active_sh_degree,
            self.max_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            None,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            None,
        )







    @property
    def get_scaling(self) -> torch.Tensor:
        """
            Return positive Gaussian scales from raw log-scale parameters.
        """
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self) -> torch.Tensor:
        """
            Return normalized quaternion rotations.
        """
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self) -> torch.Tensor:
        """
            Return raw Gaussian centers.
        """
        return self._xyz
    
    @property
    def get_features_dc(self) -> torch.Tensor:
        """
            Return DC color SH coefficients.
        """
        return self._features_dc
    
    @property
    def get_features_rest(self) -> torch.Tensor:
        """
            Return higher-order SH color coefficients.
        """
        return self._features_rest
    
    @property
    def get_opacity(self) -> torch.Tensor:
        """
            Return activated opacity values in `[0, 1]`.
        """
        return self.opacity_activation(self._opacity)

    def get_exposure_from_name(self, image_name: str) -> torch.Tensor:
        """
            Return the learned or loaded exposure matrix for one image name.
        """
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    #### VAR
    def set_bts_labels(self, labels) -> None:
        """
            Gắn per-Gaussian label (BTS/boundary/background) từ protocol visibility.
            Bắt buộc gọi ngay sau load_ply(), khi N khớp tuyệt đối.
        """
        assert labels.shape[0] == self._xyz.shape[0], \
            f"label size {labels.shape[0]} != gaussian count {self._xyz.shape[0]}"
        self._bts_label = torch.as_tensor(labels, dtype=torch.int8, device="cuda")

    @property
    def is_background(self) -> torch.Tensor:
        """
            Bool mask (N,): True o Gaussian background, dung de freeze gradient.
        """
        if self._bts_label.numel() == 0:
            return torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        return self._bts_label == 0















    def prune_points(self, mask: torch.Tensor) -> None:
        """
            Remove Gaussians selected by `mask` and prune matching optimizer state.
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        #VAR
        if self._bts_label.numel() > 0:                              
            self._bts_label = self._bts_label[valid_points_mask]     