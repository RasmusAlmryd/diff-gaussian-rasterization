#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
from torch.optim.optimizer import Optimizer
from dataclasses import dataclass

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    sparse_J
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        sparse_J
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        sparse_J
    ):

        # Restructure arguments the way that the C++ lib expects them
        
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_residuals, num_buckets, color, invdepths, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_residuals, num_buckets, color, invdepths, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_residuals = num_residuals
        ctx.num_buckets = num_buckets
        ctx.sparse_J = sparse_J
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_residuals = ctx.num_residuals
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        sparse_J = ctx.sparse_J
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer = ctx.saved_tensors

        # print(f"num rendered: {num_rendered}, gaussians: {means3D.size()}, diff {means3D.size(0)-num_rendered}")
        # print(f'grad_out_color size: {grad_out_color.size()}, grad_out_depth size: {grad_out_depth.size()}')
        # _C.sum

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                dc,
                sh, 
                grad_out_depth, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                num_residuals,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
                residualBuffer,
		        raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, J_values, J_indices, p_sum= _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, J_values, J_indices, p_sum = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_dc,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
        )

        sparse_J.values = J_values
        sparse_J.indices = J_indices
        sparse_J.p_sum = p_sum
        sparse_J.num_gaussians = means3D.shape[0]
        sparse_J.num_residuals = raster_settings.image_width * raster_settings.image_height
        sparse_J.num_entries = num_residuals

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

@dataclass
class SparseJacobianData:
    values: torch.Tensor
    indices: torch.Tensor
    p_sum: torch.Tensor
    num_gaussians: int
    num_residuals: int
    num_entries: int


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings, sparse_jacobian):
        super().__init__()
        self.raster_settings = raster_settings
        self.sparse_jacobian = sparse_jacobian
        # print("Initialized GaussianRasterizer")

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings
        sparse_jacobian = self.sparse_jacobian

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            sparse_jacobian,
        )

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            print(f'Opt.step: param: {group["name"]}, size: {param.grad}')
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)


            # _C.GN([params], [params.grad], ..)


# Add Gauss Newton
class GaussNewton(Optimizer):
    def __init__(self, params, step_alpha=0.7, step_gamma=1, sparse_jacobian: SparseJacobianData = None):
        assert sparse_jacobian is not None
        defaults = dict(step_alpha=step_alpha, step_gamma=step_gamma, sparse_jacobian=sparse_jacobian )
        super(GaussNewton, self).__init__(params=params, defaults=defaults)

    @torch.no_grad()
    def step(self, visibility, loss):

        step_gamma = self.defaults['step_gamma']
        step_alpha = self.defaults['step_alpha']
        sparse_jacobian = self.defaults['sparse_jacobian']


        param_grads = []
        N = 0 # number of parameters
        M = sparse_jacobian.num_residuals # number of residuals 
        for group in self.param_groups:

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization from adam, try to fix this to get above 3000 iterations w/o crash
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                


            # param_grads.append(param.grad.flatten())
            N += param.numel()

        if N == 0:
            return

        # J = torch.cat(param_grads).view(-1, M)

        delta = torch.zeros(N, dtype=torch.float32, device=torch.device('cuda'))

        print(sparse_jacobian)
        raise Exception("Work in progress")

        # print(f'N: {N}, M: {M}')
        # print(f'J size: {J.size()}, device: {J.device}')
        # print(f'delta size: {delta.size()}, device: {delta.device}')

        # b = -1.0 * (J.T * loss)

        # print(f'b size: {b.size()}, device: {b.device}')


        # print(f'#1 grad ptr: {hex(param_grads[0].data_ptr())}')
        # print(f'J ptr: {hex(J.data_ptr())}')

        # return

        _C.gaussNewtonUpdate(delta, sparse_jacobian.values, sparse_jacobian.indices, sparse_jacobian.p_sum, step_gamma, step_alpha, visibility, N, M, sparse_jacobian.num_entries)

        offset = 0
        with torch.no_grad():
            for group in self.param_groups:
                assert len(group["params"]) == 1, "more than one tensor in group"
                param = group["params"][0]
                if param.grad is None:
                    continue

                numel = param.numel()
                param.add(delta.view(-1)[offset:offset + numel].view(param.shape))
                # param.add(0.1)
                # torch.add()
                offset += numel
