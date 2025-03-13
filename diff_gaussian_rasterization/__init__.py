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

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
from torch.optim.optimizer import Optimizer
from dataclasses import dataclass
# from scene import GaussianModel

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
                num_rendered, num_residuals, num_buckets, color, invdepths, radii, clamped, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_residuals, num_buckets, color, invdepths, radii, clamped, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_residuals = num_residuals
        ctx.num_buckets = num_buckets
        ctx.sparse_J = sparse_J
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, clamped, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_residuals = ctx.num_residuals
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        sparse_J = ctx.sparse_J
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, clamped, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer = ctx.saved_tensors

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
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, J_values, J_indices, p_sum, cov3D= _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, J_values, J_indices, p_sum, cov3D = _C.rasterize_gaussians_backward(*args)

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
        sparse_J.raster_settings = raster_settings
        sparse_J.clamped = clamped
        sparse_J.cov3D = cov3D
        # sparse_J.radii = radii

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
    raster_settings: GaussianRasterizationSettings
    clamped: torch.Tensor
    cov3D: torch.Tensor

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
            # print(f'Opt.step: param: {group["name"]}, size: {param.grad}')
            # if (group['name'] == 'f_dc'):
            #     print(param)

            if (group['name'] == 'rotation'):
                print(param)
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)


            # _C.GN([params], [params.grad], ..)


# Add Gauss Newton
class GaussNewton(Optimizer):
    def __init__(self, params, step_alpha=0.7, step_gamma=1, sparse_jacobian: SparseJacobianData = None):
        assert sparse_jacobian is not None
        defaults = dict(step_alpha=step_alpha, step_gamma=step_gamma, sparse_jacobian=sparse_jacobian )
        super(GaussNewton, self).__init__(params=params, defaults=defaults)

    @torch.no_grad()
    def step(self, visibility, loss_residuals, radii, gaussian_model):

        step_gamma = self.defaults['step_gamma']
        step_alpha = self.defaults['step_alpha']
        sparse_jacobian = self.defaults['sparse_jacobian']


        means3D_group = self.find_parameter_group('xyz')
        scales_group  = self.find_parameter_group('scaling')
        rotations_group = self.find_parameter_group('rotation')
        opacities_group = self.find_parameter_group('opacity')
        f_dc_group = self.find_parameter_group('f_dc')
        f_rest_group = self.find_parameter_group('f_rest')

        means3D = means3D_group['params'][0]
        scales  = torch.exp(scales_group['params'][0].detach())
        # rotations = torch.nn.functional.normalize(rotations_group['params'][0].detach())
        rotations = rotations_group['params'][0].detach()
        opacities = torch.sigmoid(opacities_group['params'][0].detach())
        f_dc = f_dc_group['params'][0]
        f_rest = f_rest_group['params'][0]

        # print(rotations)
        # print(gaussian_model._rotation)
        # print(gaussian_model.get_rotation)

        # raise Exception('heeheh')

        # means3D = gaussian_model.get_xyz
        # scales  = gaussian_model.get_scaling
        # rotations = gaussian_model.get_rotation
        # opacities = gaussian_model.get_opacity
        # f_dc = gaussian_model.get_features_dc
        # f_rest = gaussian_model.get_features_rest

        # print(scales)

        

        print('means3D: ', means3D.shape ,'\n scales: ', scales.shape ,'\n rotation: ', rotations.shape ,'\n opacity: ', opacities.shape ,'\n f_dc: ', f_dc.shape ,'\n f_rest: ', f_rest.shape)


        param_grads = []
        N = 0 # number of parameters
        M = sparse_jacobian.num_residuals # number of residuals 
        for group in self.param_groups:
            # print(f'Opt.step: param: {group["name"]}, size: {group["params"][0].shape}')
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
        #print("Sparse J values: ", sparse_jacobian.values)
        #raise Exception("Work in progress")
    
        # print(sparse_jacobian)
        # print(sparse_jacobian.cov3D)
        # print(sparse_jacobian.clamped.view(-1).sum())
    
    

        # print(f'N: {N}, M: {M}')
        # print(f'J size: {J.size()}, device: {J.device}')
        # print(f'delta size: {delta.size()}, device: {delta.device}')

        # b = -1.0 * (J.T * loss)

        # print(f'b size: {b.size()}, device: {b.device}')


        # print(f'#1 grad ptr: {hex(param_grads[0].data_ptr())}')
        # print(f'J ptr: {hex(J.data_ptr())}')

        # return

        # test params

        # print(visibility)
        # if(not visibility.any()):
        #     print('no gaussian visible')
        #     return


        max_coeffs= 0
        if(f_rest.size(0) != 0):
            max_coeffs = f_rest.size(1)

        P = means3D.size(0)
        D = sparse_jacobian.raster_settings.sh_degree
        width = sparse_jacobian.raster_settings.image_width
        height = sparse_jacobian.raster_settings.image_height

        # print(sparse_jacobian.values)
        # print(P, D, max_coeffs)
        
        _C.gaussNewtonUpdate(
            P, D, max_coeffs, width, height, # max_coeffs = M
            means3D,
            radii,
            f_dc,
            f_rest,
            sparse_jacobian.clamped,
            opacities,
            scales, 
            rotations,
            sparse_jacobian.raster_settings.scale_modifier,
            sparse_jacobian.cov3D,
            sparse_jacobian.raster_settings.viewmatrix,
            sparse_jacobian.raster_settings.projmatrix,
            sparse_jacobian.raster_settings.tanfovx, 
            sparse_jacobian.raster_settings.tanfovy,
            sparse_jacobian.raster_settings.campos, 
            sparse_jacobian.raster_settings.antialiasing,

            delta, 
            sparse_jacobian.values, 
            sparse_jacobian.indices, 
            sparse_jacobian.p_sum, 
            loss_residuals, 
            step_gamma, 
            step_alpha, 
            visibility, 
            N, 
            M, 
            sparse_jacobian.num_entries)


        # num_params_per_gaussian = 59
        # num_gaussians = P
        # print(P)

        # r = torch.rand((num_gaussians, 15, 3))
        # # r = torch.rand((num_gaussians, 3))

        # delta = [i for i in range(num_params_per_gaussian*num_gaussians)]
        # delta = torch.tensor([delta], device=torch.device('cuda'))
        # print(delta)
        
        # delta = delta.view(num_gaussians,59)

        # print(delta.shape)
        # print(delta)

        # delta = delta.T
        # print(delta)
        # print(delta.reshape(-1)[0:r.numel()].view(-1, num_gaussians).T.view(r.shape))

        # torch.cuda.synchronize()
        # delta = delta.cpu()
        # print(torch.nonzero(delta))
        # print(delta)

        delta = delta.view(P,59)
        delta = delta.T
        # delta = delta.contiguous()
        # print(delta.shape)
        # print(delta.shape, delta.dtype, delta.is_contiguous())
        # if torch.isnan(delta).any():
        #     print("NaN detected in delta!")
        # if torch.isinf(delta).any():
        #     print("Inf detected in delta!")
        # print(delta.shape)
        # print(torch.nonzero(delta))
        # print(torch.nonzero(torch.tensor([0,0,0,1,0,2])))
        # print(delta)
        # print('{}\n{}\n{}\n{}\n{}\n'.format(sparse_jacobian.raster_settings.viewmatrix,
        #     sparse_jacobian.raster_settings.projmatrix,
        #     sparse_jacobian.raster_settings.tanfovx, 
        #     sparse_jacobian.raster_settings.tanfovy,
        #     sparse_jacobian.raster_settings.campos))
        delta = delta.reshape(-1)
        # if torch.nonzero(delta).any():
        #     print('non zero elements')

        

        # offset = 0
        # with torch.no_grad():
        #     for group in self.param_groups:
        #         assert len(group["params"]) == 1, "more than one tensor in group"
        #         param = group["params"][0]
        #         if param.grad is None:
        #             continue

        #         numel = param.numel()
        #         print('Name: ', group['name'])
        #         # param.add(delta.view(-1)[offset:offset + numel].view(param.shape))
        #         # param.add(delta[offset:offset + numel].view(-1, P).T.view(param.shape))
        #         # print(param)
        #         # r = (torch.rand(param.shape,  device=torch.device('cuda'))) * 0.01
        #         # param += param.add(r)
        #         # param += r
        #         param.data += delta[offset:offset + numel].view(-1, P).T.view(param.shape)

        #         # print(param)
        #         # print(param)
        #         # param.add(0.1)
        #         # torch.add()
        #         offset += numel




        offset = 0
        with torch.no_grad():
            for group in [means3D_group, scales_group, rotations_group, opacities_group, f_dc_group, f_rest_group]:
                assert len(group["params"]) == 1, "more than one tensor in group"
                param = group["params"][0]
                if param.grad is None:
                    continue

                numel = param.numel()
                # print('Name: ', group['name'])
                # if(group['name'] == 'f_dc'):
                #     print(param)
                #     print('update step:')
                #     print(delta[offset:offset + numel].view(-1, P).T.view(param.shape))
                # print(delta[offset:offset + numel].view(-1, P).T.view(param.shape))
                # print(param)
                param.data += delta[offset:offset + numel].view(-1, P).T.view(param.shape)
                # print('AFTER')
                # print(param)

                offset += numel
                
        # raise Exception("Work in progress")
        

    def find_parameter_group(self, name):
        for group in self.param_groups:
            if group['name'] == name:
                return group
            
        raise Exception(f'could not find parameter group of name: {name}')
