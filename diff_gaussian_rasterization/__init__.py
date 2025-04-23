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
            raster_settings.num_views,
            raster_settings.view_index,
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
                raster_settings.num_views,
                raster_settings.view_index,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, J_values, J_indices, p_sum, cov3D, conic_o = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, J_values, J_indices, p_sum, cov3D, conic_o = _C.rasterize_gaussians_backward(*args)

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

        # print(cov3D)
        # print(scales)
        # print(rotations)

        if(raster_settings.view_index == 0):
            sparse_J.raster_settings.clear()

        sparse_J.values = J_values
        sparse_J.indices = J_indices
        sparse_J.p_sum = p_sum
        sparse_J.num_gaussians = means3D.shape[0]
        sparse_J.num_residuals = raster_settings.image_width * raster_settings.image_height
        sparse_J.num_entries = num_residuals
        sparse_J.raster_settings.append(raster_settings)
        sparse_J.clamped = clamped
        sparse_J.cov3D = cov3D
        sparse_J.conic_o = conic_o

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
    num_views: int
    view_index: int

@dataclass
class SparseJacobianData:
    values: torch.Tensor
    indices: torch.Tensor
    p_sum: torch.Tensor
    num_gaussians: int
    num_residuals: int
    num_entries: int
    raster_settings: list[GaussianRasterizationSettings]
    clamped: torch.Tensor
    cov3D: torch.Tensor
    conic_o: torch.Tensor

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

        # print(means3D,
        #     means2D,
        #     dc,
        #     shs,
        #     colors_precomp,
        #     opacities,
        #     scales, 
        #     rotations,
        #     cov3D_precomp,
        #     raster_settings)
        

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

            # if (group['name'] == 'rotation'):
            #     print(param)
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)

            # if (group['name'] == 'f_dc' or group['name'] == 'scaling'):
            #     print(group['name'])
            #     print(param)
            #     print(param.grad)
                

            # _C.GN([params], [params.grad], ..)

        # raise Exception('end of step adam')


# Add Gauss Newton
class GaussNewton(Optimizer):
    def __init__(self, params, step_alpha=0.7, step_gamma=1, sparse_jacobian: SparseJacobianData = None):
        assert sparse_jacobian is not None
        defaults = dict(step_alpha=step_alpha, step_gamma=step_gamma, sparse_jacobian=sparse_jacobian )
        super(GaussNewton, self).__init__(params=params, defaults=defaults)

    @torch.no_grad()
    def step(self, visibility, view_residuals, view_radii, gaussian_model, iteration, gt_images):

        print("step")
        
        step_gamma = self.defaults['step_gamma']
        step_alpha = self.defaults['step_alpha']
        sparse_jacobian = self.defaults['sparse_jacobian']


        means3D_group = self.find_parameter_group('xyz')
        scales_group  = self.find_parameter_group('scaling')
        rotations_group = self.find_parameter_group('rotation')
        opacities_group = self.find_parameter_group('opacity')
        f_dc_group = self.find_parameter_group('f_dc')
        f_rest_group = self.find_parameter_group('f_rest')

        means3D = means3D_group['params'][0].detach()
        scales  = torch.exp(scales_group['params'][0].detach())
        rotations = torch.nn.functional.normalize(rotations_group['params'][0].detach())
        # rotations = rotations_group['params'][0].detach()
        opacities = torch.sigmoid(opacities_group['params'][0].detach())
        f_dc = f_dc_group['params'][0].detach()
        f_rest = f_rest_group['params'][0].detach()

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

        # print(means3D)
        # print(scales)
        # print(rotations)
        # print(opacities)
        # print(f_dc)
        # print(f_rest)

        # print(iteration)
        # if iteration > 1:

        #     return

        

        # print('means3D: ', means3D.shape ,'\n scales: ', scales.shape ,'\n rotation: ', rotations.shape ,'\n opacity: ', opacities.shape ,'\n f_dc: ', f_dc.shape ,'\n f_rest: ', f_rest.shape)


        param_grads = []
        N = 0 # number of parameters
        M = sparse_jacobian.num_residuals # number of residuals 
        for group in self.param_groups:
            # print(f'Opt.step: param: {group["name"]}, size: {group["params"][0].shape}')
            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue
            
            # print(group['name'])
            # # print(param.grad)
            # print(param)

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

        view_residuals = torch.stack(view_residuals, dim=0)
        view_radii = torch.stack(view_radii, dim=0)

        # print(view_residuals)
        # # print(view_residuals.reshape(-1))
        # print(view_residuals.shape)

        delta = torch.zeros(N, dtype=torch.float32, device=torch.device('cuda'))


        print('residual derivatives sum:', sparse_jacobian.values.sum())
        # raise Exception("Work in progress")

        # print('residual sum: ', loss_residuals.sum())

        # settings = []
        # num_views = len(sparse_jacobian.raster_settings)
        # for view_index in range(num_views):
        #     setting = _C.Raster_settings(
        #         sparse_jacobian.raster_settings[view_index].image_height,
        #         sparse_jacobian.raster_settings[view_index].image_width ,
        #         sparse_jacobian.raster_settings[view_index].tanfovx ,
        #         sparse_jacobian.raster_settings[view_index].tanfovy ,
        #         sparse_jacobian.raster_settings[view_index].bg,
        #         sparse_jacobian.raster_settings[view_index].scale_modifier ,
        #         sparse_jacobian.raster_settings[view_index].viewmatrix ,
        #         sparse_jacobian.raster_settings[view_index].projmatrix,
        #         sparse_jacobian.raster_settings[view_index].sh_degree ,
        #         sparse_jacobian.raster_settings[view_index].campos ,
        #         sparse_jacobian.raster_settings[view_index].prefiltered ,
        #         sparse_jacobian.raster_settings[view_index].debug ,
        #         sparse_jacobian.raster_settings[view_index].antialiasing ,
        #         sparse_jacobian.raster_settings[view_index].num_views,
        #         sparse_jacobian.raster_settings[view_index].view_index,
        #         )
            
        #     settings.append(setting)
        #     print( sparse_jacobian.raster_settings[view_index].viewmatrix)

    

        view_matrices = []
        proj_matrices = []
        campos = []
        num_views = len(sparse_jacobian.raster_settings)
        for view_index in range(num_views):
            view_matrices.append(sparse_jacobian.raster_settings[view_index].viewmatrix)
            proj_matrices.append(sparse_jacobian.raster_settings[view_index].projmatrix)
            campos.append( sparse_jacobian.raster_settings[view_index].campos)

        view_matrices = torch.stack(view_matrices, dim=0)
        proj_matrices = torch.stack(proj_matrices, dim=0)
        campos = torch.stack(campos, dim=0)
        print(view_matrices)
        # print(proj_matrices)
        # print(campos)
        

        max_coeffs= 0
        if(f_rest.size(0) != 0):
            max_coeffs = f_rest.size(1)

        P = means3D.size(0)
        D = sparse_jacobian.raster_settings[0].sh_degree
        width = sparse_jacobian.raster_settings[0].image_width
        height = sparse_jacobian.raster_settings[0].image_height

        # print(sparse_jacobian.values.isnan().any())
        

        _C.gaussNewtonUpdate(
            P, D, max_coeffs, width, height, # max_coeffs = M
            means3D,
            view_radii,
            f_dc,
            f_rest,
            sparse_jacobian.clamped,
            opacities,
            scales, 
            rotations,
            sparse_jacobian.raster_settings[0].scale_modifier,
            sparse_jacobian.cov3D,
            sparse_jacobian.conic_o,
            view_matrices, #sparse_jacobian.raster_settings[0].viewmatrix,
            proj_matrices, #sparse_jacobian.raster_settings[0].projmatrix,
            sparse_jacobian.raster_settings[0].tanfovx, 
            sparse_jacobian.raster_settings[0].tanfovy,
            campos, #sparse_jacobian.raster_settings[0].campos, 
            sparse_jacobian.raster_settings[0].antialiasing,
            # settings,
            num_views,

            delta, 
            sparse_jacobian.values, 
            sparse_jacobian.indices, 
            sparse_jacobian.p_sum, 
            view_residuals, 
            step_gamma, 
            step_alpha, 
            visibility, 
            N, 
            M, 
            sparse_jacobian.num_entries)
        
        # raise Exception("Work in progress")


        delta = delta.view(P,59)
        delta = delta.T
        print(delta)

        delta = delta.reshape(-1)




        if delta.isnan().any() or torch.all(delta == 0):
            print('not acceptable delta')
            del sparse_jacobian.values 
            del sparse_jacobian.indices
            del sparse_jacobian.p_sum 
            # raise Exception('stooooop')
            sparse_jacobian.raster_settings.clear()

            return
        
        max_scale_exp = 5

        def error(delta, alpha):
            with torch.no_grad():
                error = 0
                for view_index in range(num_views):
                    offset = 0
                    new_means3D = torch.clone(means3D_group['params'][0])
                    new_scales = torch.clone(scales_group['params'][0])
                    new_rotations = torch.clone(rotations_group['params'][0])
                    new_opacities = torch.clone(opacities_group['params'][0])
                    new_f_dc = torch.clone(f_dc_group['params'][0])
                    new_f_rest = torch.clone(f_rest_group['params'][0])

                    new_params = [new_means3D,new_scales,new_rotations,new_opacities,new_f_dc,new_f_rest]

                    for idx, param in enumerate([means3D,scales,rotations,opacities,f_dc,f_rest]):
                        numel = param.numel()
                        new_params[idx] += alpha * delta[offset:offset + numel].view(-1, P).T.view(param.shape)
                        offset += numel
                        # print(new_params[idx])

                    # new_scales = torch.exp(torch.clamp(new_scales, None, max_scale_exp))

                    new_scales = new_scales.clamp(None, max_scale_exp)
                    new_scales = torch.exp(new_scales)
                    new_rotations = torch.nn.functional.normalize(new_rotations)
                    new_opacities = torch.sigmoid(new_opacities)
                    
                    new_means2D = torch.zeros_like(new_means3D, dtype=new_means3D.dtype, requires_grad=True, device="cuda") + 0

                    rendered_image , _, _ = rasterize_gaussians(new_means3D, new_means2D, new_f_dc, new_f_rest, torch.Tensor([]), new_opacities,new_scales, new_rotations, torch.Tensor([]), sparse_jacobian.raster_settings[view_index], None)
                
                    rendered_image = rendered_image.clamp(0, 1)

                    # print("E image sum: ", rendered_image.sum())
                    
                    loss = (gt_images[view_index] - rendered_image) ** 2
                    loss = loss.sum()
                    error += loss.item()

                return error



        mu_prev = float('inf')
        alpha = 1.0
        gamma = 0.7
        while alpha >= 1e-3:
            mu = error(delta, alpha)
            print('alpha: ', alpha)
            print('error: ', mu)
            if mu > mu_prev:
                alpha = alpha/gamma
                break
            alpha *= gamma
            mu_prev = mu
            
        
    
        print('final alpha: ', alpha)



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
                # if(group['name'] == 'scaling'):
                #     param.data += torch.ones_like(param) * 0.1
                #     continue
                # if group['name'] != 'xyz':
                # param.data += delta[offset:offset + numel].view(-1, P).T.view(param.shape)
                param.data += alpha * delta[offset:offset + numel].view(-1, P).T.view(param.shape)


                if(group['name'] == 'scaling'):
                    param.data = torch.clamp(param.data, None, max_scale_exp)
                

                # print('AFTER')
                # print(param)

                offset += numel

        
            # means3D = means3D_group['params'][0]
            # means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0

            # scales = scales_group['params'][0]
            # rotations = rotations_group['params'][0]
            # opacities = opacities_group['params'][0]
            # f_dc = f_dc_group['params'][0]
            # f_rest = f_rest_group['params'][0]

            
            # scales = torch.exp(scales)

            # rotations = torch.nn.functional.normalize(rotations)
            # opacities = torch.sigmoid(opacities)

            # color, _, _ = rasterize_gaussians(means3D, means2D, f_dc, f_rest, torch.Tensor([]), opacities ,scales, rotations, torch.Tensor([]), sparse_jacobian.raster_settings[0], None)
            # print('check color sum: ', color.sum())
                

        sparse_jacobian.raster_settings.clear()
        
        # raise Exception("Work in progress")
        
        

    def find_parameter_group(self, name):
        for group in self.param_groups:
            if group['name'] == name:
                return group
            
        raise Exception(f'could not find parameter group of name: {name}')
