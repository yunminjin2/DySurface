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
import torch
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings as GRS
# from diff_gaussian_rasterization import GaussianRasterizer as GR
# from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gaussian_rasterization_foremost import GaussianRasterizerForemost

from gaussian_splatting.utils.point_utils import depth_to_normal
from gaussian_splatting.scene.sparseGaGS import matrix_to_quaternion

def deform_points(pc, t):
    t = torch.tensor([t]).float().cuda()
    
    pc.get_t(t)
    pc.transformation()
    
    delta_anchor = pc.delta_xyz
    delta_scales = pc.delta_scaling
    delta_rot = pc.six_d

    return delta_anchor, delta_scales, delta_rot
    
        

def simple_render(viewpoint_camera, gs_asset: dict, pipe, bg_color: torch.Tensor, scene=None, scaling_modifier: float = 1.0, return_depth=False, return_normal=False):
    """
    Deformed Gaussian asset을 받아 렌더링을 수행합니다.
    """
    xyz = gs_asset['xyz'].contiguous().cuda()
    scales = gs_asset['scale'].contiguous().cuda()
    rot = gs_asset['rotation'].contiguous().cuda()
    shs = gs_asset['feature'].contiguous().cuda()
    opacity = gs_asset['opacity'].contiguous().cuda()

    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0

    # 3. Contiguous 검증 및 강제 변환
    assert xyz.shape[0] == scales.shape[0] == rot.shape[0] == shs.shape[0] == opacity.shape[0], "Shape mismatch detected!"
    # 2. NaN/Inf 검증
    if torch.isnan(xyz).any() or torch.isinf(xyz).any():
        print("Error: xyz contains NaN/Inf!")
    if torch.isnan(scales).any() or torch.isinf(scales).any():
        print("Error: scales contains NaN/Inf!")
    if torch.isnan(rot).any() or torch.isinf(rot).any():
        print("Error: rot contains NaN/Inf!")
    if torch.isnan(shs).any() or torch.isinf(shs).any():
        print("Error: shs contains NaN/Inf!")
    if torch.isnan(opacity).any() or torch.isinf(opacity).any():
        print("Error: opacity contains NaN/Inf!")

    
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier, # cov3D_precomp 사용 시 내부적으로 무시될 수 있으나 기본값 유지
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1, # 동적 할당
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )
    return_dict = {}
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # [핵심] cov3D_precomp가 전달되면 rasterizer의 CUDA 커널은 scales와 rotations를 무시합니다.
    out = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,    # SVD로 분리하지 않았으므로 None 할당
        rotations=rot, # SVD로 분리하지 않았으므로 None 할당
        cov3D_precomp=None
    )
    
    return_dict.update({
        "render": torch.clamp(out[0], 0, 1),
    })
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    out= rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = torch.ones_like(xyz).cuda(),
        opacities = opacity,
        scales = scales,
        rotations = rot,
    )
    render_alpha = out[0][0:1]
    return_dict.update({"mask": render_alpha})
    
    
    if return_depth:
        depth = (xyz-viewpoint_camera.camera_center).norm(dim=1, keepdim=True).repeat([1,3])

        out = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = depth,
            opacities = opacity,
            scales = scales,
            rotations = rot,
            cov3D_precomp = None)
        

        accumulated_depth = out[0].float().mean(0, keepdim=True)
        valid_depth = render_alpha > 1e-8
        rendered_depth_hand = torch.where(
            valid_depth,
            accumulated_depth / render_alpha.clamp_min(1e-8),
            torch.zeros_like(accumulated_depth),
        )
        render_depth = rendered_depth_hand / (rendered_depth_hand.max() + 1e-8)

        return_dict.update({
            'real_depth': torch.clamp(rendered_depth_hand, -10000, 10000),   # [1, H, W]
            'render_depth': (rendered_depth_hand/(rendered_depth_hand.max() + 1e-8)).repeat(3, 1, 1) # [3, H, W]
        })
        
        if return_normal:
            
            surf_normal = depth_to_normal(viewpoint_camera, render_depth) 
            surf_normal = surf_normal.permute(2, 0, 1)
            surf_normal = ((surf_normal + 1) / 2 * render_alpha)

            return_dict.update({
                'render_depth_normal': torch.clamp(surf_normal, 0.0, 1.0), 
                # 'render_gs_normal': torch.clamp((render_norm + 1) * render_alpha / 2, 0., 1.) # [3, H, W]
            })

    return return_dict

def render_dynamic(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scene=None, transform_scaler = 0.1, scaling_modifier = 1.0, return_scaffold=False, return_asset=False, return_depth=False, return_normal=False,  dynamic_decode=False, retain_grad=False, do_refine=False, refine_delta=None, debug = False, gs_asset=None):
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    anchor = pc.get_xyz
    shs = pc.get_features
    opacity = pc.get_opacity
    mask = opacity > 0
    asset_scale = pc._scaling
    scales = pc.get_scaling
    asset_rot = pc._rotation
    rot = pc.get_rotation
    if gs_asset is not None:
        anchor = gs_asset['xyz']
        shs = gs_asset['feature']
        opacity = gs_asset['opacity']
        scales = pc.scaling_activation(gs_asset['scaling'])
        asset_rot = gs_asset['rotations']
        rot = pc.rotation_activation(asset_rot)

    n_anchor = anchor.shape[0]
    shs = shs.reshape(n_anchor, -1, 3)
    
    return_dict =  {}
    
    out= rasterizer(
        means3D = anchor,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rot,
        cov3D_precomp = None
    )
    
    return_dict.update({
        "canon_render": torch.clamp(out[0], 0, 1),
    })
    
    with torch.no_grad():
        pc.get_t(torch.tensor([pc.canon_t]).float().cuda())
        pc.transformation()

        pred_canon_anchor = pc.get_transformed_xyz
        pred_canon_scales = pc.get_transformed_scaling
        pred_canon_rot = pc.get_transformed_rotation

        out= rasterizer(
            means3D = pred_canon_anchor,
            means2D = screenspace_points,
            shs = shs,
            colors_precomp = None,
            opacities = opacity,
            scales = pred_canon_scales,
            rotations = pred_canon_rot,
            cov3D_precomp = None
        )

        return_dict.update({
            "pred_canon_render": torch.clamp(out[0], 0, 1).detach(),
        })


    if dynamic_decode:
        t = torch.tensor([viewpoint_camera.t]).float().cuda()

        pc.get_t(t)
        pc.transformation()
        
        anchor = pc.get_transformed_xyz
        scales = pc.get_transformed_scaling
        rot = pc.get_transformed_rotation

        asset_scale = pc.o_scaling
        asset_rot = matrix_to_quaternion(pc.o_rotation)
        # asset_rot = pc.o_rotation

    xyz2d, idxs = project_points_to_image(anchor, viewpoint_camera)
    scales = scales.clamp(min=1e-4)
    
    out= rasterizer(
        means3D = anchor,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rot,
        cov3D_precomp = None
    )
    
    rendered_image, radii =  out[0], out[1]


    return_dict.update({
        "render": torch.clamp(rendered_image, 0, 1),
        "idxs": idxs,
        "xyz2d": xyz2d,
        "viewspace_points": screenspace_points,
        "radii": radii,
        "visibility_filter" : radii > 0,
        "selection_mask": mask,
        "scaling": scales,
        "opacity": opacity,
    })
    
    if return_asset:
        return_dict.update({"gs_asset": {
                "xyz": anchor,
                "feature": shs,
                "rotations": asset_rot,
                "scaling": asset_scale,
                "opacity": pc.inverse_opacity_activation(opacity),
            }
        })

    # 2. Rasterizer 초기화 (배경색 적용)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    out= rasterizer(
        means3D = anchor,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = torch.ones_like(pc.get_xyz).cuda(),
        opacities = opacity,
        scales = scales,
        rotations = rot,
    )
    render_alpha = out[0][0:1]
    return_dict.update({"mask": render_alpha})
    
    
    if return_depth:
        depth = (anchor-viewpoint_camera.camera_center).norm(dim=1, keepdim=True).repeat([1,3])

        out = rasterizer(
            means3D = anchor,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = depth,
            opacities = opacity,
            scales = scales,
            rotations = rot,
            cov3D_precomp = None)
        

        accumulated_depth = out[0].float().mean(0, keepdim=True)
        valid_depth = render_alpha > 1e-8
        rendered_depth_hand = torch.where(
            valid_depth,
            accumulated_depth / render_alpha.clamp_min(1e-8),
            torch.zeros_like(accumulated_depth),
        )
        render_depth = rendered_depth_hand / (rendered_depth_hand.max() + 1e-8)

        return_dict.update({
            'real_depth': torch.clamp(rendered_depth_hand, -10000, 10000),   # [1, H, W]
            'render_depth': (rendered_depth_hand/(rendered_depth_hand.max() + 1e-8)).repeat(3, 1, 1) # [3, H, W]
        })
        
        if return_normal:
            
            surf_normal = depth_to_normal(viewpoint_camera, render_depth) 
            surf_normal = surf_normal.permute(2, 0, 1)
            surf_normal = ((surf_normal + 1) / 2 * render_alpha)

            return_dict.update({
                'render_depth_normal': torch.clamp(surf_normal, 0.0, 1.0), 
                # 'render_gs_normal': torch.clamp((render_norm + 1) * render_alpha / 2, 0., 1.) # [3, H, W]
            })

    if dynamic_decode:
        return_dict.update({
            'delta_anchor': pc.delta_xyz,
            # 'dynamic_mask': dynamic_mask
        })


    return return_dict


def project_points_to_image(xyz, view):
    """
    xyz: [N, 3] tensor of 3D points (on CUDA)
    viewpoint_camera: camera object with .world_view_transform, .full_proj_transform, .image_width, .image_height
    Returns: [N, 2] tensor of 2D pixel coordinates
    """

    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    points_ndc = xyz_h @ view.full_proj_transform

    points_ndc = points_ndc / points_ndc[:, 3:]
    x = (points_ndc[:, 0] + 1) * 0.5 * (view.image_width - 1)
    y = (points_ndc[:, 1] + 1) * 0.5 * (view.image_height - 1)
    pixel_coords = torch.stack([x, y], dim=1)  # [N, 2]

    valid_x = (x >= 0) & (x < view.image_width)
    valid_y = (y >= 0) & (y < view.image_height)
    valid_mask = valid_x & valid_y

    pixel_coords = pixel_coords[valid_mask].long()

    return pixel_coords, valid_mask






