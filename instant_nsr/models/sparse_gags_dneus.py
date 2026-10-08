import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import instant_nsr.models as models
from instant_nsr.models.base import BaseModel
from instant_nsr.models.utils import chunk_batch
from instant_nsr.systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect
from instant_nsr.models.network_utils import get_encoding
from instant_nsr.models import register
from gaussian_splatting.scene.transform_field import TransformFieldRay
import time as timer

# 예제 사용

    


# Positional encoding (section 5.1)




class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    # Global kernel size 
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)

@register('sparse-gags-dneus')
class DynamicNeuSModel(BaseModel):
    def setup(self):
        # self.embed_fn, input_ch_time = get_embedder(self.config.embedder.multires, 1, self.config.embedder.i_embed)

        self.config.geometry.time_feat_dim = self.config.geometry.time_encoding_config.n_frequencies * 2
        self.config.geometry.position_feat_dim = self.config.geometry.position_encoding_config.n_frequencies * 2 * 3 + 3
        self.config.texture.time_feat_dim = self.config.texture.time_encoding_config.n_frequencies * 2
        
        self.use_hash= False
        self.use_plane = False 
        self.use_transform= False
        self.use_voxel = False

        self.xyz_encoding = get_encoding(3, self.config.geometry.position_encoding_config)
        self.time_encoding = get_encoding(1, self.config.geometry.time_encoding_config)
        self.view_encoding = get_encoding(3, self.config.geometry.view_encoding_config)
        
        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01            

        self.variance = VarianceNetwork(self.config.variance)
        aabb_radius = self.config.radius
        self.scene_center = torch.tensor(self.config.get('scene_center', [0, 0, 0])).float()
        self.register_buffer('scene_aabb', torch.as_tensor([self.scene_center[0]-aabb_radius, self.scene_center[1]-aabb_radius, self.scene_center[2]-aabb_radius, self.scene_center[0]+aabb_radius, self.scene_center[1]+aabb_radius, self.scene_center[2]+aabb_radius], dtype=torch.float32)) # Static Scene
        
        # self.register_buffer('scene_aabb', torch.zeros(6, dtype=torch.float32)) # Dynamic Scene
        if self.config.grid_prune:
            if not self.config.gs_sampling:
                self.occupancy_grid = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.AABB
                )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=128,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

        # self.config.geometry.add_feat_dim = self.config.voxel_feat_dim
        # self.config.texture.add_feat_dim = 64 # self.config.voxel_feat_dim
        
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB

        # This changes the first sparsify layer input width:
        #   False -> 64 (deformation feature only)
        #   True  -> 91 (deformation feature + 27-D embedded VoxGS color)
        # Keep it configurable because older ablation checkpoints were trained
        # with the former architecture.
        include_voxgs = bool(
            self.config.transform_field.get("include_voxgs", True)
        )
        self.tineuvox = TransformFieldRay(
            W1=32,
            W2=64,
            voxelsize=self.config.voxel_size, 
            out_dim=3, 
            pos_emb=self.config.geometry.position_encoding_config.n_frequencies,
            time_emb=self.config.geometry.time_encoding_config.n_frequencies,
            scene_radius = self.config.radius,
            config = self.config.transform_field,
            # scene_radius=config.model.radius,
            ablation_include_voxgs=include_voxgs,
        ).to('cuda').eval()


        # Backup deform
        # self.tineuvox = TiNeuVoxel.TiNeuVox(
        #     xyz_min=self.scene_aabb[:3], 
        #     xyz_max=self.scene_aabb[3:],
        #     voxel_dim=self.config.voxel_feat_dim,
        #     net_width=self.config.geometry.add_feat_dim,
        #     num_voxels=self.config.voxel_resolution**3,
        #     xyz_emb = self.xyz_encoding,
        #     time_emb = self.time_encoding,
        # )

    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)
        # update_module_step(self._time, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):

            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha

        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            return density[...,None] * self.render_step_size_bg
        
        if self.training and self.config.grid_prune:
            # Sampling guided by predicted depth of rendering branch instead of maintaining an occupancy grid for the frontground
            if not self.config.gs_sampling:
                self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
            # Maintain an occupancy grid for the background
            if self.config.learned_background:
                self.occupancy_grid_bg.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn_bg, occ_thre=self.config.get('grid_prune_occ_thre_bg', 0.01))
    
    # surface extraction
    def isosurface(self, time=0):
        mesh = self.geometry.isosurface(time=time)
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def update_scene_aabb(self, aabb: torch.Tensor):
        """
        Update scene AABB on the fly.
        aabb: Tensor of shape (6,) - [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        assert aabb.shape == (6,)
        self.scene_aabb.copy_(aabb)

    @torch.no_grad()
    def ray_upsample_hier_uniform(self, rays_o, rays_d, gs_model=None, gs_depth=None, time=None, stratified=True, use_depth_guide=False, anchor= None, debug=False, stable_train=False, perturb=0, uniform_ratio=0.1, trans_gs_asset=None):
        ################ nerfies 보정 값
        sdf_lambda = self.config.get('sdf_lambda', 1.0)
        t_min, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        
        if gs_depth is not None:
            # Obtain the depth points according to the depthmap
            gs_depth_positions = rays_o[:, None, :] + rays_d[:, None, :] * gs_depth[..., None]
            gs_depth_positions = gs_depth_positions.reshape(-1, 3)

            # Calculate the sdf values in the depth points
            with torch.no_grad():
                vox_ret = self.forward_deform(gs_depth_positions, time, trans_gs_asset=trans_gs_asset)
                delta_p = vox_ret['delta']
                canon_positions = gs_depth_positions + delta_p
    
                _input = canon_positions

                gs_depth_sdf = self.geometry(_input, with_grad=False, with_feature=False) * sdf_lambda

            gs_depth_probe = gs_depth

            # Coarse and fine intervals 'a' (positive related to the |sdf value|)
            # ``radius`` controls the global reconstruction AABB, while
            # ``interval_radius`` controls only the local depth-guided sampling
            # window. Keep the two scales independent (D-NeRF uses 0.7 / 1.4).
            interval_radius = self.config.get('interval_radius', self.config.radius)
            interval = gs_depth_sdf.abs() * interval_radius * 3

            interval_fine = gs_depth_sdf.abs() * interval_radius

            # Set lower bound for sampling intervals.
            interval = interval.clamp_min(self.geometry._finite_difference_eps_list[0]*16)
            interval_fine = interval_fine.clamp_min(self.geometry._finite_difference_eps_list[0]*8)

            low_bound = gs_depth_probe.squeeze(dim=-1) - interval
            upper_bound = gs_depth_probe.squeeze(dim=-1) + interval
            # low_bound = 0.5 * (t_min + t_max).squeeze(dim=-1) - interval
            # upper_bound = 0.5 * (t_min + t_max).squeeze(dim=-1) + interval

            low_bound_fine = gs_depth_probe.squeeze(dim=-1) - interval_fine * 0.4
            upper_bound_fine = gs_depth_probe.squeeze(dim=-1) + interval_fine* 0.4
            # Separate AABB intersection mask and depth validity mask
            aabb_mask = (t_max > 0) & (t_max < 1e9)
            depth_mask = (t_min < gs_depth_probe.squeeze(dim=-1)-0.01) & (gs_depth_probe.squeeze(dim=-1) < t_max)
            # Rays considered intersected must satisfy both
            intersected_ray_indices = (aabb_mask & depth_mask).nonzero(as_tuple=False).view(-1)

            
        # t_min = torch.stack([low_bound.squeeze(dim=-1), t_min],dim=1).max(dim=1).values
        # t_max = torch.stack([upper_bound.squeeze(dim=-1), t_max],dim=1).min(dim=1).values

        if use_depth_guide:
            # 
            t_min_fine = torch.stack([low_bound_fine.squeeze(dim=-1), t_min],dim=1).max(dim=1).values
            t_max_fine = torch.stack([upper_bound_fine.squeeze(dim=-1), t_max],dim=1).min(dim=1).values

            n_equispaced = self.config.num_samples_equispaced
            n_equispaced_fine = self.config.num_samples_equispaced

            non_depth_mask = aabb_mask & (~depth_mask)
            non_depth_indices = non_depth_mask.nonzero(as_tuple=False).view(-1)
            if non_depth_indices.numel() > 0:
                # number of extra indices to add
                k = min(int(rays_o.shape[0] * uniform_ratio), non_depth_indices.numel())
                if k > 0:
                    if k == non_depth_indices.numel():
                        extra_indices = non_depth_indices
                    else:
                        # uniformly spaced sampling (deterministic)
                        linspace_ids = torch.linspace(0, non_depth_indices.numel() - 1, steps=k, device=non_depth_indices.device)
                        linspace_ids = linspace_ids.round().long()
                        # safeguard in case rounding produced duplicates
                        linspace_ids = torch.unique(linspace_ids)
                        # if due to uniqueness we lost some, append remaining sequentially
                        if linspace_ids.numel() < k:
                            needed = k - linspace_ids.numel()
                            all_ids = torch.arange(non_depth_indices.numel(), device=non_depth_indices.device)
                            remaining = all_ids[~torch.isin(all_ids, linspace_ids)]
                            if remaining.numel() > 0:
                                linspace_ids = torch.cat([linspace_ids, remaining[:needed]], dim=0)
                        extra_indices = non_depth_indices[linspace_ids]
                    # concatenate and keep unique (preserve order: intersected first, then extras not already included)
                    intersected_set = set(intersected_ray_indices.tolist())
                    extras_filtered = [idx.item() for idx in extra_indices if idx.item() not in intersected_set]
                    if len(extras_filtered) > 0:
                        extras_tensor = torch.as_tensor(extras_filtered, device=intersected_ray_indices.device, dtype=intersected_ray_indices.dtype)
                        intersected_ray_indices = torch.cat([intersected_ray_indices, extras_tensor], dim=0)
        else:
            intersected_ray_indices = ((t_max > 0) & (t_max < 1e9)).nonzero(as_tuple=False).view(-1)
            n_equispaced = self.config.num_samples_full
            n_equispaced_fine=0


        t_min = t_min[intersected_ray_indices][:, None]
        t_max = t_max[intersected_ray_indices][:, None]
        
        rays_o_ = rays_o[intersected_ray_indices]
        rays_d_ = rays_d[intersected_ray_indices]

        if stratified:
            rands = torch.rand(n_equispaced, dtype=rays_o.dtype, device=rays_o.device)
        else:
            rands = torch.ones(n_equispaced, dtype=rays_o.dtype, device=rays_o.device) * 0.5
        rands += torch.arange(n_equispaced, dtype=rays_o.dtype, device=rays_o.device)

        dists = rands[None, :] / n_equispaced * (t_max - t_min) + t_min    # [N_rays, N_samples]
        
        if use_depth_guide:
            t_min_fine = t_min_fine[intersected_ray_indices][:, None]
            t_max_fine = t_max_fine[intersected_ray_indices][:, None]
            if stratified:
                rands_fine = torch.rand(n_equispaced_fine, dtype=rays_o.dtype, device=rays_o.device)
            else:
                rands_fine = torch.ones(n_equispaced_fine, dtype=rays_o.dtype, device=rays_o.device) * 0.5
            rands_fine += torch.arange(n_equispaced_fine, dtype=rays_o.dtype, device=rays_o.device)
            dists_fine = rands_fine[None, :] / n_equispaced_fine * (t_max_fine - t_min_fine) + t_min_fine    # [N_rays, N_samples]
            dists = torch.cat([dists, dists_fine], dim=-1)
            dists, _ = torch.sort(dists, dim=-1)
            
        # format data
        ray_indices = torch.arange(rays_o.shape[0], dtype=torch.int64, device=rays_o.device)[intersected_ray_indices][:, None]
        ray_indices = ray_indices.expand(-1, n_equispaced+n_equispaced_fine).reshape(-1)
        
        rays_valid_full = torch.zeros(rays_o.shape[0]).cuda()
        rays_valid_full[intersected_ray_indices] += 1
        rays_valid_full = rays_valid_full.bool()

        midpoints = dists.reshape(-1, 1)

        positions = rays_o_[:, None, :] + rays_d_[:, None, :] * dists[..., None]
        positions = positions.reshape(-1, 3)


        interval_radius = self.config.get('interval_radius', self.config.radius)
        last_i_dists = 1.732 * 2 * interval_radius / (n_equispaced+n_equispaced_fine)
        interval_dists = dists[..., 1:] - dists[..., :-1]
        interval_dists = torch.cat([interval_dists,
                                    torch.empty_like(interval_dists[..., :1], dtype=dists.dtype, device=dists.device).fill_(last_i_dists)],
                                   dim=-1).reshape(-1, 1)
        positions = positions.reshape(-1, 3)
        # save_ply_with_color('0.ply', positions)
        
        return ray_indices, midpoints, positions, interval_dists, intersected_ray_indices, rays_valid_full

        
    def forward_deform(self, positions, times, trans_gs_asset=None, return_grad=False):
        if len(times.shape) == 1:
            times = times[None,]
        if times.shape[0] == 1:
            times = times.repeat(positions.shape[0], 1)
        
        ret = self.tineuvox(positions, trans_gs_asset, times, return_grad=return_grad)

        return ret
    
    # def forward_deform(self, positions, time, trans_gs_asset=None, return_grad=False):
    #     ret = self.tineuvox(positions, time)
    #     return ret

    # DNeRF Sampling
    def forward_(self, rays, gs_model, gs_depth=None, use_depth_guide=False, anchor=None, time=torch.tensor(0).float().cuda(), debug=False, stable_train=False, trans_gs_asset=None, **kwargs):
        timer.time()
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        self.trans_gs_asset = trans_gs_asset
        
        # num_samples = self.config.num_samples_fine + self.config.num_samples_equispaced
        if self.config.gs_sampling:
            # neus importance sampling
            # with torch.no_grad():
            # ray_indices, midpoints, positions, dists, intersected_ray_indices, rays_valid_full = self.ray_upsample_hier(rays_o=rays_o, rays_d=rays_d,gs_model=gs_model, gs_depth=gs_depth, time=time, use_depth_guide=use_depth_guide, anchor=anchor, debug=debug, stable_train=stable_train, trans_gs_asset=trans_gs_asset)
            ray_indices, midpoints, positions, dists, intersected_ray_indices, rays_valid_full = self.ray_upsample_hier_uniform(rays_o=rays_o, rays_d=rays_d,gs_model=gs_model, gs_depth=gs_depth, time=time, use_depth_guide=use_depth_guide, anchor=anchor, debug=debug, stable_train=stable_train, trans_gs_asset=trans_gs_asset)
            
            # t_dirs = (rays_d / torch.norm(rays_d, dim=-1, keepdim=True))[ray_indices]
            t_dirs = rays_d[ray_indices]
       
        if self.training:
            if positions.shape[0]==0:
                out={'zero_samples': True}
                return {
                    **out
                }

        if len(time.shape) == 1:
            times = time[None,]
        if times.shape[0] == 1:
            times = times.repeat(positions.shape[0], 1)
        self.time_encoding(times)
        if positions.shape[0] != 0:
            deform_out = self.forward_deform(positions, time, trans_gs_asset=trans_gs_asset, return_grad=True)
        
            delta_p = deform_out['delta']
            feat_p = deform_out['feature']
                        
            canon_positions = positions + delta_p
            _input = canon_positions
            
            if self.config.geometry.grad_type == 'finite_difference':
                sdf, sdf_grad, feature, sdf_laplace = self.geometry(_input, with_grad=True, with_feature=True, with_laplace=True)
            else:
                sdf, sdf_grad, feature = self.geometry(_input,  with_grad=True, with_feature=True,with_laplace=False)
            
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
            # rgb = self.texture(feature, t_dirs, feat_p.detach(), normal)# .reshape(intersected_ray_indices.shape[0], -1, 3)
            rgb = self.texture(feature, t_dirs, feat_p, normal)# .reshape(intersected_ray_indices.shape[0], -1, 3)
            # rgb = self.texture(feature, t_dirs, feature, normal)# .reshape(intersected_ray_indices.shape[0], -1, 3)
            
            weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
            opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)        
            depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
            comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
            comp_normal = (F.normalize(comp_normal, p=2, dim=-1) + 1) / 2
            rgb_map = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        
        else:
            rgb_map = torch.zeros(n_rays, 3).cuda()
            sdf = torch.zeros(n_rays).cuda()
            opacity = torch.zeros(n_rays, 1).cuda()
            depth = torch.zeros(n_rays, 1).cuda()
            delta_p = torch.zeros(1, 3).cuda()
            comp_normal = torch.zeros(n_rays, 3).cuda()

        if self.training:
            curvature = self.geometry.get_sdf_and_curvature_1d_precomputed_gradient_normal_based(_input, normal)
            # curvature of sdf field
            # curvature = self.geometry.get_sdf_and_curvature_1d_precomputed_gradient_normal_based(positions, normal, time)
        
        out = {
            'sdf_samples': sdf,
            'comp_rgb': rgb_map + 1 * (1 - opacity),
            'comp_rgb_full': rgb_map + 1 * (1 - opacity),
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'rays_valid_full': opacity > 0,
            'num_samples_full': torch.as_tensor([len(midpoints)], dtype=torch.int32, device=rays.device),
            'delta_canon_x': delta_p,
            'comp_normal': comp_normal,
            'ori_positions': positions,
            # ``sdf_samples`` is evaluated at these time-to-canonical warped
            # positions.  Keep them in the output so any canonical geometry
            # supervision uses the same coordinates as the SDF query.
            'canon_positions': canon_positions,
            # 'ray_keep_idx': ray_keep_idx,
        }

        if self.training:
            out.update({
                'sdf_grad_samples': sdf_grad,
                'sdf_features':feature,
                'weights': weights.view(-1),
                'points': midpoints.reshape(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1) ,
                # 'delta_positions': delta_positions,
                'sample_rgb':rgb,
                'zero_samples': False,
                'intersected_ray_indices': intersected_ray_indices,
                'curvature': curvature,
            })
            
        return out
    
    def forward(self, rays, gs_model=None, gs_depth=None, use_depth_guide=False, anchor=None, time=torch.tensor(0).float().cuda(), debug=False, stable_train=False, iteration=None, trans_gs_asset=None):
        if len(time.shape) == 0:
            time = time[None,]
        if self.training:
            out = self.forward_(rays, gs_model, gs_depth, use_depth_guide, anchor=anchor, time=time, debug=debug, stable_train=stable_train, trans_gs_asset=trans_gs_asset)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays, gs_model, time=time, use_depth_guide=use_depth_guide, gs_depth=gs_depth, trans_gs_asset=trans_gs_asset)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        
        return losses

    @torch.no_grad()
    def export(self, export_config, trans_gs_asset=None):
        mesh = self.isosurface(trans_gs_asset)
        if export_config.export_vertex_color:
            deform_out = chunk_batch(self.forward_deform, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), times=trans_gs_asset['canon_time'].to(self.rank), trans_gs_asset=trans_gs_asset)
            
            feat_p = deform_out['feature']

            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            rgb = self.texture(feature, -normal, feat_p, normal) # set the viewing directions to the normal to get "albedo"
            
            mesh['v_rgb'] = rgb.cpu()

        return mesh


    def create_time_net(self):

        layers = [nn.Linear(3 + self.time_encoding.n_output_dims, 256)]

        for i in range(8 - 1):
            in_channels = 256
            if i in self.skips:
                in_channels += 3 # self.xyz_encoding.n_output_dims
            layers += [nn.Linear(in_channels, 256)]

        return nn.ModuleList(layers), nn.Linear(256, 3)

    def query_time(self, new_pts, t, net, net_final):
        if len(t.shape) == 1:
            t = t[None,]
        if t.shape[0] == 1:
            t = t.repeat(new_pts.shape[0], 1)
        h = torch.cat([new_pts, t], dim=-1)
                
        
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return {
            'delta': net_final(h)
        }
