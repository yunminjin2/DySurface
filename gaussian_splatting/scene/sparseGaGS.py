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
import numpy as np
from gaussian_splatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from gaussian_splatting.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from gaussian_splatting.utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.utils.general_utils import strip_symmetric, build_scaling_rotation, build_scaling_6d_rotation
from gaussian_splatting.scene.transform_field import Embedding, TransformField

from einops import repeat
import torch.nn.functional as F


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.tensor([w, x, y, z])

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


class SparseGaGSGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def build_covariance_from_scaling_6d_rotation(scaling, scaling_modifier, rotation):
           
            L = build_scaling_6d_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)         
            symm = strip_symmetric(actual_covariance)
            
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.covariance_6d_activation = build_covariance_from_scaling_6d_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int ,canon_t: float =0.0, update_init_factor: int=100, update_hierachy_factor: int=4,):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.refiner_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.expname = None
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.voxelsize = 0.005
        self.delta_xyz  = torch.zeros([1, 3]).to('cuda')
        self.delta_loss = torch.zeros(1).to('cuda')
        self.inverse_delta_loss = torch.zeros(1).to('cuda')
        self.canon_t = torch.tensor([canon_t], dtype=torch.float32).to('cuda')
        self.m_scaling_mask_hook = None  # gradient mask handle for anisotropic scaling

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_color(self):
        return self._features_dc[:, 0]
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_t(self, t):
        self.t = torch.tensor(t, dtype=torch.float32).to('cuda')    # [1]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_transformed_covariance(self, on_canon=False, return_full_matrix=False, return_sqrt=False, inverse_scales=False):
        if on_canon:
            scaling = self.get_scaling
        else:
            scaling = self.get_transformed_scaling
        if inverse_scales:
            scaling = 1. / scaling.clamp(min=1e-8)
        scaled_rotation = self.o_rotation * scaling[:, None]
        if return_sqrt:
            return scaled_rotation
        
        cov3Dmatrix = scaled_rotation @ scaled_rotation.transpose(-1, -2)
        if return_full_matrix:
            return cov3Dmatrix
        
        cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.get_transformed_scaling.device)
        cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
        cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
        cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
        cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
        cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
        cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
        
        return cov3D
    
    @property
    def get_transformed_xyz(self):
        return self.o_xyz
    

    @property
    def get_transformed_scaling(self):
        return self.scaling_activation(self.o_scaling)
    
    
    @property
    def get_transformed_rotation(self):
        return matrix_to_quaternion(self.rotation_activation(self.o_rotation))
        # return self.rotation_activation(self.o_rotation)

    

    def transformation(self, t=None, return_xyz=False):
        '''
        _xyz : N, 3
        t : 1
        do inference, get the output of _xyz, _rotation, _scaling
        '''
        
        self._xyz_embeded = self.embedding_xyz(self._xyz.detach()) # [N, 63]
        
        _t = self.t
        _t = repeat(_t, 'n -> m n', m=self._xyz.shape[0]) # [N, 1]
        self.t_embeded = self.embedding_t(_t)     # [N, 13]
        self.t_input = self.t_embeded
        # transform_field_input = torch.cat([self._xyz_embeded, self.t_input, self._xyz.detach()], -1)
        
        # transform_field_output = self.trans_vox(self._xyz, SH2RGB(self.get_color), _t)
        transform_field_output, vox_inds, vox_feat = self.trans_vox(self._xyz, _t, return_feat=True)
        

        self.delta_xyz = transform_field_output[:,:3]
        self.six_d = transform_field_output[:,3:7]
        self.delta_scaling = transform_field_output[:, 7:10]
        self.vox_feat = vox_feat
        self.vox_inds = vox_inds
        
        self.o_xyz = self._xyz + self.delta_xyz
        # self.o_rotation = rotation_6d_to_matrix(self.six_d) @ rotation_6d_to_matrix(self._rotation)
        self.o_rotation = quaternion_to_matrix(self.six_d) @ quaternion_to_matrix(self._rotation)
        # self.o_rotation = self._rotation + self.six_d
        self.o_scaling = self._scaling + self.delta_scaling   
        
  
        self.delta_loss = torch.mean(torch.abs(transform_field_output))
    
        # self.inv_transform_field_output = self.inverse_trans_vox(self.t, {
        #     'xyz': self.o_xyz,
        # })
        
        # self.inverse_delta_xyz = self.inv_transform_field_output['delta_inv'][:, :3]
        # self.inverse_delta_loss = torch.mean(torch.abs(self.inverse_delta_xyz))
        # self.pred_canon_xyz = self.inv_transform_field_output['canon_xyz']



    # def canon_transformation(self):
    #     self._xyz_embeded = self.embedding_xyz(self._xyz.detach()) # [N, 63]
        
    #     _t = self.canon_t
    #     _t = repeat(_t, 'n -> m n', m=self._xyz.shape[0]) # [N, 1]
    #     self.t_embeded = self.embedding_t(_t)     # [N, 13]
    #     self.t_input = self.t_embeded
    #     # transform_field_input = torch.cat([self._xyz_embeded, self.t_input, self._xyz.detach()], -1)
        
    #     canon_transform_out = self.trans_vox(self._xyz, _t)
    #     self.canon_delta_xyz = canon_transform_out[: ,:3]
    #     self.canon_delta_rot = canon_transform_out[: , 3:7]
    #     self.canon_delta_scaling = canon_transform_out[: ,7:]

    #     self.c_xyz = self._xyz + self.canon_delta_xyz
    #     self.c_rotation = quaternion_to_matrix(self.canon_delta_rot) @ quaternion_to_matrix(self._rotation)
    #     self.c_scaling = self._scaling + self.canon_delta_scaling
        

    
    def from_6d_get_covariance_static(self, scaling_modifier = 1):
        return self.covariance_6d_activation(self.get_scaling, scaling_modifier, self._rotation)

    def from_6d_get_covariance_dynamic(self, scaling_modifier = 1):
        return self.covariance_6d_activation(self.get_transformed_scaling, scaling_modifier, self.o_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, config, other_args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # xyz
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())  #rgb
        # self.max_sh_degree = 3
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()    # [N, 3, 16]
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # torch.from_numpy(np.asarray(pcd.points) = torch.Size([182686, 3])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 
                                0.0000001)  # torch.Size([182686])
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)         # [182686, 3]
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")   # 6d rotations
        rots[:,0] = 1.
        rots[:,-2] = 1.
        

        self.trans_vox = TransformField(
            voxelsize=other_args.voxelsize, 
            out_dim=10, 
            args=other_args,
            # scene_radius=config.model.radius,
        ).to('cuda').eval()
        
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), 
                                                     dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))  
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) 
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        
    def training_setup(self, training_args, other_args):
        #setup transform field
        # self.mlp_deform = TransformField_GaGS(voxelsize=self.voxelsize, args=other_args).to('cuda')
        self.embedding_xyz = Embedding(3, other_args.position_emb_level)
        self.embedding_t = Embedding(1, other_args.time_emb_level)
        self.voxel_points = 0
        
        self.percent_dense = training_args.percent_dense   # 0.01
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(0.9, 0.999))
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.transformfield_scheduler_args = get_expon_lr_func(lr_init=training_args.network_lr_init,
                                                    lr_final=training_args.network_lr_final,
                                                    lr_delay_steps=training_args.start_dynamic_iter + 1,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
  
    def update_learning_rate(self, iteration, start_dynamic_iter):
        ''' Learning rate scheduling per step '''
        if iteration <= start_dynamic_iter:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr
        else:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "trans_vox" or param_group["name"] == "inverse_trans_vox":
                    transform_field_lr = self.transformfield_scheduler_args(iteration)
                    param_group['lr'] = transform_field_lr
                    return lr
                

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_canon_gs(self, path):
        def _construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l


        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in _construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_sparse_vox(self, path):
        from instant_nsr.models.utils import save_ply_with_color

        mkdir_p(os.path.dirname(path))
        sub_vox = self.trans_vox.voxelize(self._xyz, SH2RGB(self.get_color))[0]
                
        if sub_vox is None:
            return

        # TorchSparse SparseTensor -> (N, 3) points (+ optional colors)
        if hasattr(sub_vox, 'C'):
            coords = sub_vox.C
            if coords is None or coords.numel() == 0:
                return
            if coords.dim() != 2 or coords.shape[1] < 3:
                raise ValueError(f"sub_sparse_voxel.C must be (N,3/4), got {tuple(coords.shape)}")
            points = coords[:, :3].to(dtype=torch.float32)

            colors = None
            if hasattr(sub_vox, 'F') and sub_vox.F is not None:
                
                colors =sub_vox.F
                
                if colors.dim() == 2 and colors.shape[1] >= 3:
                    rgb = colors[:, :3].to(dtype=torch.float32)
                    rgb.amin().item()
                    rgb_max = rgb.amax().item()

                    # Only treat features as colors when they look like RGB.
                    if rgb_max >= 220.0 + 1e-4:
                        colors = (rgb / 255.0).clamp(0.0, 1.0)
                    else:
                        colors = rgb.clamp(0.0, 1.0)
            
            save_ply_with_color(path, points, colors)
            return

        if isinstance(sub_vox, torch.Tensor):
            save_ply_with_color(path, sub_vox)
            return

        raise TypeError(f"Unsupported sub_sparse_voxel type: {type(sub_vox)}")

    def save_gs(self, path, time=0.0):
        def _construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l


        mkdir_p(os.path.dirname(path))
        if time is None:
            time = 0.0
        time = torch.tensor([time]).cuda()
        self.get_t(time)
        self.transformation()

        xyz = self.get_transformed_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.o_scaling.detach().cpu().numpy()
        
        rotation = matrix_to_quaternion(self.o_rotation).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in _construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def save_mlp_checkpoints(self, path):
        mkdir_p(os.path.dirname(path))
        
        torch.save({
            # 'mlp_query': self.mlp_query.state_dict(),
            'mlp_deform': self.trans_vox.state_dict(),
        }, path)
    
    def reset_opacity(self):
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        # import pdb; pdb.set_trace()
        # self._scaling = nn.Parameter(self.scaling_inverse_activation(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = nn.Parameter(torch.zeros((self.get_xyz.shape[0]), device="cuda"))    
        self.covgrad = 0
        self.active_sh_degree = self.max_sh_degree
    
    def load_mlp_checkpoints(self,path):
        
        checkpoint = torch.load(path)
        # self.mlp_query.load_state_dict(checkpoint['mlp_query']) 
        self.trans_vox.load_state_dict(checkpoint['trans_vox']) 

    
    @torch.no_grad()
    def update_tineuvox(self, canon_time=0.5):
        self._xyz_embeded = self.embedding_xyz(self._xyz.detach()) # [N, 63]
        
        t = repeat(torch.tensor([canon_time], device="cuda"), 'n -> m n', m=self._xyz.shape[0]) # [N, 1]. # use mid pose voxel
        # t_embeded = self.embedding_t(self.t)     # [N, 13]
        # t_input = self.t_embeded
        # transform_field_input = torch.cat([self._xyz_embeded, self.t_input, self._xyz.detach()], -1)
        coarse_output = self.trans_vox(self._xyz, t)
        
        coarse_delta_xyz = coarse_output[:,:3]
        coarse_delta_rot = coarse_output[:,3:7]
        coarse_delta_scaling = coarse_output[:,7:]
        # self.c_xyz = self._xyz + coarse_delta_xyz
        
        # self.trans_vox.set_sparse_voxel_grid(self.c_xyz)
              

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "trans_vox" or group["name"]=="inverse_trans_vox":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self.max_radii2D.shape[0] != valid_points_mask.shape[0]:
            padding = valid_points_mask.shape[0] - self.max_radii2D.shape[0]
            if padding > 0:
                self.max_radii2D = torch.cat([
                    self.max_radii2D,
                    torch.zeros(padding, device=valid_points_mask.device, dtype=self.max_radii2D.dtype),
                ], dim=0)
            elif padding < 0:
                self.max_radii2D = self.max_radii2D[:valid_points_mask.shape[0]]
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        # tensors_dict 里面只包含满足要求的n个点的信息
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"]=="inverse_trans_vox" or group["name"]=="trans_vox":
                continue
            try:
                assert len(group["params"]) == 1
            except:
                print(group["name"])
                assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            
            # group['params'][0].shape = [150_000, 3]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                '''
                在Adam优化器中, 'exp_avg_sq' 存储了每个参数的平方梯度的移动平均值。
                'exp_avg' 存储了每个模型参数的梯度的指数移动平均值。这个移动平均值在每个训练步骤中都会更新，以捕捉梯度的变化趋势。
                
                stored_state["exp_avg"].shape = [150_000, 3]
                stored_state["exp_avg_sq"].shape = [150_000, 3]
                '''
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                '''
                stored_state["exp_avg"].shape = [150_202, 3]
                stored_state["exp_avg_sq"].shape = [150_202, 3]
                '''
                
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        '''
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        '''
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

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, iteration, args, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")  # n+m
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) 
        
        if iteration <= args.stable_until_iter:
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)     # [148, 3]
            # stds = self.get_transformed_scaling[selected_pts_mask].repeat(N,1)     # [148, 3]
            means =torch.zeros((stds.size(0), 3),device="cuda")        # [74*2, 3]
            samples = torch.normal(mean=means, std=stds)               # [148, 3]
            #rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)    # [148, 3, 3]
            rots = rotation_6d_to_matrix(self._rotation[selected_pts_mask]).repeat(N,1,1) 
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
            self.actually_scaling = torch.cat([self.actually_scaling, 
                                               self.actually_scaling[selected_pts_mask],
                                               self.actually_scaling[selected_pts_mask]])
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
            
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)
            self.actually_scaling = self.actually_scaling[~prune_filter]
        else:
            padded_scale = torch.zeros((n_init_points, 3), device="cuda")
            padded_scale[:grads.shape[0]] = self.get_transformed_scaling
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(padded_scale, dim=1).values > \
                                                self.percent_dense*scene_extent)
            '''
            change
            '''
            stds = padded_scale[selected_pts_mask].repeat(N,1)     # [148, 3]
            # stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            # stds = self.get_transformed_scaling[selected_pts_mask].repeat(N,1)     # [148, 3]
            means =torch.zeros((stds.size(0), 3),device="cuda")        # [74*2, 3]
            samples = torch.normal(mean=means, std=stds)               # [148, 3]
            padded_rots = torch.zeros((n_init_points, 4), device="cuda")
            # padded_rots[:grads.shape[0]] = self.get_transformed_rotation   
            '''
            change
            '''   
            padded_rots[:grads.shape[0]] = self.get_transformed_rotation.clone()
            # rots = rotation_6d_to_matrix(self._rotation)[selected_pts_mask].repeat(N,1,1)
            
            rots = quaternion_to_matrix(padded_rots[selected_pts_mask].repeat(N,1))
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            self.actually_scaling = torch.cat([self.actually_scaling, 
                                                self.actually_scaling[selected_pts_mask],
                                                self.actually_scaling[selected_pts_mask]])
            
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
            

            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)
            self.actually_scaling = self.actually_scaling[~prune_filter]

            

    def densify_and_clone(self, grads, grad_threshold, scene_extent, iteration, args):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        if iteration > args.stable_until_iter:
            if selected_pts_mask.shape[0] != self.get_transformed_scaling.shape[0]:
                raise ValueError("Densification gradient count does not match Gaussian count")
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_transformed_scaling, dim=1).values \
                                                <= self.percent_dense*scene_extent)
            
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            self.actually_scaling = self.get_transformed_scaling.clone()
            
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
            self.actually_scaling = torch.cat([self.actually_scaling, 
                                                self.actually_scaling[selected_pts_mask]])
        else:
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            self.actually_scaling = self.scaling_activation(self._scaling).clone()
            
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
            
            self.actually_scaling = torch.cat([self.actually_scaling, 
                                               self.actually_scaling[selected_pts_mask]])
        

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration, args):
        grads = self.xyz_gradient_accum / self.denom   
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, iteration, args)
        
        self.densify_and_split(grads, max_grad, extent, iteration, args)
        
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
       
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            if iteration > args.stable_until_iter:
                big_points_ws = self.actually_scaling.max(dim=1).values > 0.1 * extent
            else:
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
           
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def training_statis(self, viewspace_point_tensor, opacity, visibility_filter):
        temp_opacity = opacity.clone().detach()
        temp_opacity[temp_opacity < 0] = 0
        
        # self.anchor_gradient_accum[visibility_filter] += temp_opacity / 2
        self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter,:2], dim=-1, keepdim=True) / 2
        
        self.denom[visibility_filter] += 1


    def adjust_xyz(self, extent, add_contents=None, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005, xyz_sdf=None, inside_box=None, xyz_inside_box=None, growing_weight=0.0002):

        if self.denom.numel() == 0:
            return {"grown": 0, "pruned": 0, "reset": 0}

        weight_prune = 1.0
        if xyz_sdf is not None:
            def simple_sdf_activate(x, sigma=0.01):
                return torch.exp(-x**2 / sigma)

        grads = self.xyz_gradient_accum / self.denom.clamp_min(1)
        grads[grads.isnan()] = 0.0

        grads_norm = grads.squeeze(dim=-1)

        if xyz_sdf is not None:
            if inside_box is None:
                inside_box = torch.zeros_like(grads_norm, dtype=torch.bool, device=grads_norm.device)
            xyz_sdf_activated = simple_sdf_activate(xyz_sdf)
            xyz_sdf_activated[~inside_box] = 0.0
            grads_norm = grads_norm + growing_weight * xyz_sdf_activated

        before_grow = self.get_xyz.shape[0]
        self.xyz_growing(grads_norm, grad_threshold)
        grown = int(self.get_xyz.shape[0] - before_grow)

        xyz_opacity_sdf_accum = self.xyz_gradient_accum

        if xyz_sdf is not None:
            if xyz_inside_box is None:
                xyz_inside_box = inside_box
            xyz_sdf_activated = simple_sdf_activate(xyz_sdf)
            xyz_sdf_activated[~xyz_inside_box] = 1.0
            padding_length = self.get_xyz.shape[0] - xyz_sdf_activated.shape[0]
            if padding_length > 0:
                padding_ones = torch.ones([padding_length], device=self.get_xyz.device, dtype=xyz_sdf_activated.dtype)
                padded_xyz_sdf_activated = torch.cat([xyz_sdf_activated, padding_ones], dim=0)
            else:
                padded_xyz_sdf_activated = xyz_sdf_activated
            xyz_opacity_sdf_accum = self.xyz_gradient_accum - weight_prune * self.denom * (1 - padded_xyz_sdf_activated.unsqueeze(dim=1))

        prune_mask = (xyz_opacity_sdf_accum < min_opacity * self.denom).squeeze(dim=1)
        xyzs_mask = (self.denom > check_interval * success_threshold).squeeze(dim=1)
        prune_mask = torch.logical_and(prune_mask, xyzs_mask)

        scaling_mask = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_and(prune_mask, scaling_mask)

        reset_mask = torch.logical_and(xyzs_mask, ~prune_mask)
        reset = int(reset_mask.sum().item())
        if reset > 0:
            self.xyz_gradient_accum[reset_mask] = 0
            self.denom[reset_mask] = 0

        pruned = int(prune_mask.sum().item())
        if pruned > 0:
            self.prune_points(prune_mask)

        return {"grown": grown, "pruned": pruned, "reset": reset}

    def xyz_growing(self, grads, threshold):
        init_length = self.get_xyz.shape[0]
        
        # update threshold
        cur_threshold = threshold*((self.update_hierachy_factor//2))
        # mask from grad threshold
        candidate_mask = (grads >= cur_threshold)

        
        length_inc = self.get_xyz.shape[0] - init_length
        candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
        

        all_xyz = self.get_xyz
        
        # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
        # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
        size_factor = self.update_init_factor // (self.update_hierachy_factor)
        cur_size = 0.01 * size_factor

        selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
        
        ## split data for reducing peak memory calling

        candidate_xyz = selected_xyz*cur_size

        
        if candidate_xyz.shape[0] > 0:
            new_scaling = torch.ones_like(candidate_xyz).float().cuda()*cur_size # *0.05
            new_scaling = torch.log(new_scaling)
            new_rotation = torch.zeros([candidate_xyz.shape[0], 4], device=candidate_xyz.device).float()
            new_rotation[:,0] = 1.0

            new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_xyz.shape[0], 1), dtype=torch.float, device="cuda"))

            new_features_dc = self._features_dc[candidate_mask]
            new_features_rest = self._features_rest[candidate_mask]

            d = {
                "xyz": candidate_xyz,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "f_dc": new_features_dc,
                "f_rest": new_features_rest,
                "opacity": new_opacities,
            }
            
            temp_xyz_denom = torch.cat([self.denom, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
            del self.denom
            self.denom = temp_xyz_denom

        
            temp_opacity_accum = torch.cat([self.xyz_gradient_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
            del self.xyz_gradient_accum
            self.xyz_gradient_accum = temp_opacity_accum

            temp_max_radii2D = torch.cat([self.max_radii2D, torch.zeros([new_opacities.shape[0]], device='cuda').float()], dim=0)
            del self.max_radii2D
            self.max_radii2D = temp_max_radii2D

            torch.cuda.empty_cache()

            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self._xyz = optimizable_tensors["xyz"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            



    @torch.no_grad()
    def get_sdf(self, x, return_sdf=True, return_sdf_grad=False, sdf_grad_max_value=10, return_beta=False, density_threshold=0.6, density_factor=1, opacity_min_clamp=1e-16, on_canon=False, K=4):
        def get_beta(x, 
                    closest_gaussians_idx=None, 
                    closest_gaussians_opacities=None,
                    densities=None,
                    opacity_min_clamp=1e-16,):
            if closest_gaussians_idx is None:
                raise ValueError("closest_gaussians_idx must be provided when using beta_mode='weighted_average'.")
            if closest_gaussians_opacities is None:
                raise ValueError("closest_gaussians_opacities must be provided when using beta_mode='weighted_average'.")
            
            
            min_scaling = self.get_transformed_scaling.min(dim=-1)[0][closest_gaussians_idx]
            opacities_sum = closest_gaussians_opacities.sum(dim=-1, keepdim=True)
            weights = closest_gaussians_opacities / opacities_sum.clamp(min=opacity_min_clamp)
                    
            beta = (min_scaling * weights).sum(dim=-1)
            with torch.no_grad(): 
                beta[opacities_sum[..., 0] == 0.] = min_scaling.max().detach()
        
            return beta
        
        
        gaussian_centers = self._xyz if on_canon else self.o_xyz
        

        closest_gaussians_idx = knn_points(x[None], gaussian_centers[None], K=K).idx[0]
        gaussian_inv_scaled_rotation = self.get_transformed_covariance(on_canon=on_canon, inverse_scales=True, return_full_matrix=True, return_sqrt=True)
        
    
        closest_gaussian_centers = gaussian_centers[closest_gaussians_idx]
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[closest_gaussians_idx]
        closest_gaussian_strengths = torch.ones_like(self.get_opacity)[closest_gaussians_idx]
        
        shift = (x[:, None] - closest_gaussian_centers)
        l2_dist = 1 / 2 * torch.sqrt(torch.square(shift).sum(-1)).mean(-1) # shift.abs().sum(dim=-1).mean(dim=-1) l1dist
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        theta = 1 - torch.exp(-1./2 * l2_dist)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
        densities = neighbor_opacities.sum(dim=-1)
        
        fields = {}
        fields['density'] = densities.clone()
        fields['nn_gaussian_idx'] = closest_gaussians_idx
        density_mask = densities >= 1.
        densities[density_mask] = densities[density_mask] / (densities[density_mask].detach() + 1e-12)
        
                
        beta = get_beta(x, 
                closest_gaussians_idx=closest_gaussians_idx, 
                closest_gaussians_opacities=neighbor_opacities, 
                densities=densities,
                opacity_min_clamp=opacity_min_clamp,
            )
        clamped_densities = densities.clamp(min=opacity_min_clamp) #, max=1.-1e-4)

        if return_beta:
            fields['beta'] = beta
        
        # Compute the signed distance field
              
        if return_sdf:
            # sdf_values = beta * (torch.sqrt(-2. * torch.log(clamped_densities)) - np.sqrt(-2. * np.log(min(density_threshold, 1.))))
            
            sdf_values = torch.tanh(theta * (torch.sqrt(-2. * torch.log(clamped_densities)) - np.sqrt(-2. * np.log(min(density_threshold, 1.)))))
            # - np.sqrt(-2. * np.log(max(1e-4, min(density_threshold, 1.))))
            fields['sdf'] = sdf_values
            
        # Compute the gradient of the signed distance field
        if return_sdf_grad:
            sdf_grad = neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]
            sdf_grad = sdf_grad.sum(dim=-2)
            sdf_grad = (beta / (clamped_densities * torch.sqrt(-2. * torch.log(clamped_densities))).clamp(min=opacity_min_clamp))[..., None] * sdf_grad
            fields['sdf_grad'] = sdf_grad.clamp(min=-sdf_grad_max_value, max=sdf_grad_max_value)
            
        return fields
    
    @torch.no_grad()
    def get_gaussians_closest_to_samples(self, x, gs_xyz=None, n_closest_gaussian=None):
        if n_closest_gaussian is None:
            if not hasattr(self, 'knn_to_track'):
                print("Variable knn_to_track not found. Setting it to 16.")
                self.knn_to_track = 16
            n_closest_gaussian = self.knn_to_track
        if gs_xyz is None:
            gs_xyz = self._xyz
        
        closest_gaussians_idx = knn_points(x[None], gs_xyz[None], K=n_closest_gaussian).idx[0]
        return closest_gaussians_idx

    @torch.no_grad()
    def query_point_attributes(
        self, 
        x: torch.Tensor, 
        v_pcd=None,
        density_factor=1.,
        opacity_min_clamp=1e-16,
        tgt_gs=None,
        distance_threshhold=0.01,
        cal_normal=False,
        K=8,
        detach=False
    ):
        """
        Compute density, normal, and color for queried 3D points.

        Args:
            x (torch.Tensor): Query coordinates of shape (N, 3).
            density_factor (float): Scale factor applied to density estimation.
            opacity_min_clamp (float): Minimum clamp to avoid division by zero.
            tgt_gs (dict): Target gaussian asset with keys 'xyz', 'feature',
                'rotations', 'scaling', 'opacity'. If None, uses the current
                transformed gaussians (self.o_xyz, etc.).
            distance_threshhold (float): Distance cutoff for nearby gaussians.
            return_normal (bool): Whether to return normals.
            K (int): Number of nearest gaussians to consider.

        Returns:
            dict: Contains 'density' (N,), 'normal' (N, 3) when requested, and 'color' (N, 3).
        """
        feat_dim=3
        if v_pcd is not None:
            points = v_pcd.C[: ,:3]
            features = v_pcd.F
            close_strengths = torch.ones_like(v_pcd.C[:, :1])
            close_inv_cov = None
            feat_dim = features.shape[-1]
        else:
            if tgt_gs is not None:
                points = tgt_gs['xyz']
                features = SH2RGB(tgt_gs['feature'][:, 0])
                close_strengths = torch.ones_like(self.opacity_activation(tgt_gs['opacity']))
                if cal_normal:
                    scaling = self.scaling_activation(tgt_gs['scaling'])
                    inv_scaling = 1. / scaling.clamp(min=1e-8)
                    rot_matrix = quaternion_to_matrix(self.rotation_activation(tgt_gs['rotations']))
                    close_inv_cov = rot_matrix * inv_scaling[:, None]
                else:
                    close_inv_cov = None
            else:
                points = self.o_xyz
                features = SH2RGB(self.get_color)
                close_strengths = torch.ones_like(self.get_opacity)
                close_inv_cov = self.get_transformed_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True) if cal_normal else None

            if detach:
                points = points.detach()
                features = features.detach()
                close_strengths = close_strengths.detach()
                if cal_normal:
                    close_inv_cov = close_inv_cov.detach()
        # 1. 주변 가장 가까운 K개의 가우시안 인덱스 검색 (KNN)
        closest_gaussians_idx = self.get_gaussians_closest_to_samples(x, gs_xyz=points, n_closest_gaussian=K)
        K = closest_gaussians_idx.shape[1]
        # 2. 주변 가우시안들의 파라미터 수집
        close_centers = points[closest_gaussians_idx]  # (N, K, 3)
        close_idx = ((close_centers.mean(1) - x)**2).sum(-1) < distance_threshhold 
        
        N = x[close_idx].shape[0]
        closest_gaussians_idx = closest_gaussians_idx[close_idx]
        close_centers_picked = points[closest_gaussians_idx]  # (N, K, 3)
        
        close_strengths = close_strengths[closest_gaussians_idx] # (N, K, 1)
        close_feat = features[closest_gaussians_idx] # (N, 3)
        feat = torch.clamp_min(close_feat, 0.0).view(N, K, feat_dim)

        squared_center_distances = ((close_centers - x[:, None, :]) ** 2).sum(-1)
        return_dict = {
            # Preserve the historical local-mean distance for existing users.
            'dist': ((close_centers.mean(1) - x) ** 2).sum(-1),
            # Canonical SDF supervision needs a distance to the actual point
            # set, not to the mean of K neighboring Gaussian centers.
            'nearest_dist_sq': squared_center_distances.min(dim=-1).values,
        }

        if cal_normal:
            close_inv_cov = close_inv_cov[closest_gaussians_idx] # (N, K, 3, 3)
        
            shift = (x[close_idx, None, :] - close_centers_picked)  # (N, K, 3)
            warped_shift = close_inv_cov.transpose(-1, -2) @ shift[..., None] # (N, K, 3, 1)
        
            maha_dist = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8) # (N, K)
            weights = density_factor * close_strengths[..., 0] * torch.exp(-0.5 * maha_dist) # (N, K)
            density = weights.mean(dim=-1) # (N,)
        
            density_grad = (weights[..., None] * (close_inv_cov @ warped_shift)[..., 0]).sum(dim=-2) 
            normal = -torch.nn.functional.normalize(density_grad, dim=-1) # (N, 3)


            feat = (weights[..., None] * feat).sum(dim=1) / density.clamp(min=opacity_min_clamp)[..., None]
        else:
            feat = feat.mean(dim=1)
        feat_full = torch.zeros((x.shape[0], feat_dim), device=x.device)
        feat_full[close_idx] = feat
        return_dict['feat'] = feat_full

        
        if cal_normal:
            density_full = torch.zeros((x.shape[0]), device=x.device)
            density_full[close_idx] = density
            return_dict['density'] = density_full

            normal_full = torch.zeros((x.shape[0], 3), device=x.device)
            normal_full[close_idx] = normal
            return_dict['normal'] = normal_full
        
        return return_dict
            
    

    def create_from_mesh(self, mesh, spatial_lr_scale: float = 1.0, other_args=None, n_gs=30000):
        """
        Initialize Gaussians directly on a mesh surface with rigid binding variables.
        """
        def _to_numpy(arr):
            if arr is None: return None
            if torch.is_tensor(arr): return arr.detach().cpu().numpy()
            return np.asarray(arr)

        verts = faces = colors = normals = None
        
        if hasattr(mesh, "verts_packed"):
            verts = _to_numpy(mesh.verts_packed())
            try: faces = _to_numpy(mesh.faces_packed())
            except Exception: pass
            try:
                if getattr(mesh, "textures", None) is not None and hasattr(mesh.textures, "verts_features_packed"):
                    colors = _to_numpy(mesh.textures.verts_features_packed())
            except Exception: pass
        elif hasattr(mesh, "vertices"):
            verts = _to_numpy(mesh.vertices)
            faces = _to_numpy(getattr(mesh, "faces", None))
            colors = _to_numpy(getattr(mesh, "vertex_colors", None))
        elif isinstance(mesh, dict):
            verts = _to_numpy(mesh.get("verts") or mesh.get("vertices") or mesh.get("v") or mesh.get("v_pos"))
            faces = _to_numpy(mesh.get("faces") or mesh.get("f") or mesh.get("t_pos_idx"))
            colors = _to_numpy(mesh.get("colors") or mesh.get("vertex_colors") or mesh.get("v_rgb"))
        else:
            verts = _to_numpy(mesh)

        if verts is None or faces is None:
            raise TypeError("mesh must provide vertices and faces.")

        faces_int = faces.astype(np.int64)
        tris = verts[faces_int]
        
        # 1. 면적 기반 확률 계산 및 샘플링
        areas = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0], axis=1), axis=1) / 2.0
        area_sum = areas.sum()
        probs = np.ones_like(areas) / len(areas) if area_sum <= 0 else areas / area_sum
        
        sample_count = int(n_gs)
        face_choices = np.random.choice(faces_int.shape[0], size=sample_count, p=probs)
        
        # 2. Barycentric Coordinates 계산
        u = np.random.rand(sample_count, 1)
        v = np.random.rand(sample_count, 1)
        sqrt_u = np.sqrt(u)
        bary = np.hstack((1 - sqrt_u, sqrt_u * (1 - v), sqrt_u * v)).astype(np.float32)
        
        # 3. 데이터 조립 (위치 및 법선 벡터 계산)
        bound_face_indices = faces_int[face_choices] 
        bound_tris = verts[bound_face_indices]
        samples = (bound_tris * bary[:, :, None]).sum(axis=1)

        # [추가] 샘플링된 Face의 법선 벡터(Normal) 계산
        e1 = bound_tris[:, 1] - bound_tris[:, 0]
        e2 = bound_tris[:, 2] - bound_tris[:, 0]
        face_normals = np.cross(e1, e2)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
        
        if colors is None:
            colors = np.ones((verts.shape[0], 3), dtype=np.float32) * 0.5
        colors = colors.astype(np.float32)
        if colors.shape[1] > 3: colors = colors[:, :3]
        if colors.max() > 1.0 + 1e-3: colors = colors / 255.0

        face_colors = colors[bound_face_indices]
        color_samples = (face_colors * bary[:, :, None]).sum(axis=1)

        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud = torch.tensor(samples, dtype=torch.float32, device="cuda")
        face_normals_tensor = torch.tensor(face_normals, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(color_samples, dtype=torch.float32, device="cuda"))
        
        self.m_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        self.m_features[:, :3, 0] = fused_color
        self.m_features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)
        base_scale = torch.sqrt(dist2).unsqueeze(1) # [N, 1]

        # ------------------------------------------------------------------
        # [수정 1] Anisotropic Scaling (Z축 Flattening)
        # XY축은 인접 점 거리 기반, Z축은 2D GS와 유사하게 극도로 얇게(예: 0.01배) 설정
        # ------------------------------------------------------------------
        xy_scale = torch.log(base_scale * 0.5).repeat(1, 2)
        z_scale = torch.log(base_scale * 0.001) # 필요 시 상수 값(예: -10.0)으로 고정 가능
        # detach to ensure leaf tensor (avoid non-leaf optimizer error)
        self.m_scaling = nn.Parameter(torch.cat([xy_scale, z_scale], dim=1).detach(), requires_grad=True)

        # ------------------------------------------------------------------
        # [수정 2] Rotation Alignment (표면 법선 정렬)
        # 로컬 Z축(0, 0, 1)이 Face Normal을 향하도록 회전하는 Quaternion 계산
        # ------------------------------------------------------------------
        z_axis = torch.tensor([0.0, 0.0, 1.0], device="cuda", dtype=torch.float32).unsqueeze(0).expand_as(face_normals_tensor)
        
        # Quaternion = [cos(theta/2), sin(theta/2) * axis] 수학적 최적화 (w = 1 + dot)
        w = 1.0 + (z_axis * face_normals_tensor).sum(dim=-1, keepdim=True)
        xyz = torch.cross(z_axis, face_normals_tensor, dim=-1)
        rots = torch.cat([w, xyz], dim=-1)
        rots = torch.nn.functional.normalize(rots, dim=-1)

        # 예외 처리: 법선이 정확히 [0, 0, -1] 이라 분모가 0이 되는 특이점(Singularity) 방지
        singular_mask = (w.squeeze() < 1e-6)
        if singular_mask.any():
            rots[singular_mask, :] = torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda", dtype=torch.float32)

        self.m_rots = F.normalize(rots, dim=-1) # [N, 4] 정규화된 쿼터니언으로 회전 초기화

        self.m_opacity = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        # [필수] Mesh 변형을 구동하기 위한 버퍼(Buffer) 등록
        canonical_verts = torch.tensor(verts, dtype=torch.float32, device="cuda")
        self.bound_faces = torch.tensor(bound_face_indices, dtype=torch.long, device="cuda")
        self.bound_bary = torch.tensor(bary, dtype=torch.float32, device="cuda")
        
        self.R_can, self.S_can = compute_face_orientation(canonical_verts, self.bound_faces)
        
        # Only the XY scaling should receive gradients; mask out Z before stepping
        # if self.m_scaling_mask_hook is None:
        #     mask = torch.tensor([1.0, 1.0, 0.0], device=self.m_scaling.device).view(1, 3)
        #     self.m_scaling_mask_hook = self.m_scaling.register_hook(lambda g, m=mask: g * m)     
        self.refiner_optimizer = torch.optim.Adam(
            [
                # {'params': [self.m_scaling], 'lr': 1e-4},
                {'params': list(self.trans_vox.parameters()), 'lr': 1e-4},
            ],
            lr=0.0,
            eps=1e-15,
            betas=(0.9, 0.999),
        )


    def transform_points(self, points, t=torch.tensor(0.0).float().cuda()):
        points = points.contiguous().cuda()
        t = t.contiguous().cuda()
        # 1. Deformation Network 처리 및 Mesh Vertex 변형
        _t = repeat(t, 'n -> m n', m=points.shape[0]) 
        
        transform_field_output, vox_inds, vox_feat = self.trans_vox(points, _t, return_feat=True)
        deformed_points = points + transform_field_output[:, :3] # [V, 3]

        # 2. Binding 데이터를 이용한 새로운 위치(xyz) 조립
        f_idx = self.bound_faces 
        bary = self.bound_bary   

        v_def = deformed_points[f_idx]      
        deformed_xyz = (v_def * bary.unsqueeze(-1)).sum(dim=1) # [N_gs, 3]

        # 3. Canonical 및 Deformed 상태의 Face 방향/스케일 계산
        R_def, S_def = compute_face_orientation(deformed_points.detach(), f_idx)

        # 4. 변화량(Delta) 계산
        # R_delta = R_def * (R_can)^T 
        R_can_transposed = self.R_can.transpose(1, 2)
        R_delta = torch.bmm(R_def, R_can_transposed) # [N_gs, 3, 3]
        # 행렬을 쿼터니언으로 변환
        q_delta = matrix_to_quaternion(R_delta) # [N_gs, 4]
        # 쿼터니언 곱셈으로 회전 합성: 최종 회전 = R_delta * R_gs
        deformed_rotation = quaternion_multiply(q_delta, self.m_rots)
        deformed_rotation = F.normalize(deformed_rotation, dim=-1)

        # 5. Gaussian 파라미터에 Delta 적용
        # 스케일 변화 비율 계산 (0으로 나누기 방지 및 스케일링 조정)
        S_ratio = (S_def / (self.S_can + 1e-6)).clamp(min=0.8, max=1.2) # [N_gs, 1]
        # 스케일에 비율 곱하기: 최종 스케일 = GS 스케일 * Face 스케일 변화율
        deformed_scale = torch.exp(self.m_scaling) * S_ratio

        # 6. 최종 렌더링 에셋 패키징
        deformed_features = self.m_features.permute(0, 2, 1) # [N_gs, C, D] -> [N_gs, D, C]
        deformed_opacity = self.m_opacity # 렌더러 직전에서 필요시 torch.sigmoid 적용

        gs_asset = {
            'xyz': deformed_xyz,
            'feature': deformed_features,
            'opacity': deformed_opacity,
            'scale': deformed_scale,       # Activation(exp)이 적용된 최종 스케일
            'rotation': deformed_rotation,  # 정규화된 최종 쿼터니언
            'deformed_points': deformed_points
        }
        return gs_asset



from einops import repeat

def safe_normalize(x, eps=1e-6):
    """0으로 나누는 것을 방지하는 안전한 정규화 함수"""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

def quaternion_multiply(q1, q2):
    """두 쿼터니언의 회전을 합성하는 함수 [N, 4]"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def matrix_to_quaternion(matrix):
    """3x3 회전 행렬을 쿼터니언으로 변환하는 함수"""
    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros((matrix.shape[0], 4), dtype=matrix.dtype, device=matrix.device)

    cond1 = trace > 0
    cond2 = (m00 > m11) & (m00 > m22) & ~cond1
    cond3 = (m11 > m22) & ~cond1 & ~cond2
    cond4 = ~cond1 & ~cond2 & ~cond3

    if cond1.any():
        s = torch.sqrt(trace[cond1] + 1.0) * 2
        q[cond1, 0] = 0.25 * s
        q[cond1, 1] = (m21[cond1] - m12[cond1]) / s
        q[cond1, 2] = (m02[cond1] - m20[cond1]) / s
        q[cond1, 3] = (m10[cond1] - m01[cond1]) / s
    if cond2.any():
        s = torch.sqrt(1.0 + m00[cond2] - m11[cond2] - m22[cond2]) * 2
        q[cond2, 0] = (m21[cond2] - m12[cond2]) / s
        q[cond2, 1] = 0.25 * s
        q[cond2, 2] = (m01[cond2] + m10[cond2]) / s
        q[cond2, 3] = (m02[cond2] + m20[cond2]) / s
    if cond3.any():
        s = torch.sqrt(1.0 + m11[cond3] - m00[cond3] - m22[cond3]) * 2
        q[cond3, 0] = (m02[cond3] - m20[cond3]) / s
        q[cond3, 1] = (m01[cond3] + m10[cond3]) / s
        q[cond3, 2] = 0.25 * s
        q[cond3, 3] = (m12[cond3] + m21[cond3]) / s
    if cond4.any():
        s = torch.sqrt(1.0 + m22[cond4] - m00[cond4] - m11[cond4]) * 2
        q[cond4, 0] = (m10[cond4] - m01[cond4]) / s
        q[cond4, 1] = (m02[cond4] + m20[cond4]) / s
        q[cond4, 2] = (m12[cond4] + m21[cond4]) / s
        q[cond4, 3] = 0.25 * s
        
    return F.normalize(q, dim=-1)

def compute_face_orientation(verts, faces_idx):
    """
    지정된 Face(삼각형)의 직교 기저(Orientation)와 스케일을 계산합니다.
    verts: [V, 3], faces_idx: [N_gs, 3]
    """
    v0 = verts[faces_idx[:, 0]] # [N_gs, 3]
    v1 = verts[faces_idx[:, 1]]
    v2 = verts[faces_idx[:, 2]]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))

    # [N_gs, 3, 3] 회전 행렬 조립
    orientation = torch.stack([a0, a1, a2], dim=-1)

    s0 = torch.norm(v1 - v0, dim=-1, keepdim=True)
    s1 = torch.abs(torch.sum(a2 * (v2 - v0), dim=-1, keepdim=True))
    scale = (s0 + s1) / 2.0 # [N_gs, 1]

    return orientation, scale
