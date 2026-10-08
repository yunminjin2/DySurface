import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_ray_directions_nerf(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    # i, j =np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # dirs = np.stack([(i - W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)


    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        -torch.ones_like(i)  # ← make it -Z forward
    ], -1)

    return directions

def get_ray_directions_nerfies(W, H, focal):

    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    i, j = i+0.5, j+0.5
    
    dirs = torch.stack([-(i-.5*W)/focal, -(j-.5*H)/focal, torch.ones_like(i)], -1)

    return dirs

def get_ray_directions_dnerf(W, H, focal):

    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    i, j = i+0.5, j+0.5
    
    dirs = torch.stack([(i-.5*W)/focal, -(j-.5*H)/focal, -torch.ones_like(i)], -1)

    return dirs


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:3,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def get_rays_from_c2w(W, H, focal, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    i, j = i+0.5, j+0.5
    
    dirs = torch.stack([(i-.5*W)/focal, -(j-.5*H)/focal, -torch.ones_like(i)], -1)
    
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[None, :, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[0, :3,3].expand(rays_d.shape)
    # rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    return rays_o, rays_d, dirs


def sample_point_along_ray(rays_o, rays_d, near, far, num_samples=64):
    t_vals = torch.linspace(0., 1., steps=num_samples)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], num_samples)
    
    points = rays_o[:, None] + rays_d[:, None] * z_vals[..., None] # (N_rays, N_samples, 3)
    return points
    