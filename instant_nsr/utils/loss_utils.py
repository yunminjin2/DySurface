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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    # Ensure both inputs are on the same device
    device = img1.device
    img1 = img1.to(device)
    img2 = img2.to(device)
    window = window.to(device).type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def compute_chamfer_distance(pred_mesh, gt_mesh, num_samples=100000):
    """
    Compute Chamfer Distance between predicted mesh and ground truth mesh.
    
    Args:
        pred_mesh: Predicted mesh (trimesh object or dict with 'v_pos' key)
        gt_mesh: Ground truth mesh (trimesh object)
        num_samples: Number of points to sample from each mesh
    
    Returns:
        chamfer_distance: Scalar tensor
    """
    
    # Extract vertices from predicted mesh
    if isinstance(pred_mesh, dict) and 'v_pos' in pred_mesh:
        pred_vertices = pred_mesh['v_pos']
        if isinstance(pred_vertices, torch.Tensor):
            pred_vertices = pred_vertices.detach().cpu().numpy()
    elif hasattr(pred_mesh, 'vertices'):
        pred_vertices = pred_mesh.vertices
    else:
        raise ValueError("pred_mesh must be a trimesh object or dict with 'v_pos' key")
    
    # Extract vertices from ground truth mesh
    if hasattr(gt_mesh, 'vertices'):
        gt_vertices = gt_mesh.vertices
    else:
        raise ValueError("gt_mesh must be a trimesh object")
    
    # Sample points uniformly from both meshes
    pred_points = pred_vertices
    gt_points = gt_vertices
    
    # If meshes have too many points, sample them
    if len(pred_points) > num_samples:
        indices = np.random.choice(len(pred_points), num_samples, replace=False)
        pred_points = pred_points[indices]
    
    if len(gt_points) > num_samples:
        indices = np.random.choice(len(gt_points), num_samples, replace=False)
        gt_points = gt_points[indices]
    
    # Convert to torch tensors
    pred_points = torch.from_numpy(pred_points).float().cuda()
    gt_points = torch.from_numpy(gt_points).float().cuda()
    
    # Compute one-way Chamfer Distance: pred -> gt
    pred_to_gt = torch.cdist(pred_points, gt_points, p=2)  # [N_pred, N_gt]
    pred_to_gt_min, _ = pred_to_gt.min(dim=1)  # [N_pred]
    chamfer_pred_to_gt = pred_to_gt_min.mean()
    
    # Compute one-way Chamfer Distance: gt -> pred
    gt_to_pred = torch.cdist(gt_points, pred_points, p=2)  # [N_gt, N_pred]
    gt_to_pred_min, _ = gt_to_pred.min(dim=1)  # [N_gt]
    chamfer_gt_to_pred = gt_to_pred_min.mean()
    
    # Bidirectional Chamfer Distance
    chamfer_distance = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2.0
    
    return chamfer_distance


def compute_chamfer_distance_numpy(pred_mesh, gt_mesh, num_samples=100000):
    """
    Compute Chamfer Distance between predicted mesh and ground truth mesh using numpy.
    
    Args:
        pred_mesh: Predicted mesh (trimesh object or dict with 'v_pos' key)
        gt_mesh: Ground truth mesh (trimesh object)
        num_samples: Number of points to sample from each mesh
    
    Returns:
        chamfer_distance: Scalar float
    """
    
    # Extract vertices from predicted mesh
    if isinstance(pred_mesh, dict) and 'v_pos' in pred_mesh:
        pred_vertices = pred_mesh['v_pos']
        if isinstance(pred_vertices, torch.Tensor):
            pred_vertices = pred_vertices.detach().cpu().numpy()
    elif hasattr(pred_mesh, 'vertices'):
        pred_vertices = pred_mesh.vertices
    else:
        raise ValueError("pred_mesh must be a trimesh object or dict with 'v_pos' key")
    
    # Extract vertices from ground truth mesh
    if hasattr(gt_mesh, 'vertices'):
        gt_vertices = gt_mesh.vertices
    else:
        raise ValueError("gt_mesh must be a trimesh object")
    
    # Ensure numpy arrays
    if not isinstance(pred_vertices, np.ndarray):
        pred_vertices = np.array(pred_vertices)
    if not isinstance(gt_vertices, np.ndarray):
        gt_vertices = np.array(gt_vertices)
    
    # Sample points uniformly from both meshes
    pred_points = pred_vertices
    gt_points = gt_vertices
    
    # If meshes have too many points, sample them
    if len(pred_points) > num_samples:
        indices = np.random.choice(len(pred_points), num_samples, replace=False)
        pred_points = pred_points[indices]
    
    if len(gt_points) > num_samples:
        indices = np.random.choice(len(gt_points), num_samples, replace=False)
        gt_points = gt_points[indices]
    
    # Compute one-way Chamfer Distance: pred -> gt
    # For each pred point, find distance to nearest gt point
    pred_to_gt_distances = []
    for pred_pt in pred_points:
        # Compute L2 distances to all gt points
        distances = np.linalg.norm(gt_points - pred_pt[np.newaxis, :], axis=1)
        min_dist = distances.min()
        pred_to_gt_distances.append(min_dist)
    
    chamfer_pred_to_gt = np.mean(pred_to_gt_distances)
    
    # Compute one-way Chamfer Distance: gt -> pred
    # For each gt point, find distance to nearest pred point
    gt_to_pred_distances = []
    for gt_pt in gt_points:
        # Compute L2 distances to all pred points
        distances = np.linalg.norm(pred_points - gt_pt[np.newaxis, :], axis=1)
        min_dist = distances.min()
        gt_to_pred_distances.append(min_dist)
    
    chamfer_gt_to_pred = np.mean(gt_to_pred_distances)
    
    # Bidirectional Chamfer Distance
    chamfer_distance = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2.0
    
    return float(chamfer_distance)


def compute_chamfer_distance_numpy_vectorized(pred_mesh, gt_mesh, num_samples=100000, batch_size=1000):
    """
    Compute Chamfer Distance using numpy with vectorized operations (faster).
    
    Args:
        pred_mesh: Predicted mesh (trimesh object or dict with 'v_pos' key)
        gt_mesh: Ground truth mesh (trimesh object)
        num_samples: Number of points to sample from each mesh
        batch_size: Batch size for distance computation to avoid memory issues
    
    Returns:
        chamfer_distance: Scalar float
    """
    
    # Extract vertices from predicted mesh
    if isinstance(pred_mesh, dict) and 'v_pos' in pred_mesh:
        pred_vertices = pred_mesh['v_pos']
        if isinstance(pred_vertices, torch.Tensor):
            pred_vertices = pred_vertices.detach().cpu().numpy()
    elif hasattr(pred_mesh, 'vertices'):
        pred_vertices = pred_mesh.vertices
    else:
        raise ValueError("pred_mesh must be a trimesh object or dict with 'v_pos' key")
    
    # Extract vertices from ground truth mesh
    if hasattr(gt_mesh, 'vertices'):
        gt_vertices = gt_mesh.vertices
    else:
        raise ValueError("gt_mesh must be a trimesh object")
    
    # Ensure numpy arrays
    if not isinstance(pred_vertices, np.ndarray):
        pred_vertices = np.array(pred_vertices)
    if not isinstance(gt_vertices, np.ndarray):
        gt_vertices = np.array(gt_vertices)
    
    # Sample points uniformly from both meshes
    pred_points = pred_vertices
    gt_points = gt_vertices
    
    # If meshes have too many points, sample them
    if len(pred_points) > num_samples:
        indices = np.random.choice(len(pred_points), num_samples, replace=False)
        pred_points = pred_points[indices]
    
    if len(gt_points) > num_samples:
        indices = np.random.choice(len(gt_points), num_samples, replace=False)
        gt_points = gt_points[indices]
    
    # Compute one-way Chamfer Distance: pred -> gt (batched to avoid memory issues)
    pred_to_gt_min_dists = []
    for i in range(0, len(pred_points), batch_size):
        batch_pred = pred_points[i:i+batch_size]
        # Compute pairwise L2 distances: [batch_size, N_gt]
        diff = batch_pred[:, np.newaxis, :] - gt_points[np.newaxis, :, :]  # [batch, N_gt, 3]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # [batch, N_gt]
        min_dists = distances.min(axis=1)  # [batch]
        pred_to_gt_min_dists.append(min_dists)
    
    pred_to_gt_min_dists = np.concatenate(pred_to_gt_min_dists)
    chamfer_pred_to_gt = pred_to_gt_min_dists.mean()
    
    # Compute one-way Chamfer Distance: gt -> pred (batched)
    gt_to_pred_min_dists = []
    for i in range(0, len(gt_points), batch_size):
        batch_gt = gt_points[i:i+batch_size]
        # Compute pairwise L2 distances: [batch_size, N_pred]
        diff = batch_gt[:, np.newaxis, :] - pred_points[np.newaxis, :, :]  # [batch, N_pred, 3]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # [batch, N_pred]
        min_dists = distances.min(axis=1)  # [batch]
        gt_to_pred_min_dists.append(min_dists)
    
    gt_to_pred_min_dists = np.concatenate(gt_to_pred_min_dists)
    chamfer_gt_to_pred = gt_to_pred_min_dists.mean()
    
    # Bidirectional Chamfer Distance
    chamfer_distance = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2.0
    
    return float(chamfer_distance)




