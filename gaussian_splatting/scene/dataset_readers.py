# Copyright (C) 2023, Inria GRAPHDECO research group.
# Research and evaluation use under LICENSE.md.

import copy
import json
import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from gaussian_splatting.utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal, getWorld2View2
from gaussian_splatting.utils.sh_utils import SH2RGB

class CameraInfoDynamic(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    t: np.float64


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    val_cameras: list
    test_cameras: list
    predict_cameras: list
    nerf_normalization: dict
    ply_path: str


def generate_360_predict_cameras(base_cam, num_frames, phi=-30.0):
    """
    주어진 기준 카메라(base_cam)의 반경을 바탕으로 360도 회전하는 카메라 객체 리스트를 생성합니다.
    """
    w2c_orig = np.eye(4, dtype=np.float32)
    w2c_orig[:3, :3] = base_cam.R.T  
    w2c_orig[:3, 3] = base_cam.T
    
    cam_center = np.linalg.inv(w2c_orig)[:3, 3]
    radius = np.linalg.norm(cam_center)
    
    angles = np.linspace(-180, 180, num_frames + 1)[:-1]
    predict_cam_infos = []

    for i, angle in enumerate(angles):
        theta_rad = angle / 180. * np.pi
        phi_rad = phi / 180. * np.pi
        
        t_mat = np.eye(4, dtype=np.float32)
        t_mat[2, 3] = radius
        
        r_phi = np.eye(4, dtype=np.float32)
        r_phi[1, 1] = np.cos(phi_rad); r_phi[1, 2] = -np.sin(phi_rad)
        r_phi[2, 1] = np.sin(phi_rad); r_phi[2, 2] = np.cos(phi_rad)
        
        r_theta = np.eye(4, dtype=np.float32)
        r_theta[0, 0] = np.cos(theta_rad); r_theta[0, 2] = -np.sin(theta_rad)
        r_theta[2, 0] = np.sin(theta_rad); r_theta[2, 2] = np.cos(theta_rad)
        
        coord_trans = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        c2w = coord_trans @ r_theta @ r_phi @ t_mat
        c2w[:, 1:3] *= -1
        
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3]) 
        T = w2c[:3, 3]
        
        time_val = i / (num_frames - 1) if num_frames > 1 else 0.0
        
        cam_i = copy.copy(base_cam)
        cam_i = cam_i._replace(R=R, T=T, t=time_val)
        predict_cam_infos.append(cam_i)
        
    return predict_cam_infos


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)   # (3, 301)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)   # (1, 301)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    
    return {"translate": translate, "radius": radius}


def fetchPly(path, scene_center=None, scene_scale=1.0):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if scene_center is not None:
        positions = (positions - scene_center) * scene_scale
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readCamerasFromTransformsDynamicSynthetic(path, transformsfile, white_background, config=None, extension=".png"):
    cam_infos = []
    
    wh = [800, 800] if config == None or (not hasattr(config.dataset, 'img_wh')) else config.dataset.img_wh
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            axis_align = np.diag([1, -1, -1, 1])
            c2w_colmap = c2w @ axis_align
            w2c_colmap = np.linalg.inv(c2w_colmap)
            R_colmap = w2c_colmap[:3, :3]
            T_colmap = w2c_colmap[:3,  3] 
            R = R_colmap.T
            T = T_colmap

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # arr = cv2.resize(arr, (400, 400), interpolation=cv2.INTER_AREA)
            arr = cv2.resize(arr, wh, interpolation=cv2.INTER_AREA)
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            mask = norm_data[..., 3:].astype(np.float32) if config.dataset.apply_mask else None
            
            if mask is not None:
                mask = cv2.resize(mask, wh, interpolation=cv2.INTER_AREA)[..., None]
            
            
            fovy = focal2fov(fov2focal(fovx, image.size[0] // 2), image.size[1] // 2)
            FovY = fovy 
            FovX = fovx
            t = np.float64((frame["time"]))
            #t = np.float32(t)
            cam_infos.append(CameraInfoDynamic(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mask, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], t=t))
    
    return cam_infos


def readDynamicSyntheticSceneInfo(path, config, eval, extension=".png"):
    white_background = config.model.background_color == 'white' 
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsDynamicSynthetic(path, "transforms_train.json", white_background, config=config, extension=extension)
    print("Reading Val Transforms")
    val_cam_infos = readCamerasFromTransformsDynamicSynthetic(path, "transforms_val.json", white_background, config=config, extension=extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsDynamicSynthetic(path, "transforms_test.json", white_background, config=config, extension=extension)

    if config.dataset.predict._360:
        predict_cam_infos = generate_360_predict_cameras(
            base_cam=train_cam_infos[config.dataset.predict.view_id],
            num_frames=config.dataset.predict.num_frames,
            phi=-30.0
        )
    else:
        predict_cam_infos = []
        for t in range(config.dataset.predict.num_frames):
            cam_i = copy.copy(train_cam_infos[config.dataset.predict.view_id])
            cam_i = cam_i._replace(t=t/(config.dataset.predict.num_frames -1))
            predict_cam_infos.append(cam_i)

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 150_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           predict_cameras=predict_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {"dynamic_synthetic": readDynamicSyntheticSceneInfo}
