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

import os
import random
import json
import torch
from tqdm import tqdm
from gaussian_splatting.utils.system_utils import searchForMaxIteration
from gaussian_splatting.utils.pose_utils import pose_spherical
from gaussian_splatting.scene.dataset_readers import sceneLoadTypeCallbacks
from gaussian_splatting.scene.sparseGaGS import SparseGaGSGaussianModel
from gaussian_splatting.arguments import ModelParams
import numpy as np 
from gaussian_splatting.utils.camera_utils import cameraList_from_camInfos, cameraDynamic_to_JSON
import copy

__all__ = ['Scene', 'SparseGaGSGaussianModel']

class Scene:
    def __init__(self, training_args, other_args : ModelParams, config, gaussians, load_iteration=None, shuffle=True, if_pretrain=False, pretrain_path=None, given_scale=1.0, given_center=[0, 0, 0], resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = other_args.model_path
        os.makedirs(self.model_path, exist_ok=True)
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(pretrain_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        self.train_cameras = {}
        self.original_train_cameras = {}
        self.val_cameras = {}
        self.test_cameras = {}
        self.predict_cameras = {}
        
        transforms_path = os.path.join(other_args.source_path, "transforms_train.json")
        if not os.path.isfile(transforms_path):
            raise FileNotFoundError(f"D-NeRF transforms not found: {transforms_path}")
        scene_info = sceneLoadTypeCallbacks["dynamic_synthetic"](other_args.source_path, config, other_args.eval)
    
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.val_cameras:
                camlist.extend(scene_info.val_cameras)

            for id, cam in enumerate(camlist):
                json_cams.append(cameraDynamic_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # Save original order before shuffling
        original_train_cameras = copy.deepcopy(scene_info.train_cameras)
        
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, other_args)
            self.original_train_cameras[resolution_scale] = cameraList_from_camInfos(original_train_cameras, resolution_scale, other_args)
            print("Loading Val Cameras")
            self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, other_args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, other_args)
            self.predict_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.predict_cameras, resolution_scale, other_args)
        
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, config, other_args)
        
        if self.loaded_iter:
            if not if_pretrain:
                new_path = pretrain_path or self.model_path
                if os.path.exists(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "gs_asset.ply")):
                    self.gaussians.load_gs(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "gs_asset.ply"))

                if os.path.exists(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")):
                    self.gaussians.load_ply(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
                    
                if os.path.exists(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter),  "checkpoint.pth")):
                    self.gaussians.load_mlp_checkpoints(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "checkpoint.pth"))
            
            else:
                new_path = pretrain_path 
                
                if os.path.exists(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")):
                    print("!!!using Pretrained GS:", new_path, str(self.loaded_iter))
                    if hasattr(self.gaussians, 'load_gs_anchors'):
                        self.gaussians.load_gs_anchors(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
                    elif hasattr(self.gaussians, 'load_ply'):
                        self.gaussians.load_ply(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
                else:
                    raise FileNotFoundError(
                        f"GS point cloud not found under {new_path}/point_cloud/iteration_{self.loaded_iter}"
                    )

                if os.path.exists(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "mlp_deform.pth")):
                    print("!!!Loading MLP Deform")
                    self.gaussians.trans_vox.load_state_dict(torch.load(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "mlp_deform.pth"))['mlp_deform'])
                
                if os.path.exists(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "tineuvox.pth")):
                    print("!!!Loading TiNeuVox")
                    self.gaussians.trans_vox.load_state_dict(torch.load(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "tineuvox.pth"))['trans_vox'])
                
                # self.gaussians.create_from_pcd_dynamic(training_args, other_args)
                
                # if  os.path.exists(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter),  "checkpoint.pth")):
                #     print("!!!Loading Full Pretrained GS")
                    
                #     self.gaussians.load_mlp_checkpoints(os.path.join(new_path, "point_cloud", "iteration_" + str(self.loaded_iter), "checkpoint.pth"))
                
                

        # else:

    @torch.no_grad()
    def save(self, iteration, only_gs=False, time=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if time is not None:
            point_cloud_path_with_time = os.path.join(point_cloud_path, "time_{:.4f}".format(time))
            os.makedirs(point_cloud_path, exist_ok=True)
        else:
            point_cloud_path_with_time = point_cloud_path
        
        # model_save_path = os.path.join(self.model_path, "transform_field_model/iteration_{}".format(iteration))
        ### GS asset save ###
        if hasattr(self.gaussians, 'save_gs') and callable(getattr(self.gaussians, 'save_gs')):
            self.gaussians.save_gs(os.path.join(point_cloud_path_with_time, f"point_cloud.ply"), time=time)

        ### GS anchor save ### 
        if hasattr(self.gaussians, 'save_canon_gs') and callable(getattr(self.gaussians, 'save_canon_gs')):
            self.gaussians.save_canon_gs(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        if hasattr(self.gaussians, 'save_sparse_vox') and callable(getattr(self.gaussians, 'save_sparse_vox')):
            self.gaussians.save_sparse_vox(os.path.join(point_cloud_path, "sparse_vox.ply"))

        if not only_gs:
            ### Scaffold GS save ###
            if hasattr(self.gaussians, 'save_scaffold_ply') and callable(getattr(self.gaussians, 'save_scaffold_ply')):
                self.gaussians.save_scaffold_ply(os.path.join(point_cloud_path, "scaffold_gs.ply"))
            
            # ### Scaffold MLP save ###
            # if hasattr(self.gaussians, 'save_mlp_checkpoints') and callable(getattr(self.gaussians, 'save_mlp_checkpoints')):
            #     self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))
        
        if hasattr(self.gaussians, 'trans_vox'):
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}/tineuvox.pth".format(iteration))
            os.makedirs(os.path.dirname(point_cloud_path), exist_ok=True)                 
            torch.save({
                'trans_vox': self.gaussians.trans_vox.state_dict(),
            }, point_cloud_path)






    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getOriginalTrainCameras(self, scale=1.0):
        return self.original_train_cameras[scale]

    def getValCameras(self, scale=1.0):
        return self.val_cameras[scale]
    
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getPredictCameras(self, scale=1.0):
        return self.predict_cameras[scale]
    
    def getCameras_time_variation(self, frame_pose, scale=1.0, duration=2, fps=30):
        '''
        Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, t=cam_info.t, data_device=args.data_device)
        '''
        cam = self.test_cameras[scale][frame_pose]
        self.Cameras_time_variation = []
        time_list = torch.linspace(0, 1, duration*fps)
        
        for time in time_list:
            cam_time_variation = copy.deepcopy(cam)
            cam_time_variation.t = time

            self.Cameras_time_variation.append(cam_time_variation)
        
        return self.Cameras_time_variation
    
    def getInterpolate_allCameras(self, scale=1.0, frame=150):
        cam = self.train_cameras[scale]
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
        to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
        self.Cameras_all_variation = []
        idx = torch.randint(0, len(cam), (1,)).item()
        view = cam[idx]
        for i, pose in enumerate(tqdm(render_poses, desc='get cam ....')):
            fid = i/(frame-1)
            matrix = np.linalg.inv(np.array(pose))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            cam_all_variation = copy.deepcopy(view)
            cam_all_variation.t = fid
            cam_all_variation.reset_extrinsic(R, T)
            self.Cameras_all_variation.append(cam_all_variation)
        return self.Cameras_all_variation
    
    def getPoseinterpolateCameras(self, scale=1.0):
        cam = self.train_cameras[scale]
        frame = 150
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
        to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
        self.Cameras_pose_variation = []
        idx = torch.randint(0, len(cam), (1,)).item()
        view = cam[idx]
        for i, pose in enumerate(tqdm(render_poses, desc='get cam ....')):
            fid = 0.5
            matrix = np.linalg.inv(np.array(pose))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            cam_all_variation = copy.deepcopy(view)
            cam_all_variation.t = fid
            cam_all_variation.reset_extrinsic(R, T)
            self.Cameras_pose_variation.append(cam_all_variation)
        return self.Cameras_pose_variation
        
        
