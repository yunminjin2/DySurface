import os
import json
import numpy as np
import imageio
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

from instant_nsr.models.ray_utils import get_ray_directions_dnerf
from instant_nsr.utils.misc import get_rank
from instant_nsr.datasets import register

import cv2
import trimesh

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def rodrigues_mat_to_rot(R):
  eps =1e-16
  trc = np.trace(R)
  trc2 = (trc - 1.)/ 2.
  #sinacostrc2 = np.sqrt(1 - trc2 * trc2)
  s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if (1 - trc2 * trc2) >= eps:
    tHeta = np.arccos(trc2)
    tHetaf = tHeta / (2 * (np.sin(tHeta)))
  else:
    tHeta = np.real(np.arccos(trc2))
    tHetaf = 0.5 / (1 - tHeta / 6)
  omega = tHetaf * s
  return omega

def rodrigues_rot_to_mat(r):
  wx,wy,wz = r
  theta = np.sqrt(wx * wx + wy * wy + wz * wz)
  a = np.cos(theta)
  b = (1 - np.cos(theta)) / (theta*theta)
  c = np.sin(theta) / theta
  R = np.zeros([3,3])
  R[0, 0] = a + b * (wx * wx)
  R[0, 1] = b * wx * wy - c * wz
  R[0, 2] = b * wx * wz + c * wy
  R[1, 0] = b * wx * wy + c * wz
  R[1, 1] = a + b * (wy * wy)
  R[1, 2] = b * wy * wz - c * wx
  R[2, 0] = b * wx * wz - c * wy
  R[2, 1] = b * wz * wy + c * wx
  R[2, 2] = a + b * (wz * wz)
  return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_dnerf_data(basedir, half_res=True, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        times = []
        skip = testskip
            
        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip])-1)
            times.append(cur_time)

        assert times[0] == 0, "Time must start at 0"

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)
    
    [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    torch.linspace(0., 1., render_poses.shape[0])
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res



class BlenderDynamicDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        transform_file = os.path.join(self.config.root_dir, f"transforms_train.json") if self.split in ['train', 'predict'] else os.path.join(self.config.root_dir, f"transforms_{self.split}.json")
        
        with open(transform_file, 'r') as f:
            meta = json.load(f)
        
        # Load val transforms when split is train
        # if self.split == 'train':
            # val_transform_file = os.path.join(self.config.root_dir, "transforms_val.json")
            # if os.path.exists(val_transform_file):
            #     with open(val_transform_file, 'r') as f:
            #         meta_val = json.load(f)
            #     # Merge val frames into meta
                # if 'frames' in meta and 'frames' in meta_val:
                #     meta['frames'].extend(meta_val['frames'])
        

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane

        # NOTE: transforms_*.json camera_angle_x is defined for the original resolution (W, H).
        # If we downscale images (e.g., 800 -> 400), intrinsics must be scaled by the same factor.
        camera_angle_x = float(meta['camera_angle_x'])
        focal_orig = 0.5 * W / np.tan(0.5 * camera_angle_x)
        scale = float(self.w) / float(W)
        self.focal = focal_orig * scale

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_dnerf(self.w, self.h, self.focal).to(self.rank) # (h, w, 3)           
        # self.directions = get_ray_directions_nerf(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           
        
        self.all_c2w, self.all_images, self.all_fg_masks, self.all_times = [], [], [], []

        if split == "predict":
            if config.predict._360:
                num_frames = config.predict.num_frames
                
                # 1. 기준이 되는 view_id의 카메라 정보 가져오기
                frame = meta['frames'][config.predict.view_id]
                c2w_orig = np.array(frame['transform_matrix'])
                
                # 2. 카메라 궤적 파라미터 계산
                # 원점(객체)으로부터의 거리(radius)를 기존 카메라 위치에서 추출
                radius = np.linalg.norm(c2w_orig[:3, 3])
                phi = -30.0 # 하향 30도 (표준 Blender 데이터셋 세팅)
                
                # 3. 360도 회전을 위한 방위각(theta) 배열 생성 (-180 ~ 180도)
                # 360도 루프를 위해 마지막 각도는 생략(endpoint=False 효과)
                angles = np.linspace(-180, 180, num_frames + 1)[:-1]
                
                render_poses = []
                for angle in angles:
                    # pose_spherical은 4x4 텐서를 반환하므로 [:3, :4]만 슬라이싱
                    pose = pose_spherical(angle, phi, radius)[:3, :4]
                    render_poses.append(pose)
                
                self.all_c2w = torch.stack(render_poses, dim=0).float().to(self.rank)
            
                # 4. 시간(Time) 설정
                # [옵션 A] 카메라가 도는 동안 객체도 함께 움직이게 할 경우 (기존 로직 유지)
                self.all_times = (torch.arange(num_frames, dtype=torch.float32) / (num_frames - 1)).to(self.rank)
                
                # [옵션 B] 특정 시간에 객체를 멈춰두고(예: t=0.5) 카메라만 회전시킬 경우 (Bullet-time)
                # 사용하려면 위 [옵션 A]를 주석 처리하고 아래 줄의 주석을 해제하세요.
                # frozen_time = 0.5
                # self.all_times = torch.full((num_frames,), frozen_time, dtype=torch.float32).to(self.rank)
            else:
                frame = meta['frames'][config.predict.view_id]
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                self.all_c2w = c2w[None,].repeat(config.predict.num_frames, 1, 1).float().to(self.rank)
                self.all_times = (torch.arange(config.predict.num_frames, dtype=torch.float32) / (config.predict.num_frames - 1)).to(self.rank)
            
        else:
            for i, frame in enumerate(meta['frames']):
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                
                self.all_c2w.append(c2w)
                img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}" + '.png' )
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                
                _time = torch.tensor(frame['time']).float()

                self.all_fg_masks.append(img[..., -1]) # (h, w)
                self.all_images.append(img[...,:3])
                self.all_times.append(_time)

            self.all_c2w, self.all_images, self.all_fg_masks, self.all_times= \
                torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
                torch.stack(self.all_images, dim=0).float().to(self.rank), \
                torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
                torch.stack(self.all_times, dim=0).float().to(self.rank)
           
        
        # Load ground truth mesh for Chamfer Distance evaluation
        self.gt_mesh = None
        self.gt_vertices = None
        self.gt_faces = None
        
        if self.split == 'test':
            if hasattr(self.config, 'gt_mesh_path'):
                obj_path = self.config.gt_mesh_path
                if os.path.exists(obj_path):
                    try:
                        self.gt_mesh = trimesh.load(obj_path, force='mesh')
                        self.gt_vertices = torch.from_numpy(self.gt_mesh.vertices).float().to(self.rank)
                        self.gt_faces = torch.from_numpy(self.gt_mesh.faces).long().to(self.rank)
                        if self.rank == 0:
                            print(f"[INFO] Loaded ground truth mesh from {obj_path}")
                            print(f"[INFO] Mesh vertices: {self.gt_vertices.shape[0]}, faces: {self.gt_faces.shape[0]}")
                    except Exception as e:
                        if self.rank == 0:
                            print(f"[WARNING] Failed to load ground truth mesh from {obj_path}: {e}")




class BlenderDynamicDataset(Dataset, BlenderDynamicDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_c2w)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderDynamicIterableDataset(IterableDataset, BlenderDynamicDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@register('blender_dynamic')
class BlenderDynamicDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderDynamicIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDynamicDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDynamicDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDynamicDataset(self.config, self.config.predict_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=int(self.config.get('num_workers', 0)),
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
