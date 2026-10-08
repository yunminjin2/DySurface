import importlib

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

import os
import copy
import numpy as np
import cv2
import random
from tqdm import tqdm 
import uuid
import math
from argparse import Namespace

from instant_nsr.models.utils import save_ply_with_color
from instant_nsr.systems.mesh_utils import get_normals
from instant_nsr.models.ray_utils import get_rays
from instant_nsr.systems.base import BaseSystem
from instant_nsr.systems.criterions import PSNR, binary_cross_entropy
from instant_nsr.utils.loss_utils import l1_loss, ssim
from instant_nsr.utils.visualize_utils import convert2video
from instant_nsr.systems import register

from gaussian_splatting.scene import Scene, SparseGaGSGaussianModel
from gaussian_splatting import gaussian_renderer
from gaussian_splatting.lpipsPyTorch import lpips
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.sh_utils import SH2RGB
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


# RENDER_FUNC = gaussian_renderer.render
# RENDER_FUNC = gaussian_renderer.render_gags
RENDER_FUNC = gaussian_renderer.render_dynamic
DEFORM_FUNC = gaussian_renderer.deform_points
SIMPLE_RENDER_FUNC = gaussian_renderer.simple_render


LPIPS_MODEL = 'alex'  ##### alex | squeeze | vgg #####

class NaNGuardCallback(Callback):
    """Log grad norm and skip optimizer steps that have non-finite gradients."""
    def on_after_backward(self, trainer, pl_module):
        total_sq = 0.0
        for n, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            finite_mask = torch.isfinite(g)
            if finite_mask.any():
                total_sq += (g[finite_mask].norm() ** 2).item()
        pl_module.log("debug/grad_norm", math.sqrt(total_sq) if total_sq > 0 else 0.0, prog_bar=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):
        has_nan = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in pl_module.parameters()
        )
        if has_nan:
            # Drop all grads so this step is skipped
            for g in optimizer.param_groups:
                for param in g["params"]:
                    if param.grad is not None:
                        param.grad = None


def _cfg_to_dict(cfg_node):
    if cfg_node is None:
        return {}
    if hasattr(cfg_node, 'items'):
        return {k: _cfg_to_dict(v) for k, v in cfg_node.items()}
    if isinstance(cfg_node, (list, tuple)):
        return [_cfg_to_dict(v) for v in cfg_node]
    if hasattr(cfg_node, '__iter__') and not isinstance(cfg_node, (str, bytes)):
        return [_cfg_to_dict(v) for v in cfg_node]
    return cfg_node

@torch.no_grad()
def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, if_scaffold=False, dynamic_decode=False,):
    if iteration % 10 == 0:
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)

        if wandb is not None:
            try:
                wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
            except Exception as e:
                print(f"Failed to log to wandb: {e}")
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getPredictCameras()},
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        predict_img_list = []

        
        for config in validation_configs:
            
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ref_psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    render_pkg = renderFunc(
                        viewpoint, 
                        scene.gaussians, 
                        *renderArgs, 
                        return_depth=True,
                        return_normal=True,
                        dynamic_decode=dynamic_decode,
                    )
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    # ref_image = torch.clamp(render_pkg['refined_render'], 0.0, 1.0)
                    depth_gs = render_pkg["render_depth"]
                    depth_gs = depth_gs/depth_gs.max()
                    normal_gs = render_pkg["render_depth_normal"]
                    
                    if config['name'] in ['train', 'test']:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        
                        if tb_writer and (idx < 38):
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/normal".format(viewpoint.image_name), normal_gs[None], global_step=iteration)
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_gs[None], global_step=iteration)
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                            if wandb:
                                render_image_list.append(image[None])
                                errormap_list.append((gt_image[None]-image[None]).abs())

                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                                if wandb:
                                    gt_image_list.append(gt_image[None])
                        
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        # ref_psnr_test += psnr(ref_image, gt_image).mean().double()
                        ssim_test += ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                        
                        # Calculate LPIPS for GS during training
                        gs_render = image.unsqueeze(0)  # [1, 3, H, W]
                        gt_render = gt_image.unsqueeze(0)  # [1, 3, H, W]
                        lpips_test += lpips(gs_render, gt_render, net_type=LPIPS_MODEL).mean().double()

                    else:
                        # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        # save_image = torch.cat([gt_image,image], dim=2)
                        save_image = torch.cat([image], dim=2)
                        predict_img_list.append(
                            (save_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
                        )



                if config['name'] in ['train', 'test']:
                    psnr_test /= len(config['cameras'])
                    ref_psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))


                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    # Log metrics to wandb
                    if wandb is not None:
                        try:
                            wandb.log({
                                f"gs/{config['name']}/l1": l1_test.item(),
                                f"gs/{config['name']}/psnr": psnr_test.item(), 
                                f"gs/{config['name']}/lpips": lpips_test.item(),
                                f"gs/{config['name']}/ssim": ssim_test.item(),
                            })
                        except Exception as e:
                            print(f"Failed to log metrics to wandb: {e}")
                else:
                    convert2video(f"{scene.model_path}/gs_{iteration}", predict_img_list, fps=6, ext=['gif'])

        if tb_writer:
            # tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_xyz.shape[0] if hasattr(scene.gaussians, 'get_xyz') else scene.gaussians.get_anchor.shape[0], iteration)
        
        torch.cuda.empty_cache()

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 

    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


@register('sparse-gags-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__(config)
        self.custom_global_step = 0  # checkpoint offset; added to global_step for absolute iteration
        self.geometry_awared_control = False
        self.mesh = None
        self.gaussians = None
        # Progressive time curriculum (reintroduced)
        self.use_progressive_training = bool(getattr(self.config.system, 'use_progressive_training', False))
        self.progressive_time_start = float(getattr(self.config.system, 'progressive_time_start', 0.21))  # initial |t| window radius
        self.progressive_time_max = float(getattr(self.config.system, 'progressive_time_max', 1.0))      # max |t| window radius
        # New schedule parameters; fall back to legacy ramp if provided
        legacy_ramp = getattr(self.config.system, 'progressive_time_ramp_iters', None)
        # Allow alias keys: progressive_step_start / progressive_step_end
        start_alias = getattr(self.config.system, 'progressive_step_start', None)
        end_alias = getattr(self.config.system, 'progressive_step_end', None)
        self._canonical_feat_lr_scaled = False
        self.progressive_start_iter = int(getattr(self.config.system, 'progressive_start_iter', start_alias if start_alias is not None else 0))
        self.progressive_end_iter = int(getattr(self.config.system, 'progressive_end_iter', end_alias if end_alias is not None else (legacy_ramp if legacy_ramp is not None else 5000)))
        if self.progressive_end_iter <= self.progressive_start_iter:
            # Degenerate schedule -> disable progressive expansion
            self.progressive_end_iter = self.progressive_start_iter
        self._last_logged_prog_thr_step = -1
        # Option to disable second-order grads (avoid grid_sample 3D double backward)

        self.add_network_flag = 0
        self.wandb = None
        self.metrics_table = None

        self.dr = importlib.import_module('nvdiffrast.torch')
        if self.config.cmd_args.get('use_wandb', False):
            try:
                import wandb
                dataset_name = self.config.dataset.root_dir.split('/')[-1] if hasattr(self.config.dataset, 'root_dir') else 'unknown'
                exp_name = self.config.trial_name
                
                wandb.login()
                
                # Create a simple config dict for wandb
                config_dict = {
                    'dataset_name': dataset_name,
                    'trial_name': exp_name,
                    'exp_dir': str(self.config.exp_dir),
                    'save_dir': str(self.config.save_dir),
                    'ckpt_dir': str(self.config.ckpt_dir),
                    'use_wandb': True
                }
                
                wandb.init(
                    project=f"DynamicGSDF-{dataset_name}",
                    name=exp_name,
                    settings=wandb.Settings(start_method="fork"),
                    config=config_dict
                )
                self.wandb = wandb
                print(f"Using Wandb for logging: {wandb.run.url}")
                
            except Exception as e:
                print(f"Failed to initialize Wandb: {e}")
                print("Falling back to TensorBoard only")
                self.wandb = None
        
        if self.config.model.background_color == 'white':
            self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            self.model.background_color = self.background.to(self.rank)
        elif self.config.model.background_color == 'black':
            self.background = torch.zeros((3,), dtype=torch.float32, device=self.rank)
        elif self.config.model.background_color == 'random':
            self.background = torch.rand((3,), dtype=torch.float32, device=self.rank)
        else:
            raise NotImplementedError
        self.model.background_color = self.background.to(self.rank)
        self.init_step = 0
        self.created_from_mesh = False

        if self.config.model.if_gaussian:
            print(config.dataset.root_dir)

            self.pretrain_step = self.config.system.pretrain_step if hasattr(self.config.system, 'pretrain_step') else 5000
            print('Pretrain Step: ', self.pretrain_step)
            out_path = f"{config.exp_dir}/{config.trial_name}/output/"

            if not hasattr(self.config, 'gaussian_arguments'):
                raise ValueError("Missing `gaussian_arguments` in config. Please define gaussian_arguments.model/pipeline/optimization/runtime in the yaml config.")

            gs_cfg = self.config.gaussian_arguments
            for key in ('model', 'pipeline', 'optimization', 'runtime'):
                if not hasattr(gs_cfg, key):
                    raise ValueError(f"Missing `gaussian_arguments.{key}` in config.")

            model_kwargs = _cfg_to_dict(gs_cfg.model)
            model_kwargs['source_path'] = os.path.abspath(config.dataset.root_dir)
            model_kwargs['model_path'] = out_path
            model_kwargs['resolution'] = int(
                getattr(
                    config.dataset,
                    'gs_resolution',
                    config.dataset.img_downscale,
                )
            )
            model_kwargs['eval'] = bool(config.cmd_args.get('eval', model_kwargs.get('eval', False)))
            self.lp = Namespace(**model_kwargs)

            opt_kwargs = _cfg_to_dict(gs_cfg.optimization)
            self.op = Namespace(**opt_kwargs)

            pipeline_kwargs = _cfg_to_dict(gs_cfg.pipeline)
            self.piplin = Namespace(**pipeline_kwargs)

            runtime_kwargs = _cfg_to_dict(gs_cfg.runtime)
            test_iterations = runtime_kwargs.get('test_iterations')
            save_iterations = runtime_kwargs.get('save_iterations')
            if test_iterations is None or save_iterations is None:
                raise ValueError("Missing `gaussian_arguments.runtime.test_iterations` or `gaussian_arguments.runtime.save_iterations` in config.")
            checkpoint_iterations = runtime_kwargs.get('checkpoint_iterations') or []
            test_iterations = [int(v) for v in test_iterations]
            save_iterations = [int(v) for v in save_iterations]
            checkpoint_iterations = [int(v) for v in checkpoint_iterations]

            self.args = Namespace(
                start_dynamic_iter=self.op.start_dynamic_iter,
                stable_until_iter=self.op.stable_until_iter,
                detect_anomaly=bool(runtime_kwargs.get('detect_anomaly', False)),
                mesh_render=bool(self.config.cmd_args.get('mesh_render', runtime_kwargs.get('mesh_render', False))),
                checkpoint_iterations=list(checkpoint_iterations),
                test_iterations=list(test_iterations),
                save_iterations=list(save_iterations),
                source_path=self.lp.source_path,
                model_path=self.lp.model_path,
            )

            os.makedirs(self.lp.model_path, exist_ok=True)
            print(f'model_path: {self.lp.model_path}')
            self.loggger = get_logger(self.lp.model_path)
            self.loggger.info(f'gaussian_model_args: {self.lp}')
            self.loggger.info(f'gaussian_optimization_args: {self.op}')
            self.loggger.info(f'gaussian_pipeline_args: {self.piplin}')
            self.loggger.info(f'gaussian_runtime_args: {self.args}')
            self.tb_writer = self.prepare_output_and_logger(self.lp)
            
            # safe_state(args.quiet)                        
            # Start GUI server, configure and run training
            # network_gui.init(args.ip, args.port)
            
            torch.autograd.set_detect_anomaly(self.args.detect_anomaly)
            self.config = config

            self.saving_iterations = self.args.save_iterations
            self.testing_iterations = self.args.test_iterations

            self.set_canon_pose = self.config.dataset.canonicalize_poses
            self.canon_t = torch.tensor(self.config.dataset.canon_time).float().cuda()

            self.gaussians = SparseGaGSGaussianModel(self.lp.sh_degree, canon_t=self.canon_t)
            
            self.ema_loss_for_log = 0.0
            self.dataset_size=0
            self.last_iteration_time=0
            

            if self.config.model.using_pretrain:
                # Priority for GS pretrain dir:
                #   1) --resume_gs (explicit CLI override)
                #   2) sibling output/ of --resume's ckpt dir (train.sh: skip 명시 안 해도 자동)
                #   3) config.model.using_pretrain_path
                resume_ckpt = self.config.cmd_args.get('resume', None) if hasattr(self.config, 'cmd_args') else None
                resume_sibling_output = None
                if resume_ckpt:
                    ckpt_dir = os.path.dirname(resume_ckpt)
                    cand = os.path.join(os.path.dirname(ckpt_dir), 'output')
                    if os.path.isdir(cand):
                        resume_sibling_output = cand
                if self.config.cmd_args.resume_gs:
                    gs_pretrain_path = self.config.cmd_args.resume_gs
                elif resume_sibling_output:
                    gs_pretrain_path = resume_sibling_output
                else:
                    gs_pretrain_path = self.config.model.using_pretrain_path
                # Default: load PLY indexed by config.system.pretrain_step.
                # Override: when --resume_gs_iteration is passed (e.g. from time_render.sh) use that instead.
                cli_gs_iter = int(self.config.cmd_args.get('resume_gs_iteration', 0) or 0)
                gs_pretrain_iter = cli_gs_iter if cli_gs_iter > 0 else self.pretrain_step
                print(f'[scene-load] gs PLY dir = {gs_pretrain_path}')
                print(f'[scene-load] gs PLY iter = {gs_pretrain_iter} '
                      f'(cli_resume_gs_iteration={cli_gs_iter}, config_pretrain_step={self.pretrain_step})')

                self.scene = Scene(
                    self.op,
                    self.lp, 
                    self.config, 
                    self.gaussians, 
                    load_iteration=gs_pretrain_iter, 
                    shuffle=True, 
                    if_pretrain=self.config.model.using_pretrain,pretrain_path=gs_pretrain_path, 
                    given_scale=self.config.dataset.neuralangelo_scale, 
                    given_center=self.config.dataset.neuralangelo_center
                )
            
                self.gaussians.training_setup(self.op, self.lp)
                # self.gaussians.training_setup_for_dynamic(self.op, self.lp)
                
                self.gaussians.update_learning_rate(gs_pretrain_iter, self.op.start_dynamic_iter)

                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.viewpoint_candidate = self.scene.getTrainCameras().copy()

                self.progress_bar = tqdm(range(int(gs_pretrain_iter), self.op.iterations), desc="Training progress")               

            #Pretrain Scaffold-GS from scratch.
            else:
                self.scene = Scene(
                    self.op,
                    self.lp, 
                    self.config, 
                    self.gaussians, 
                    shuffle=False, 
                    given_scale=self.config.dataset.neuralangelo_scale,
                    given_center=self.config.dataset.neuralangelo_center
                )
                
                self.progress_bar = tqdm(range(0, self.op.iterations), desc="Training progress")               
                self.gaussians.training_setup(self.op, self.lp)
                
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.viewpoint_candidate = self.scene.getTrainCameras().copy()

                # pretrain scaffold gs
                self.pretrain_gs()

                # self.gaussians.create_from_pcd_dynamic(self.op, self.lp)
                # self.gaussians.training_setup_for_dynamic(self.op, self.lp)

                
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
            
    
            # Refinement is an optimization stage and must not run while
            # constructing a system for validation/prediction. Apart from
            # making validation mutate the loaded GS, doing it here delayed
            # checkpoint loading by several hours.
            if bool(self.config.cmd_args.get('train', False)):
                # refine_gs overwrites the pretrained GS canonical params with the
                # canon_time snapshot and re-fits trans_vox; the backup (better
                # performing) pipeline skipped it, so it is opt-in.
                if bool(self.config.system.get('use_refine_gs', False)):
                    self.refine_gs()
                self.update_gs_geometric()
                self.scene.save(-1, time=0.0)
        
        
        # Save non-refined gs
        
        # training_report(
        #     self.tb_writer, 
        #     self.config.dataset.root_dir.split('/')[-1], 
        #     -1, 
        #     torch.tensor(0), 
        #     torch.tensor(0), 
        #     l1_loss, 
        #     -1, 
        #     self.args.test_iterations, 
        #     self.scene, 
        #     RENDER_FUNC, 
        #     (self.piplin, self.background, self.scene, self.config.model.transform_scaler), 
        #     self.wandb, 
        #     self.loggger,
        #     dynamic_decode=True
        # )

    @property
    def global_epoch(self):
        """Absolute training step: Lightning global_step + checkpoint offset."""
        return self.global_step + self.custom_global_step

    def on_save_checkpoint(self, checkpoint):
        checkpoint['custom_global_step'] = self.custom_global_step

    def on_load_checkpoint(self, checkpoint):
        self.custom_global_step = checkpoint.get('custom_global_step', 0)
        # Predict/validate runs without prior fit have self.global_step == 0 and
        # the codebase never sets custom_global_step during training, so it stays
        # 0. on_*_batch_start uses (custom_global_step or global_step) → 0 →
        # ProgressiveBandHashGrid.update_step(0) leaves only start_level features
        # active, producing a coarse SDF and a low-quality marching-cubes mesh.
        # finite_difference_eps='progressive' gets stuck at the first eps too.
        # Fall back to trainer.max_steps so progressive encodings reach terminal.
        if not self.custom_global_step:
            try:
                self.custom_global_step = int(self.config.trainer.get('max_steps', 0)) or 0
            except Exception:
                pass

    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + (self.config.model.get('num_samples_per_ray_bg', 0) if self.config.model.learned_background else 0))
        self.train_num_rays = self.config.model.train_num_rays

    def gaussian_backend_loss(self, render_pkg, iteration):
        """Optional renderer-specific regularizer; the 3DGS backend uses none."""
        return render_pkg["render"].new_zeros(())


    # GaGS
    def pretrain_gs(self):
        self.args.source_path.split('/')[-1]
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        
        for iteration in range(1, self.op.iterations + 1): 
            iter_start.record()

            ########## 
            if iteration == self.args.start_dynamic_iter:
                self.gaussians.optimizer.add_param_group(
                    {'params': list(self.gaussians.trans_vox.parameters()), 'lr': self.op.network_lr_init, "name": "trans_vox"},
                )
            
                # self.gaussians.optimizer.add_param_group(
                #     {'params': list(self.gaussians.inverse_trans_vox.parameters()), 'lr': self.op.network_lr_init, "name": "inverse_trans_vox"}
                # )
                
                # self.gaussians.update_tineuvox(self.config.dataset.canon_time)
                

            # Original GS
            # self.gaussians.update_learning_rate(iteration) 
            
            # GaGS
            self.gaussians.update_learning_rate(iteration, self.op.start_dynamic_iter)

            if iteration % 1000 == 0:   
                self.gaussians.oneupSHdegree()

            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.dataset_size = len(self.viewpoint_stack)

            idx = random.randint(0, len(self.viewpoint_stack)-1)
            viewpoint_cam = self.viewpoint_stack.pop(idx)
            
            render_pkg = RENDER_FUNC(
                viewpoint_cam, 
                self.gaussians, 
                self.piplin, 
                self.background, 
                self.scene,
                transform_scaler=self.config.model.transform_scaler,
                return_normal=True, 
                return_depth=True,
                return_asset=True,
                dynamic_decode=iteration > self.op.start_dynamic_iter,
            )

            image, viewspace_point_tensor, visibility_filter, mask, gs_normal, radii, gs_depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg['mask'], render_pkg["render_depth_normal"], render_pkg['radii'], render_pkg['render_depth']

            gt_image = viewpoint_cam.original_image.cuda()
            psnr_ = psnr(image, gt_image).mean().double()

            rgb_loss = (1.0 - self.op.lambda_dssim) * (l1_loss(image, gt_image)) + self.op.lambda_dssim * (1.0 - ssim(image, gt_image)) + l1_loss(render_pkg['canon_render'], render_pkg['pred_canon_render']) * 0.4
            backend_loss = self.gaussian_backend_loss(render_pkg, iteration)

            delta_reg_loss = torch.tensor(0.0).cuda()
            if self.set_canon_pose :
                sigma = self.config.dataset.progressive_time_sigma
                t = torch.tensor([viewpoint_cam.t]).float().cuda()
                lambda_delta_loss = 0.5 * ((1/sigma) * torch.exp(-(t - self.canon_t)**2 / (2 * sigma**2)) + 1).item()
            else:
                lambda_delta_loss = self.config.system.loss.lambda_delta_reg
            
            delta_reg_loss += self.gaussians.delta_xyz.abs().mean() * lambda_delta_loss

            
            # Mask
            mask_loss = 0
            if self.config.dataset.apply_mask and mask is not None:
                gt_mask = viewpoint_cam.original_mask.cuda()
                mask_loss = l1_loss(mask, gt_mask)

            loss = rgb_loss * self.op.lambda_rgb + delta_reg_loss * self.op.lambda_delta_reg + mask_loss * self.op.lambda_mask + backend_loss
            loss.backward()
            iter_end.record()
            
            with torch.no_grad():
                # Progress bar
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                if iteration % 10 == 0:
                    self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}",
                                            "psnr":f"{psnr_:.{2}f}",
                                            "points":f"{self.gaussians.get_xyz.shape[0]}",  # gags
                                            "delta":f"{self.gaussians.delta_loss.item():.{5}f}",  # gags                                    
                                            "vp" : f"{self.gaussians.voxel_points}"
                                        })
                    self.progress_bar.update(10)
                if iteration == self.op.iterations:
                    self.progress_bar.close()
                
                if iteration % 50 == 0:           
                    _save_image = torch.cat([gt_image, gt_mask.repeat(3, 1, 1), gs_normal, image], dim=2)
                    cv2.imwrite(self.get_save_path('gs_canonical_preview.png'), (torch.cat([render_pkg['canon_render'], render_pkg['pred_canon_render']], dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
                    cv2.imwrite(self.get_save_path('gs_train_preview.png'), cv2.resize((_save_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8), (_save_image.shape[2], _save_image.shape[1]))[..., ::-1])
            
                # Log and save
                training_report(
                    self.tb_writer,
                    self.config.dataset.root_dir.split('/')[-1],
                    iteration,
                    rgb_loss,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    self.args.test_iterations,
                    self.scene,
                    RENDER_FUNC,
                    (self.piplin, self.background, self.scene, self.config.model.transform_scaler,),
                    self.wandb,
                    self.loggger,
                    dynamic_decode=iteration > self.op.start_dynamic_iter,
                )
                if (iteration in self.saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.scene.save(iteration, time=0.0, only_gs=True)
                    self.scene.save(iteration, time=1.0, only_gs=True)
        
                # Densification
                if iteration < self.op.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                    if iteration > self.op.densify_from_iter and iteration % self.op.densification_interval == 0:
                        size_threshold = 20 if iteration > self.op.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.op.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold, iteration, self.args) # Gags
                        
                        # self.gaussians.update_tineuvox(self.config.dataset.canon_time)
                
                    if iteration % self.op.opacity_reset_interval == 0 or (self.config.model.background_color  == "white" and iteration == self.op.densify_from_iter):
                        self.gaussians.reset_opacity()


                # Optimizer step
                if iteration < self.op.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in self.args.checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpt" + str(iteration) + ".pth")


    def set_canonical_gs(self, snapshot=None):
        """In-place copy a gs_asset snapshot into self.gaussians' canonical params.

        Replaces ``_xyz``/``_scaling``/``_rotation``/``_features_dc``/
        ``_features_rest``/``_opacity`` with the snapshot's values, preserving
        the underlying ``nn.Parameter`` objects (and therefore optimizer state).

        Args:
            snapshot (dict | None): gs_asset dict with keys ``xyz``, ``scaling``
                (raw), ``rotations`` (raw quaternion), ``feature``
                ([N, num_sh, 3] or [N, 3]), ``opacity`` (raw inv-sigmoid).
                Defaults to ``self.canonical_gs``.
        """
        canon_time = float(self.config.dataset.canon_time)
        best_dt = None
        viewpoint_stack = self.scene.getOriginalTrainCameras().copy()
        for i, viewpoint_cam in enumerate(viewpoint_stack):
            dt = abs(float(viewpoint_cam.t) - canon_time)
            if best_dt is None or dt < best_dt:
                render_pkg = RENDER_FUNC(
                    viewpoint_cam,
                    self.gaussians,
                    self.piplin,
                    self.background,
                    self.scene,
                    transform_scaler=self.config.model.transform_scaler,
                    return_depth=True,
                    return_normal=True,
                    dynamic_decode=True,
                    return_asset=True
                )
                best_dt = dt
                self.canonical_gs = render_pkg['gs_asset']

        if snapshot is None:
            snapshot = self.canonical_gs
        assert snapshot is not None, \
            'set_canonical_gs: snapshot is None — populate self.canonical_gs first.'

        with torch.no_grad():
            feat = snapshot['feature'].detach()
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)            # [N, 3] → [N, 1, 3] (DC only)
            self.gaussians._xyz.data.copy_(snapshot['xyz'].detach())
            self.gaussians._scaling.data.copy_(snapshot['scaling'].detach())
            self.gaussians._rotation.data.copy_(snapshot['rotations'].detach())
            self.gaussians._features_dc.data.copy_(feat[:, :1])
            if feat.shape[1] > 1:
                self.gaussians._features_rest.data.copy_(feat[:, 1:])
            self.gaussians._opacity.data.copy_(snapshot['opacity'].detach())
        print('[set_canonical_gs] gaussian canonical params <- snapshot '
              f'(N={self.gaussians._xyz.shape[0]})')


    # GaGS
    def refine_gs(self):
        """Swap self.gaussians' canonical params with self.canonical_gs snapshot,
        then refine trans_vox against the new canonical via the existing
        transformation()/RENDER_FUNC path.

        canonical_gs is a frozen render-time snapshot at canon_t. After
        in-place copy into ``_xyz``/``_scaling``/``_rotation``/``_features_*``/
        ``_opacity``, anything that uses ``self.gaussians`` as canonical
        (transformation, render_dynamic with dynamic_decode=True) automatically
        operates on the new pose. Gaussian params are frozen; only trans_vox
        learns.
        """
        self.set_canonical_gs()

        # ── Freeze gaussian params; only trans_vox trains ──
        for attr in ('_xyz', '_features_dc', '_features_rest',
                     '_scaling', '_rotation', '_opacity'):
            p = getattr(self.gaussians, attr, None)
            if isinstance(p, torch.Tensor):
                p.requires_grad_(False)
        for p in self.gaussians.trans_vox.parameters():
            p.requires_grad_(True)

        # ── Optimizer (trans_vox only) ─────────────────────────────────────
        lr_init  = float(getattr(self.config.system.loss, 'refine_lr',       1e-4))
        lr_final = float(getattr(self.config.system.loss, 'refine_lr_final', 1e-5))
        optimizer = torch.optim.Adam(
            self.gaussians.trans_vox.parameters(),
            lr=lr_init, eps=1e-15, betas=(0.9, 0.999),
        )
        total_steps = int(getattr(self.config.system, 'refine_step', 1500))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps + 1, eta_min=lr_final,
        )

        # ── Viewpoints ─────────────────────────────────────────────────────
        self.viewpoint_stack = self.scene.getOriginalTrainCameras().copy()
        num_frames = len(self.viewpoint_stack)

        # ── Loss weights ───────────────────────────────────────────────────
        w_rgb    = float(self.op.lambda_rgb)
        w_dssim  = float(self.op.lambda_dssim)
        w_mask   = float(self.op.lambda_mask)
        w_offset = float(self.op.lambda_delta_reg)
        
        pbar = tqdm(range(total_steps + 1), desc='refine_gs')
        for iter in pbar:
            idx = random.randint(0, num_frames - 1)
            viewpoint_cam = self.viewpoint_stack[idx]
            gt_image = viewpoint_cam.original_image.cuda()
            gt_mask  = viewpoint_cam.original_mask.cuda()

            # Existing path: transformation() reads new _xyz/_rotation/_scaling.
            render_pkg = RENDER_FUNC(
                viewpoint_cam, self.gaussians,
                self.piplin, self.background, self.scene,
                transform_scaler=self.config.model.transform_scaler,
                return_normal=False, return_depth=False,
                dynamic_decode=True,
            )
            gs_image = render_pkg['render']
            gs_mask  = render_pkg.get('mask', None)

            loss_rgb = (1.0 - w_dssim) * l1_loss(gs_image, gt_image) \
                       + w_dssim * (1.0 - ssim(gs_image, gt_image))
            loss_mask = (l1_loss(gs_mask, gt_mask)
                         if (self.config.dataset.apply_mask and gs_mask is not None)
                         else gs_image.new_zeros(()))
            loss_offset = self.gaussians.delta_xyz.abs().mean()

            loss = w_rgb * loss_rgb + w_mask * loss_mask + w_offset * loss_offset

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(
                f'rgb:{loss_rgb.item():.4f} mask:{float(loss_mask):.4f} offset:{loss_offset.item():.4f}'
            )

            # ── TensorBoard ────────────────────────────────────────────────
            if self.tb_writer is not None:
                tb_step = int(self.global_epoch) + iter
                self.tb_writer.add_scalar('refine_gs/total_loss', loss.item(), tb_step)
                self.tb_writer.add_scalar('refine_gs/lr', optimizer.param_groups[0]['lr'], tb_step)
                self.tb_writer.add_scalar('refine_gs/rgb',    loss_rgb.item(), tb_step)
                self.tb_writer.add_scalar('refine_gs/mask',
                                          loss_mask.item() if torch.is_tensor(loss_mask) else float(loss_mask),
                                          tb_step)
                self.tb_writer.add_scalar('refine_gs/offset', loss_offset.item(), tb_step)

            # ── Visualisation ──────────────────────────────────────────────
            if iter % 10 == 0:
                vis = torch.cat([gt_image, gs_image.detach().clamp(0, 1)], dim=2)
                vis_np = (vis.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(self.get_save_path('refine_gs.png'), vis_np[..., ::-1])
                if self.tb_writer is not None:
                    tb_step = int(self.global_epoch) + iter
                    self.tb_writer.add_image('refine_gs/gt',     gt_image.clamp(0, 1), tb_step)
                    self.tb_writer.add_image('refine_gs/render', gs_image.detach().clamp(0, 1), tb_step)

            # ── Periodic save ──────────────────────────────────────────────
            if iter > 0 and iter % 500 == 0:
                with torch.no_grad():
                    save_step = self.global_epoch + iter
                    self.scene.save(save_step, time=0.0, only_gs=True)


    @torch.no_grad()
    def update_gs_geometric(self,):
        self.gs_image_list = []
        self.gs_render_depth_list = []
        self.gs_depth_list = []
        self.gs_normal_list = []
        self.gs_delta_list = []
        self.gs_voxel = []
        self.gs_vox_inds = []
        self.canonical_gs = None
            
        canon_time = float(self.config.dataset.canon_time)
        best_dt = None
        viewpoint_stack = self.scene.getOriginalTrainCameras().copy()
        for i, viewpoint_cam in enumerate(viewpoint_stack):
            render_pkg = RENDER_FUNC(
                viewpoint_cam,
                self.gaussians,
                self.piplin,
                self.background,
                self.scene,
                transform_scaler=self.config.model.transform_scaler,
                return_depth=True,
                return_normal=True,
                dynamic_decode=True,
                return_asset=True
            )
            
            gs_image, gs_depth, gs_render_depth, gs_normal, alpha, gs_delta = render_pkg['render'], render_pkg['real_depth'], render_pkg['render_depth'], render_pkg['render_depth_normal'], render_pkg['mask'], render_pkg['delta_anchor']
            # gs_vox_inds = render_pkg['gs_asset']['vox_inds']

            # voxel_gs, voxel_gs_pcd, _, _ = self.model.tineuvox.voxelize(self.gaussians.get_transformed_xyz, SH2RGB(self.gaussians.get_color))
            voxel_gs, voxel_gs_pcd, _, _ = self.model.tineuvox.voxelize(self.gaussians.get_transformed_xyz, SH2RGB(self.gaussians.get_color))
            
            dt = abs(float(viewpoint_cam.t) - canon_time)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                self.canonical_gs = render_pkg['gs_asset']

            self.gs_image_list.append(gs_image.cpu())
            self.gs_depth_list.append(gs_depth.cpu())
            self.gs_render_depth_list.append(gs_render_depth.cpu())
            self.gs_normal_list.append(gs_normal.cpu())
            self.gs_delta_list.append(gs_delta.cpu())
            # self.gs_vox_inds.append(gs_vox_inds)
            self.gs_voxel.append(voxel_gs_pcd.cpu())

        self.cn_image_list = []
        self.cn_depth_list = []
        self.cn_normal_list = []
        for i, viewpoint_cam in enumerate(viewpoint_stack):
            render_pkg = RENDER_FUNC(
                viewpoint_cam,
                self.gaussians,
                self.piplin,
                self.background,
                self.scene,
                transform_scaler=self.config.model.transform_scaler,
                return_depth=True,
                return_normal=True,
                gs_asset = self.canonical_gs
            )
            self.cn_image_list.append(render_pkg['render'].cpu())
            self.cn_depth_list.append(render_pkg['real_depth'].cpu())
            self.cn_normal_list.append(render_pkg['render_depth_normal'].cpu())

        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        save_ply_with_color(os.path.join(self.config.ckpt_dir, 'canon_gs.ply'), self.canonical_gs['xyz'])
    
    def preprocess_data(self, batch, stage):
        # Progressive time filtering: restrict sampling to frames with |t| <= threshold
        if stage == 'train' and self.use_progressive_training and hasattr(self.dataset, 'all_times'):
            # Compute threshold based on global_step (Lightning) or internal counter
            global_step = self.global_step if hasattr(self, 'global_step') else 0
            thr = self._current_time_threshold(global_step)
            times = self.dataset.all_times
            valid_mask = times.abs() <= thr + 1e-8
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
            if len(valid_indices) == 0:
                valid_indices = torch.arange(len(times), device=times.device)
            # Logging threshold occasionally
            if global_step % 100 == 0 and global_step != self._last_logged_prog_thr_step:
                self._last_logged_prog_thr_step = global_step
                if hasattr(self, 'log'):
                    self.log('train/prog_time_thr', float(thr), prog_bar=True)
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                if stage == 'train' and self.use_progressive_training and hasattr(self, 'progressive_end_iter') and 'valid_indices' in locals():
                    # sample per-ray indices from restricted set
                    rand_frame_indices = valid_indices[torch.randint(0, len(valid_indices), size=(self.train_num_rays,), device=self.dataset.all_images.device)]
                    index = rand_frame_indices
                else:
                    index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
                
            else:
                if stage == 'train' and self.use_progressive_training and 'valid_indices' in locals():
                    index = valid_indices[torch.randint(0, len(valid_indices), size=(1,), device=self.dataset.all_images.device)]
                else:
                    index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
             
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
            t = self.dataset.all_times[index]

            if self.dataset.apply_mask:
                rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
            
            t_prev = self.dataset.all_times[max(0, index-1)]
            t_next = self.dataset.all_times[min(len(self.dataset.all_times) - 1, index+1)]
            

        elif stage in ['val']:
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            x = torch.arange(
                0, self.dataset.w, device='cuda'
            )
            y = torch.arange(
                0, self.dataset.h, device='cuda'
            )
            
            c2w = self.dataset.all_c2w[index]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            t = self.dataset.all_times[index]

        elif stage in ['predict']:
            x = torch.arange(
                0, self.dataset.w, device='cuda'
            )
            y = torch.arange(
                0, self.dataset.h, device='cuda'
            )
            
            c2w = self.dataset.all_c2w[index]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            t = self.dataset.all_times[index]

        else:
            x = torch.arange(
                0, self.dataset.w, device=self.dataset.all_images.device
            )
            y = torch.arange(
                0, self.dataset.h, device=self.dataset.all_images.device
            )
            c2w = self.dataset.all_c2w[index]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
                        
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            t = self.dataset.all_times[index]

            if self.dataset.apply_mask:
                rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
        # rays = torch.cat([rays_o, rays_d], dim=-1)
        
        if stage in ['train']:
            batch.update({
                'rays': rays,
                'directions': directions,
                'image': self.dataset.all_images[index],
                'rgb': rgb,
                'fg_mask': fg_mask,
                'used_index': index,
                'used_y': y,
                'used_x': x,
                'time': t,
                't_prev': t_prev,
                't_next': t_next
            }) 
            # if self.config.model.if_gaussian and not self.config.model.optimize_gaussian:
            #     batch.update({
            #         'gs_depth': self.gs_depth_list[index].permute(1, 2, 0)[y, x],
            #         'gs_normal': self.gs_normal_list[index].permute(1, 2, 0)[y, x],
            #     })
                
        elif stage in ['val']:
            batch.update({
                'rgb': rgb,
                'directions': directions,
                'used_y': y,
                'used_x': x,
                'rays': rays,
                'time': t,
            })
        elif stage in ['predict']:
            batch.update({
                'directions': directions,
                'used_y': y,
                'used_x': x,
                'rays': rays,
                'time': t,
            })
        else:
            batch.update({
                'rays': rays,
                'used_y': y,
                'used_x': x,
                'directions': directions,
                'rgb': rgb,
                'fg_mask': fg_mask,
                'time': t,
            })

            
    # vector similarity. 
    def cos_similarity_loss(self, a, b):
        return 1.0-((a*b).sum(dim=-1) / (a.norm(dim=-1)*b.norm(dim=-1)+1e-8)).abs().mean()

    def _current_time_threshold(self, step: int):
        """Compute current allowed |t| threshold for progressive curriculum.
        ratio = 0 before start_iter, linearly increases to 1 at end_iter, then clamped.
        Returns float threshold in [start, max].
        """
        if step <= self.progressive_start_iter:
            return self.progressive_time_start
        if self.progressive_end_iter <= self.progressive_start_iter:
            return self.progressive_time_max
        if step >= self.progressive_end_iter:
            return self.progressive_time_max
        ratio = (step - self.progressive_start_iter) / float(self.progressive_end_iter - self.progressive_start_iter)
        return self.progressive_time_start + (self.progressive_time_max - self.progressive_time_start) * ratio

    def forward(self, batch, t, gs, trans_gs_asset=None, gs_depth=None, use_depth_guide=False, anchor=None, debug=False, stable_train=False, concat_ray=False, iteration=None):
        
        return self.model(batch['rays'], gs, trans_gs_asset=trans_gs_asset, gs_depth=gs_depth, use_depth_guide=use_depth_guide, anchor=anchor, time=t, debug=debug, stable_train=stable_train, iteration=iteration)

        
    def _get_joint_gs_step(self):
        return self.global_epoch - self.config.model.start_optimize_gaussian

    def _build_joint_gs_xyz_sdf(self, batch_time, trans_gs_asset):
        gs_positions = self.gaussians.get_xyz.detach()
        device = gs_positions.device
        time_tensor = torch.as_tensor(batch_time, device=device, dtype=gs_positions.dtype).reshape(1)
        bounds = gs_positions.new_full((3,), self.config.model.radius * 2)
        inside_box = torch.logical_and(gs_positions > -bounds, gs_positions < bounds).all(dim=1)
        xyz_sdf = gs_positions.new_full((gs_positions.shape[0],), 100000.0)

        if torch.any(inside_box):
            inside_positions = gs_positions[inside_box]
            inside_xyz_sdf = self.model.geometry(inside_positions, t_emb=time_tensor, with_grad=False, with_feature=False)
            xyz_sdf[inside_box] = inside_xyz_sdf

        return xyz_sdf, inside_box

    def _run_joint_gs_densification(self, batch_time, joint_gs_step, viewspace_point_tensor, opacity_gs, visibility_filter, trans_gs_asset):
        if joint_gs_step <= self.op.start_stat or joint_gs_step >= self.op.densify_until_iter:
            return None

        self.gaussians.training_statis(viewspace_point_tensor, opacity_gs, visibility_filter)

        if joint_gs_step <= self.op.update_from or joint_gs_step % self.op.update_interval != 0:
            return None

        xyz_sdf = None
        inside_box = None
        if self.geometry_awared_control:
            xyz_sdf, inside_box = self._build_joint_gs_xyz_sdf(batch_time, trans_gs_asset)

        densify_stats = self.gaussians.adjust_xyz(
            check_interval=self.op.update_interval,
            extent=self.scene.cameras_extent,
            success_threshold=self.op.success_threshold,
            grad_threshold=self.op.densify_grad_threshold,
            min_opacity=self.op.min_opacity,
            xyz_sdf=xyz_sdf,
            inside_box=inside_box,
            xyz_inside_box=inside_box,
            growing_weight=self.config.system.growing_weight,
        )
        self.log('train/gs_densify_grown', float(densify_stats['grown']), prog_bar=False)
        self.log('train/gs_densify_pruned', float(densify_stats['pruned']), prog_bar=False)
        self.log('train/gs_densify_reset', float(densify_stats['reset']), prog_bar=False)
        return densify_stats

    def configure_callbacks(self):
        callbacks = super().configure_callbacks() or []
        callbacks.append(NaNGuardCallback())
        return callbacks

    # Training step for both Scaffold-GS and Instant-nsr
    def training_step(self, batch, batch_idx):
        #inite loss of gs
        debug = False
        loss_gaussian = 0
        if self.global_epoch < self.config.model.geometry.xyz_encoding_config.start_step:
            self.geometry_awared_control = False
            normal_w = 0
            depth_w = 0
        else:
            self.geometry_awared_control = True
            
            normal_w = self.config.system.loss.normal_w
            depth_w = self.config.system.loss.depth_w
        
                    
        if self.config.model.if_gaussian:
            viewpoint_cam = self.scene.getOriginalTrainCameras()[batch['used_index']]
            # Get the same image index as Instant-nsr
            yy = batch['used_y']
            xx = batch['used_x']
            
            # Collect 3D metrics for wandb logging
            current_time = batch['time'].item() if 'time' in batch else 0.0
    
            
            if self.config.model.optimize_gaussian and self.global_epoch > self.config.model.start_optimize_gaussian:
                iter_start = torch.cuda.Event(enable_timing = True)
                iter_end = torch.cuda.Event(enable_timing = True)
    
                self.gaussians.update_learning_rate(self.global_epoch, self.op.start_dynamic_iter)
                iter_start.record()
                render_pkg = RENDER_FUNC(
                    viewpoint_cam, 
                    self.gaussians, 
                    self.piplin, 
                    self.background, 
                    self.scene,
                    transform_scaler=self.config.model.transform_scaler,
                    return_normal=True, 
                    return_depth=True,
                    return_asset=True,
                    dynamic_decode=True,
                )
                image, viewspace_point_tensor, visibility_filter, opacity_gs, gs_depth_hand, render_gs_depth, render_gs_normal, gs_mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["opacity"], render_pkg["real_depth"], render_pkg['render_depth'], render_pkg["render_depth_normal"], render_pkg['mask']
                
                gs_vox_pcd = self.gs_voxel[batch['used_index']].cuda()
                gs_delta = render_pkg['delta_anchor']

            else:
                image = self.gs_image_list[batch['used_index']].cuda()
                gs_depth_hand = self.gs_depth_list[batch['used_index']].cuda()
                render_gs_depth = self.gs_render_depth_list[batch['used_index']].cuda()
                render_gs_normal = self.gs_normal_list[batch['used_index']].cuda()
                gs_delta = self.gs_delta_list[batch['used_index']].cuda()
                gs_vox_pcd = self.gs_voxel[batch['used_index']].cuda()
                # gs_vox_inds = self.gs_vox_inds[batch['used_index']]
                
            gs_depth = gs_depth_hand.mean(dim=0, keepdim=True).permute(1, 2, 0)
            gs_normal = render_gs_normal.permute(1, 2, 0)
            
            # Get the same pixel indexes as Instant-nsr.
            picked_gs_depth = gs_depth[yy,xx]
            picked_gs_normal = gs_normal[yy,xx]
    
            
        trans_gs_asset = {
            'xyz': self.gaussians.get_xyz.detach(),
            'feature': SH2RGB(self.gaussians.get_color.detach()),
            'delta': gs_delta.detach(),
            'GS': self.gaussians,
            'vox': gs_vox_pcd,
            'val': False,
            # 'gs_vox': gs_pcd,
        }
        
        if self.global_epoch > self.config.model.geometry.xyz_encoding_config.start_step:
            out = self(batch, batch['time'], self.gaussians, trans_gs_asset=trans_gs_asset, gs_depth=picked_gs_depth.detach(), use_depth_guide=True, debug=debug, iteration=self.global_epoch)            
        else:
            out = self(batch,  batch['time'], self.gaussians, trans_gs_asset=trans_gs_asset, gs_depth=picked_gs_depth.detach(), use_depth_guide=False, debug=False, stable_train=False, concat_ray=True, iteration=self.global_epoch)  
    
        if out['zero_samples']==True:
            return None       
        
        loss = 0.
        if out['rays_valid'].sum() > 0:
            loss_rgb_l1 = F.l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
            self.log('train/loss_rgb', loss_rgb_l1 *  self.C(self.config.system.loss.lambda_rgb_l1), prog_bar=True)
            
            loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)   
      
        # Mask l1 loss
        if self.dataset.apply_mask:
            opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
            loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
            self.log('train/loss_mask', loss_mask * self.C(self.config.system.loss.lambda_mask), prog_bar=True)
            loss += loss_mask * self.C(self.config.system.loss.lambda_mask)
    
        # predicted normal and depth of Scaffold-GS, taken as GT of the Instant-NSR side
        if self.config.model.if_gaussian:
            fixed_picked_gs_normal = picked_gs_normal[out['rays_valid'][...,0]].detach()
            fixed_picked_gs_depth = picked_gs_depth[out['rays_valid'][...,0]].detach()
            mask = batch['fg_mask'][out['rays_valid'][...,0]].detach()
            
            diff_neus = torch.abs(out['depth'][out['rays_valid'][...,0]] * mask - fixed_picked_gs_depth * mask)
    
            depth_ratio = 10.0 if self.global_epoch > self.config.model.geometry.xyz_encoding_config.start_step else 2.0
    
            diff_neus[diff_neus > self.config.model.radius/depth_ratio] = 0
            diff_neus_count = (diff_neus>0).sum()
            loss_depth_L1 = diff_neus.sum() / (diff_neus_count+1e-8)
            # normalzied the depth loss by the frontground size.
            loss += loss_depth_L1 * self.C(depth_w)/self.config.model.radius
            self.log('train/loss_depth_L1_neus', float(loss_depth_L1/self.config.model.radius))
            if 'comp_normal' in out.keys():
                normal_diff = self.cos_similarity_loss(fixed_picked_gs_normal * mask[:, None], (out['comp_normal'][out['rays_valid'][...,0]] * mask[:, None]))
                loss +=  normal_diff * normal_w
        
                self.log('train/normal_loss_neus', normal_diff)
    
        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
            
    
        # Cycle Loss
        loss += l1_loss(
            -trans_gs_asset['delta'], 
            self.model.forward_deform(trans_gs_asset['xyz'] + trans_gs_asset['delta'], torch.tensor(batch['time']).cuda(), trans_gs_asset=trans_gs_asset)['delta']
        ) * self.config.system.loss.lambda_cycle_loss
        
        # Canonical SDF-to-GS supervision: binary occupancy agreement between
        # NeuS |sdf| and the KNN distance to the canonical GS centers (_xyz).
        canon_sdf_weight = self.C(
            self.config.system.loss.lambda_canon_sdf
        )
        num_samples = out['ori_positions'].shape[0] # min(8192, out['ori_positions'].shape[0])
        sample_idx = torch.randperm(out['ori_positions'].shape[0], device=self.gaussians._xyz.device)[:num_samples]
        dist_thresh = self.config.system.loss.get('sdf_distance_threshold', 1e-4)
        # Query the GS canonical params (same space as NeuS 'ori_positions'),
        # not the canon_time render snapshot (self.canonical_gs).
        canon_gs_params = {
            'xyz': self.gaussians._xyz,
            'feature': self.gaussians.get_features,
            'opacity': self.gaussians._opacity,
        }
        gs_query = self.gaussians.query_point_attributes(
            out['ori_positions'][sample_idx],
            tgt_gs=canon_gs_params,
            K=12,
            detach=True,
        )
        sdf_loss = F.binary_cross_entropy(
            (out['sdf_samples'][sample_idx].abs() > dist_thresh).float(),
            (gs_query['dist'] > dist_thresh).float(),
        )
        
        if canon_sdf_weight > 0:
            loss += sdf_loss * canon_sdf_weight
            self.log('train/loss_canon_sdf', sdf_loss, prog_bar=False)
        # if canon_sdf_weight > 0:
        #     loss_cfg = self.config.system.loss
        #     num_ray_samples = min(
        #         int(loss_cfg.get('canon_sdf_num_samples', 32768)),
        #         out['canon_positions'].shape[0],
        #     )
        #     ray_sample_idx = torch.randperm(
        #         out['canon_positions'].shape[0],
        #         device=self.gaussians._xyz.device,
        #     )[:num_ray_samples]

        #     canon_query = out['canon_positions'][ray_sample_idx]
        #     with torch.no_grad():
        #         gs_query = self.gaussians.query_point_attributes(
        #             canon_query.detach(),
        #             tgt_gs=self.canonical_gs,
        #             K=int(loss_cfg.get('canon_sdf_knn', 12)),
        #             detach=True,
        #         )
        #         target_udf = torch.sqrt(
        #             gs_query['nearest_dist_sq'].clamp_min(0.0) + 1e-12
        #         )

        #     truncation = float(
        #         loss_cfg.get(
        #             'canon_sdf_truncation',
        #             0.05 * float(self.config.model.radius),
        #         )
        #     )
        #     huber_beta = float(
        #         loss_cfg.get('canon_sdf_huber_beta', 0.1 * truncation)
        #     )
        #     if truncation <= 0 or huber_beta <= 0:
        #         raise ValueError(
        #             "canon_sdf_truncation and canon_sdf_huber_beta "
        #             "must be positive"
        #         )

        #     # Regress a truncated unsigned distance on time-to-canonical
        #     # ray samples.
        #     predicted_udf = out['sdf_samples'][ray_sample_idx].abs()
        #     loss_canon_distance = F.smooth_l1_loss(
        #         predicted_udf.clamp(max=truncation),
        #         target_udf.clamp(max=truncation),
        #         beta=huber_beta,
        #     )

        #     # Anchor the SDF zero level set to opaque canonical GS centers.
        #     canonical_xyz = self.canonical_gs['xyz'].detach()
        #     canonical_opacity = self.gaussians.opacity_activation(
        #         self.canonical_gs['opacity'].detach()
        #     ).reshape(-1)
        #     min_opacity = float(
        #         loss_cfg.get('canon_sdf_min_opacity', 0.5)
        #     )
        #     surface_candidates = torch.nonzero(
        #         canonical_opacity >= min_opacity,
        #         as_tuple=False,
        #     ).reshape(-1)
        #     if surface_candidates.numel() == 0:
        #         surface_candidates = torch.arange(
        #             canonical_xyz.shape[0],
        #             device=canonical_xyz.device,
        #         )
        #     num_surface_samples = min(
        #         int(loss_cfg.get('canon_sdf_num_surface_samples', 8192)),
        #         surface_candidates.numel(),
        #     )
        #     surface_perm = torch.randperm(
        #         surface_candidates.numel(),
        #         device=surface_candidates.device,
        #     )[:num_surface_samples]
        #     surface_xyz = canonical_xyz[
        #         surface_candidates[surface_perm]
        #     ]
        #     surface_sdf = self.model.geometry(
        #         surface_xyz,
        #         with_grad=False,
        #         with_feature=False,
        #     )
        #     loss_canon_surface = F.smooth_l1_loss(
        #         surface_sdf,
        #         torch.zeros_like(surface_sdf),
        #         beta=huber_beta,
        #     )

        #     distance_weight = float(
        #         loss_cfg.get('canon_sdf_distance_weight', 1.0)
        #     )
        #     surface_weight = float(
        #         loss_cfg.get('canon_sdf_surface_weight', 1.0)
        #     )
        #     sdf_loss = (
        #         distance_weight * loss_canon_distance
        #         + surface_weight * loss_canon_surface
        #     )
        #     loss += sdf_loss * canon_sdf_weight
        #     self.log('train/loss_canon_sdf', sdf_loss)
        #     self.log(
        #         'train/loss_canon_sdf_distance',
        #         loss_canon_distance,
        #     )
        #     self.log(
        #         'train/loss_canon_sdf_surface',
        #         loss_canon_surface,
        #     )
        
        # Eikonal loss
        if self.config.system.loss.lambda_eikonal > 0 and 'sdf_grad_samples' in out.keys():
            loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
            self.log('train/loss_eikonal', loss_eikonal)
            loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        if 'delta_canon_x' in out.keys():
            self.log('train/loss_delta', out['delta_canon_x'].max(), prog_bar=True)
            loss_delta = torch.tensor(0).float().cuda()
            loss_delta += (out['delta_canon_x'].abs().mean()) #### .abs() * 0.1
    
            if 't_prev' in batch.keys():
                rand_time_prev = batch['time'] + (batch['t_prev'] - batch['time']) * torch.rand(1)[0]
    
                delta_x_prev = out['delta_canon_x'] - self.model.forward_deform(out['ori_positions'], rand_time_prev, trans_gs_asset=trans_gs_asset)['delta'].detach()
                
                loss_delta += delta_x_prev.pow(2).mean() * 0.1
                
            if 't_next' in batch.keys():
                rand_time_next = batch['time'] + (batch['t_next'] - batch['time']) * torch.rand(1)[0]
    
                delta_x_next = out['delta_canon_x'] - self.model.forward_deform(out['ori_positions'], rand_time_next, trans_gs_asset=trans_gs_asset)['delta'].detach()
                loss_delta += delta_x_next.pow(2).mean() * 0.1
    
            loss += loss_delta
    
        # Curvature loss, Note that the curvature loss weight is adaptived to the training iteration.
        if  'smoothing' in out.keys() and self.C(self.config.system.loss.lambda_smoothing)>0:
            loss_smoothing = out['smoothing'].abs().mean()
            self.log('train/loss_smoothing', loss_smoothing)
            loss += loss_smoothing * self.C(self.config.system.loss.lambda_smoothing)           
    
        # Loss reguralization  
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
    
        self.log('train/inv_s', out['inv_s'], prog_bar=True)
        
        
        if self.config.model.if_gaussian:
            gt_image = viewpoint_cam.original_image.cuda()
                    
            if self.global_epoch % 100 == 0:
                save_image = torch.cat([gt_image, render_gs_depth, render_gs_normal, image], dim=2)
                cv2.imwrite(self.get_save_path('train_preview.png'), cv2.resize((save_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8), (save_image.shape[2] // 4,save_image.shape[1] // 4))[..., ::-1])
    
        #     if self.global_epoch > self.config.model.start_optimize_gaussian and self.config.model.optimize_gaussian:
        #         if self.global_epoch <= self.op.update_until:
        #             # RGB L1 loss
        #             rgb_loss = (1.0 - self.op.lambda_dssim) * l1_loss(image, gt_image) + self.op.lambda_dssim * (1.0 - ssim(image, gt_image))  
        #             delta_reg_loss = self.gaussians.delta_loss.abs().mean()
                    
        #             if self.set_canon_pose :
        #                 t = torch.tensor([viewpoint_cam.t]).float().cuda()
        #                 delta_reg_loss = torch.exp(-(t - self.canon_t)**2 / (2 * 1**2))
        #             delta_reg_loss += self.gaussians.delta_xyz.abs().mean()
    
        #             mask_loss = 0
        #             if self.config.dataset.apply_mask and gs_mask is not None:
        #                 gt_mask = viewpoint_cam.original_mask.cuda()
        #                 mask_loss = l1_loss(gs_mask, gt_mask)
    
        #             # Predicted depth and normal of Instant-NSR, taken as GT of the GS side.
        #             fixed_neus_picked_depth = out['depth'][out['rays_valid'][...,0]].detach()
        #             fixed_neus_picked_normal = out['comp_normal'][out['rays_valid'][...,0]].detach()
                    
        #             normal_loss_gs = 0
        #             if normal_w > 0:
        #                 normal_loss_gs = self.cos_similarity_loss(picked_gs_normal[out['rays_valid'][...,0]],fixed_neus_picked_normal) * normal_w
    
        #                 self.log('train/GS_normal_loss_gs', normal_loss_gs)
                    
        #             depth_loss_gs = 0
        #             if depth_w > 0:
        #                 diff = torch.abs(fixed_neus_picked_depth - picked_gs_depth[out['rays_valid'][...,0]])
        #                 depth_ratio = 10.0
        #                 diff[diff > self.config.model.radius/depth_ratio] = 0
                        
        #                 diff_count = (diff>0.0).sum()
    
        #                 loss_depth_L1_gs = diff.sum() / (diff_count+1e-8)
        #                 self.log('train/GS_loss_depth', loss_depth_L1_gs)
    
        #                 # normalzied the depth loss by the frontground size.
        #                 depth_loss_gs = loss_depth_L1_gs * self.C(self.config.system.loss.depth_w)/self.config.model.radius
                    
        #             loss_cycle = torch.tensor(0.).float().cuda()
        #             if self.config.system.loss.lambda_cycle_loss > 0:
        #                     loss_cycle += l1_loss(
        #                         render_pkg['delta_anchor'],
        #                         -self.model.forward_deform(
        #                             trans_gs_asset['xyz'] + trans_gs_asset['delta'], 
        #                             torch.tensor(batch['time']).cuda(), trans_gs_asset=trans_gs_asset
        #                         )['delta'].detach()
        #                     ) * self.config.system.loss.lambda_cycle_loss
    
        #             loss_gaussian = rgb_loss * self.op.lambda_rgb + delta_reg_loss * self.op.lambda_delta_reg + mask_loss * self.op.lambda_mask + depth_loss_gs + normal_loss_gs + loss_cycle
    
        #             self.log('train/loss_gaussian', float(loss_gaussian), prog_bar=True)
    
        #             loss_gaussian.backward(retain_graph=False)
        #             iter_end.record()
                    
                    # # GS densification
                    # with torch.no_grad():
                    #     self.ema_loss_for_log = 0.4 * loss_gaussian.item() + 0.6 * self.ema_loss_for_log
                    #     if (self.global_epoch in self.saving_iterations):
                    #         self.loggger.info("\n[ITER {}] Saving Gaussians".format(self.global_epoch))
                    #         self.scene.save(self.global_epoch, render_pkg['gs_asset'])
                        
                    #     if (self.global_epoch in self.testing_iterations):
                    #         training_report(
                    #             self.tb_writer, 
                    #             self.config.dataset.root_dir.split('/')[-1], 
                    #             self.global_epoch, 
                    #             rgb_loss, 
                    #             loss_gaussian, 
                    #             l1_loss, 
                    #             iter_start.elapsed_time(iter_end), 
                    #             self.args.test_iterations, 
                    #             self.scene, 
                    #             RENDER_FUNC, 
                    #             (self.piplin, self.background, self.scene, self.config.model.transform_scaler),
                    #             self.wandb, 
                    #             self.loggger,
                    #             dynamic_decode=True
                    #         )
                        
                    #     if self.global_epoch + self.global_step < self.op.update_until:
                    #         self.gaussians.training_statis(viewspace_point_tensor, opacity_gs, visibility_filter)
                        
                    #         # self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    #         if (self.global_epoch + self.global_step) % 100 == 0: # opt.update_intern_interval == 0:
                    #         # if current_epoch_gs < 0: # opt.update_intern_interval == 0:
                    #             if not self.geometry_awared_control:
                    #                 # Original density control
                    #                 self.gaussians.adjust_xyz(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity, growing_weight=self.config.system.growing_weight)
                                    
                    #             else:
                    #                 #  Density control guided by predicted sdf
                    #                 # guide density control after warmup of Instant-nsr
                                                                       
                    #                 # gs_positions = self.gaussians.get_transformed_xyz
                    #                 gs_positions = self.gaussians.get_xyz
     
    
                    #                 min_point = torch.tensor([-self.config.model.radius, -self.config.model.radius, -self.config.model.radius],device=gs_positions.device) 
                    #                 max_point = torch.tensor([self.config.model.radius, self.config.model.radius, self.config.model.radius],device=gs_positions.device) 
                    #                 inside_box = (gs_positions > min_point) & (gs_positions < max_point)
                    #                 inside_box = inside_box.all(dim=1)
    
                    #                 # inside_positions = gs_positions[inside_box]
                    #                 # with torch.no_grad():
                    #                 #     delta_p = self.model.forward_deform(inside_positions, batch['time'])['delta']
                    #                 #     inside_positions = inside_positions + delta_p
                    #                 # set the sdf of 3D gaussians in the background to 100000.
                    #                 # xyz_sdf = torch.ones(gs_positions.shape[0]).to(gs_positions.device)*100000
                                    
                    #                 # inside_xyz_sdf = self.model.geometry(inside_positions, t_emb=torch.tensor(batch['time']).cuda(), with_grad=False, with_feature=False)
    
                    #                 # xyz_sdf[inside_box] = inside_xyz_sdf
                    #                 # calculate the sdf of anchor points in the frontground
                    #                 xyz_positions = self.gaussians.get_xyz
                    #                 xyz_inside_box = (xyz_positions > min_point) & (xyz_positions < max_point)
                    #                 xyz_inside_box = xyz_inside_box.all(dim=1)
    
                    #                 # with torch.no_grad():
                    #                 #     delta_p = self.model.forward_deform(xyz_inside_box, batch['time'])
                    #                 #     xyz_inside_box = xyz_inside_box + delta_p * self.config.model.transform_scaler
                    #                 with torch.no_grad():
                    #                     xyz_sdf = self.model.geometry(xyz_positions, with_grad=False, with_feature=False)
    
                    #                 self.gaussians.adjust_xyz(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity, xyz_sdf=xyz_sdf, inside_box=inside_box, xyz_inside_box=xyz_inside_box, growing_weight=self.config.system.growing_weight)
    
                    #         elif self.global_epoch == self.op.update_until:
                    #             del self.gaussians.opacity_accum
                    #             del self.gaussians.offset_gradient_accum
                    #             del self.gaussians.offset_denom
                    #             torch.cuda.empty_cache()
    
                    #     self.gaussians.optimizer.step()
                    #     self.gaussians.optimizer.zero_grad(set_to_none = True)
    
                        # if (self.global_epoch in self.args.checkpoint_iterations):
                        #     self.loggger.info("\n[ITER {}] Saving Checkpoint".format(self.global_epoch))
                        #     torch.save((self.gaussians.capture(), self.global_epoch), self.scene.model_path + "/chkpnt" + str(self.global_epoch) + ".pth")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ NaN/Inf in loss", loss)
            torch.set_printoptions(threshold=10_000)
            try:
                print("out stats", out.min().item(), out.max().item(), out.mean().item())
            except Exception:
                pass
            # Use batch indices safely (used_y/used_x recorded in preprocess_data)
            if isinstance(batch, dict) and 'used_y' in batch:
                uy = batch['used_y']
                try:
                    print("used_y stats", uy.min().item(), uy.max().item(), uy.unique()[:20])
                except Exception as e:
                    print("used_y stats unavailable:", e)
        
        return {'loss': loss}
   

    def validation_step(self, batch, batch_idx):
        # Lazily restore mesh_deform_net + canonical mesh from disk
        # (saved during mesh_gs_training). The canonical load lets us skip NeuS
        # re-extraction, which can yield a worse mesh than the one trained on.
        if self.mesh is None:
            self._load_canon_mesh()
        if not hasattr(self, 'mesh_deform_net') or self.mesh_deform_net is None:
            self._load_mesh_deform_net()
        global_step = self.global_epoch
        if global_step % 10:
            global_step -= global_step % 10
        print('Global step: ', global_step)
        
        if 'index' in batch:
            batch_idx = batch['index']
        
        W, H = self.dataset.img_wh
        
        gs_psnr = torch.zeros(1).float()
        gs_lpips = torch.zeros(1).float()
        gs_ssim = torch.zeros(1).float()
        if self.config.model.if_gaussian:
            viewpoint_cam = self.scene.getTestCameras()[batch_idx] 
            render_pkg = RENDER_FUNC(
                    viewpoint_cam,
                    self.gaussians, 
                    self.piplin, 
                    self.background, 
                    self.scene,
                    transform_scaler=self.config.model.transform_scaler,
                    return_scaffold=False,
                    return_asset=True,
                    return_depth=True, 
                    return_normal=True, 
                    dynamic_decode=True
                )
            gs_psnr = self.criterions['psnr'](render_pkg['render'].permute(1, 2, 0).to(batch['rgb']), batch['rgb'].view(H, W, 3))
            render_pkg['render_depth_normal'] = render_pkg['render_depth_normal'] * render_pkg['mask'] + (1 - render_pkg['mask'])
            _, voxel_gs_pcd, _, _ = self.model.tineuvox.voxelize(self.gaussians.get_transformed_xyz, SH2RGB(self.gaussians.get_color))
            
            # Calculate LPIPS for GS
            gs_render = render_pkg['render'].permute(1, 2, 0).view(H, W, 3).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            gt_image = batch['rgb'].view(H, W, 3).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            gs_lpips = lpips(gs_render, gt_image, net_type='vgg').mean()
            gs_ssim = ssim(gs_render, gt_image).mean()
            
            self.save_image_grid(f"it_gs{global_step}-{batch['index'][0].item()}.png", [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': render_pkg['render'].permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': render_pkg['render_depth'][0].cpu().numpy(), 'kwargs': {}},
                # {'type': 'rgb', 'img': render_pkg['render_gs_normal'].permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': render_pkg['render_depth_normal'].permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC'}}
            ])        
            self.scene.save(global_step, time=0.0, only_gs=True)
            self.scene.save(global_step, time=0.5, only_gs=True)
            self.scene.save(global_step, time=1.0, only_gs=True)
            
            delta_anchor = render_pkg['delta_anchor']
            trans_gs_asset = {
                'xyz': self.gaussians.get_xyz,
                'feature': SH2RGB(self.gaussians.get_color),
                'delta': delta_anchor,
                'GS': self.gaussians,
                'vox': voxel_gs_pcd,
                'val': False
            }
        
        out = self(
            batch, 
            batch['time'], 
            self.gaussians, 
            trans_gs_asset=trans_gs_asset, 
            use_depth_guide=False, 
            gs_depth=render_pkg['real_depth'].reshape(1, -1).permute(1, 0), 
            debug=True, 
            iteration=self.global_epoch
        )
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])

        # Calculate LPIPS/SSIM for NeUS
        neus_render = out['comp_rgb'].view(H, W, 3).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        gt_image = batch['rgb'].view(H, W, 3).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        neus_lpips = lpips(neus_render, gt_image, net_type=LPIPS_MODEL).mean()
        neus_ssim = ssim(neus_render, gt_image).mean()

        out['comp_normal'] = out['comp_normal'] * out['opacity'] if 'comp_normal' in out.keys() else torch.zeros_like(out['comp_rgb'])
        out['comp_normal'] = out['comp_normal'] + (1 - out['opacity'])
        
        self.save_image_grid(f"it{global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': (out['comp_normal']).view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            # {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            # {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': out['opacity'].view(H, W)[...,None].repeat(1, 1, 3), 'kwargs': {'data_format': 'HWC'}}
        ])
            
        if self.args.mesh_render:
            if not hasattr(self, '_dr_ctx'):
                self._dr_ctx = self.dr.RasterizeCudaContext()
            if self.mesh is None:
                self.export()
            v_can = self.mesh['v_pos'].float().cuda()
            v_rgb = self.mesh.get('v_rgb', None)
            if v_rgb is not None:
                if not torch.is_tensor(v_rgb):
                    v_rgb = torch.tensor(v_rgb)
                v_rgb = v_rgb.float().cuda()
            # 주의: self.mesh 딕셔너리에서 face 인덱스를 가져오는 키를 정확히 입력하십시오.
            faces = self.mesh.get('t_pos_idx') 
            faces = torch.tensor(faces).cuda()

            
            delta_xyz = self._deform_mesh_verts(v_can, viewpoint_cam.t)
            v_def = v_can + delta_xyz
            render, depth, normal, mask, mesh = render_mesh_with_nvdiffrast(
                self.dr,
                self._dr_ctx,
                viewpoint_cam,
                v_def,
                faces,
                self.background,
                v_rgb=v_rgb,
            )

            mesh_img = self.save_image_grid(f"mesh{global_step}-{batch['index'][0].item()}.png", [{'type': 'rgb', 'img': render.permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC'}},{'type': 'grayscale', 'img': depth[0].cpu().numpy(), 'kwargs': {}},{'type': 'rgb', 'img': normal.permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}},{'type': 'rgb', 'img': mesh.permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}},])

        return {
            'psnr': psnr,
            'gs_psnr': gs_psnr,
            'lpips': neus_lpips,
            'gs_lpips': gs_lpips,
            'ssim': neus_ssim,
            'gs_ssim': gs_ssim,
            'index': batch_idx,
        }
          
    
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {
                        'psnr': step_out['psnr'], 
                        'gs_psnr': step_out['gs_psnr'],
                        'lpips': step_out['lpips'],
                        'gs_lpips': step_out['gs_lpips'],
                        'ssim': step_out['ssim'],
                        'gs_ssim': step_out['gs_ssim']
                    }
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {
                            'psnr': step_out['psnr'][oi], 
                            'gs_psnr': step_out['gs_psnr'][oi],
                            'lpips': step_out['lpips'][oi],
                            'gs_lpips': step_out['gs_lpips'][oi],
                            'ssim': step_out['ssim'][oi],
                            'gs_ssim': step_out['gs_ssim'][oi]
                        }
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            gs_psnr = torch.mean(torch.stack([o['gs_psnr'] for o in out_set.values()]))
            lpips = torch.mean(torch.stack([o['lpips'] for o in out_set.values()]))
            gs_lpips = torch.mean(torch.stack([o['gs_lpips'] for o in out_set.values()]))
            ssim_val = torch.mean(torch.stack([o['ssim'] for o in out_set.values()]))
            gs_ssim_val = torch.mean(torch.stack([o['gs_ssim'] for o in out_set.values()]))
            
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)
            self.log('val/psnr_gs', gs_psnr, prog_bar=True, rank_zero_only=True)
            self.log('val/lpips', lpips, prog_bar=True, rank_zero_only=True)
            self.log('val/lpips_gs', gs_lpips, prog_bar=True, rank_zero_only=True)
            self.log('val/ssim', ssim_val, prog_bar=False, rank_zero_only=True)
            self.log('val/ssim_gs', gs_ssim_val, prog_bar=False, rank_zero_only=True)

            if self.config.system.export_mesh:
                self.export()

                   

    
    
    def predict_step(self, batch, batch_idx):
        # Lazily restore mesh_deform_net + canonical mesh from disk
        # (saved during mesh_gs_training).
        if self.mesh is None:
            self._load_canon_mesh()
        if not hasattr(self, 'mesh_deform_net') or self.mesh_deform_net is None:
            self._load_mesh_deform_net()
        if 'ode_field' in self.config.system.optimizer.params:
            self.gaussians.import_ode_field(self.model.ode_field)
        global_step = self.global_epoch
        self.saving_iteration = np.arange(self.config.dataset.predict.num_frames + 1)[::5]
        
        print('Global step: ', global_step)
        if 'index' in batch:
            batch_idx = batch['index']
        print("Time: ", batch['time'].item())
        return_dict = {}
        if self.config.model.if_gaussian:
            viewpoint_cam = self.scene.getPredictCameras()[batch_idx] 
            render_pkg = RENDER_FUNC(
                viewpoint_cam,
                self.gaussians, 
                self.piplin, 
                self.background, 
                self.scene,
                transform_scaler=self.config.model.transform_scaler,
                return_asset=True,
                return_depth=True, 
                return_normal=True, 
                dynamic_decode=True,
                return_scaffold=False,
            )
            
            render_pkg['render_depth_normal'] = render_pkg['render_depth_normal'] * render_pkg['mask'] + (1 - render_pkg['mask'])
            _, voxel_gs_pcd, _, _ = self.model.tineuvox.voxelize(self.gaussians.get_transformed_xyz, SH2RGB(self.gaussians.get_color))

            gs_img = self.save_image_grid(f"gs{global_step}-{batch['index'][0].item()}.png", [{'type': 'rgb', 'img': render_pkg['render'].permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC'}},{'type': 'grayscale', 'img': render_pkg['render_depth'][0].cpu().numpy(), 'kwargs': {}},{'type': 'rgb', 'img': render_pkg['render_depth_normal'].permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC'}}])        
            
            if batch_idx.item() in self.saving_iteration:
                self.scene.save(global_step, time=batch['time'].item(), only_gs=True)

            return_dict['gs_image'] = gs_img
            delta_anchor = render_pkg['delta_anchor']

            trans_gs_asset = {
                'xyz': self.gaussians.get_xyz,
                'feature': SH2RGB(self.gaussians.get_color),
                'delta': delta_anchor,
                'GS': self.gaussians,
                'vox': None, # voxel_gs_pcd,
                'val': False
            }

        if self.args.mesh_render:
            if not hasattr(self, '_dr_ctx'):
                self._dr_ctx = self.dr.RasterizeCudaContext()
            if self.mesh is None:
                self.export()
            v_can = self.mesh['v_pos'].float().cuda()
            v_rgb = self.mesh.get('v_rgb', None)
            if v_rgb is not None:
                if not torch.is_tensor(v_rgb):
                    v_rgb = torch.tensor(v_rgb)
                v_rgb = v_rgb.float().cuda()
            # 주의: self.mesh 딕셔너리에서 face 인덱스를 가져오는 키를 정확히 입력하십시오.
            faces = self.mesh.get('t_pos_idx') 
            faces = torch.tensor(faces).cuda()

            
            delta_xyz = self._deform_mesh_verts(v_can, viewpoint_cam.t)
            v_def = v_can + delta_xyz
            render, depth, normal, mask, mesh = render_mesh_with_nvdiffrast(
                self.dr,
                self._dr_ctx,
                viewpoint_cam,
                v_def,
                faces,
                self.background,
                v_rgb=v_rgb,
            )

            mesh_img = self.save_image_grid(f"mesh{global_step}-{batch['index'][0].item()}.png", [{'type': 'rgb', 'img': render.permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC'}},{'type': 'grayscale', 'img': depth[0].cpu().numpy(), 'kwargs': {}},{'type': 'rgb', 'img': normal.permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}},{'type': 'rgb', 'img': mesh.permute(1, 2, 0).cpu().numpy(), 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}},])

            return_dict['mesh_render'] = mesh_img
    
        return return_dict
        



    # @torch.no_grad()
    # def export(self, iter=0, save=True):
    #     trans_gs_asset = {
    #         'xyz': self.gaussians.get_xyz.contiguous().cuda(),
    #         'feature': SH2RGB(self.gaussians.get_color.contiguous().cuda()),
    #         'delta': torch.zeros_like(self.gaussians.get_xyz.contiguous().cuda()),
    #         'GS': self.gaussians,
    #         'vox': None,
    #         'canon_time': self.canon_t[None,].cuda(),
    #     }
    #     self.model.cuda()
    #     self.mesh = self.model.export(self.config.export, trans_gs_asset=trans_gs_asset)
    #     # if self.config.model.if_gaussian:
    #     #     tc = torch.tensor(self.scene.center).reshape(3)
    #     #     pts = mesh['v_pos']
    #     #     pts = pts * self.scene.scale
    #     #     pts += tc
    #     #     mesh['v_pos'] = pts

    #     if save:
    #         global_step = self.custom_global_step if self.global_epoch == -1 and self.custom_global_step else self.global_epoch + iter
    #         global_step += 1
    #         self.save_mesh(
    #             f"it{global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_canon.ply",
    #             **self.mesh
    #         )
    #         num_frames = 10
    #         t_values = torch.linspace(0.0, 1.0, steps=num_frames)

    #         for frame_idx, t_val in enumerate(t_values):
    #             transformed_mesh = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in self.mesh.items()}
    #             t = torch.tensor([t_val]).float().cuda()
    #             _t = t.expand(self.mesh['v_pos'].shape[0]).unsqueeze(-1)

    #             transform_out = chunk_batch(
    #                 self.gaussians.trans_vox, 
    #                 self.config.export.chunk_size // 2, 
    #                 True, 
    #                 self.mesh['v_pos'].cuda(), _t
    #             )
    #             transformed_mesh['v_pos'] = transformed_mesh['v_pos'] + transform_out[:, :3].detach().cpu()

    #             # 블렌더 인식을 위한 순차적 파일명 지정 (0000.ply, 0001.ply ...)
    #             # method와 resolution 변수명은 config 구조에 맞게 유지
    #             filename = f"it{global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_frame_{frame_idx:04d}.ply"
                
    #             self.save_mesh(filename, **transformed_mesh)        

    #         print(f"Exported {num_frames} meshes as sequence at iteration", global_step)

    def _deform_mesh_verts(self, v_can: torch.Tensor, t_scalar: float) -> torch.Tensor:
        """Return the deformation of canonical mesh vertices at a given time."""
        if hasattr(self, 'mesh_deform_net') and self.mesh_deform_net is not None:
            t_val = torch.tensor([[t_scalar]], dtype=torch.float32, device=v_can.device)
            with torch.no_grad():
                return self.mesh_deform_net(v_can, t_val)[:, :3]
        else:
            t_vec = torch.tensor([t_scalar], dtype=torch.float32, device=v_can.device)
            delta = self.gaussians.trans_vox(v_can, t_vec)    # [V, D]
            return delta[:, :3]

    @torch.no_grad()
    def export(self, iter=0, save=True, re_extract=True):
        # re_extract=False: skip NeuS re-extraction and use the existing self.mesh as-is.
        if re_extract or self.mesh is None:
            trans_gs_asset = {
                'xyz': self.gaussians.get_xyz.contiguous().cuda(),
                'feature': SH2RGB(self.gaussians.get_color.contiguous().cuda()),
                'delta': torch.zeros_like(self.gaussians.get_xyz.contiguous().cuda()),
                'GS': self.gaussians,
                'vox': None,
                'canon_time': self.canon_t[None,].cuda(),
            }
            self.model.cuda()
            self.mesh = self.model.export(self.config.export, trans_gs_asset=trans_gs_asset)
        # if self.config.model.if_gaussian:
        #     tc = torch.tensor(self.scene.center).reshape(3)
        #     pts = mesh['v_pos']
        #     pts = pts * self.scene.scale
        #     pts += tc
        #     mesh['v_pos'] = pts

        if save:
            global_step = self.global_epoch + iter + 1
            self.save_mesh(
                f"it{global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_canon.ply",
                **self.mesh
            )
            for t in [0.0, 0.5, 1.0]:
                transformed_mesh = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in self.mesh.items()}
                t = torch.tensor([t]).float().cuda()
                _t = t.expand(self.mesh['v_pos'].shape[0]).unsqueeze(-1)

                v_can_exp = self.mesh['v_pos'].cuda()
                delta_xyz = self._deform_mesh_verts(v_can_exp, t.item())
                transformed_mesh['v_pos'] = transformed_mesh['v_pos'] + delta_xyz.detach().cpu()

                self.save_mesh(
                    f"it{global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_t{t.item()}.ply",
                    **transformed_mesh
                )        
                print("Exported meshes at iteration ", global_step)


    def prepare_output_and_logger(self,args):   

        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
                
        # Set up output folder
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok = True)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

        # Create Tensorboard writer
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(args.model_path)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer


    def mesh_gs_training(self):
        """Joint mesh + GS post-training.

        Minimal loss set: rgb, mask, lap (mesh only), offset.
        """
        from instant_nsr.systems.mesh_deform import (
            MeshDeformNet,
            masked_l1_loss,
            build_unique_edges,
            laplacian_smoothness_loss,
        )

        include_gs   = bool(getattr(self.config.system, 'mesh_gs_include_gs',   True))
        include_mesh = bool(getattr(self.config.system, 'mesh_gs_include_mesh', False))
        assert include_gs or include_mesh, \
            'mesh_gs_training: at least one of include_gs/include_mesh must be True'

        # ── Mesh setup ────────────────────────────────────────────────────────
        v_can = faces = v_rgb = edges = None
        if include_mesh:
            if self.mesh is None:
                self.model.cuda()
                self.export(save=True)
            self._save_canon_mesh()
            v_can = self.mesh['v_pos'].float().cuda()                   # [V, 3]
            faces = torch.tensor(self.mesh['t_pos_idx']).long().cuda()  # [F, 3]
            v_rgb = self.mesh.get('v_rgb', None)
            if v_rgb is not None:
                v_rgb = (v_rgb if torch.is_tensor(v_rgb)
                         else torch.tensor(v_rgb)).float().cuda()
            edges = build_unique_edges(faces).cuda()
            if not hasattr(self, '_dr_ctx'):
                self._dr_ctx = self.dr.RasterizeCudaContext()

        # ── Viewpoints ────────────────────────────────────────────────────────
        self.viewpoint_stack = self.scene.getOriginalTrainCameras().copy()
        num_frames = len(self.viewpoint_stack)

        # ── Deform network ────────────────────────────────────────────────────
        use_trans_vox = bool(getattr(self.config.system, 'mesh_gs_use_trans_vox', True))
        reinit = bool(getattr(self.config.system, 'mesh_gs_reinit_trans_vox', False))
        if use_trans_vox:
            if reinit:
                net = type(self.gaussians.trans_vox)(
                    voxelsize=self.lp.voxelsize, out_dim=10, args=self.lp,
                ).to('cuda').train()
                self._set_mesh_deform_net(net)
            else:
                loaded = False
                if hasattr(self.config, 'cmd_args') and self.config.cmd_args.get('resume', None):
                    loaded = self._load_mesh_deform_net(for_train=True)
                if not loaded:
                    net = copy.deepcopy(self.gaussians.trans_vox).to('cuda').train()
                    self._set_mesh_deform_net(net)
        else:
            net = MeshDeformNet().to('cuda').train()
            self._set_mesh_deform_net(net)
        for p in self.mesh_deform_net.parameters():
            p.requires_grad_(True)

        mesh_lr_init  = float(getattr(self.config.system.loss, 'mesh_lr', 1e-4))
        mesh_lr_final = float(getattr(self.config.system.loss, 'mesh_lr_final', 1e-5))
        optimizer = torch.optim.Adam(
            self.mesh_deform_net.parameters(),
            lr=mesh_lr_init, eps=1e-15, betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.system.posttrain_step) + 1,
            eta_min=mesh_lr_final,
        )

        # ── Loss weights ──────────────────────────────────────────────────────
        lw = self.config.system.loss
        def _w(k, default): return float(getattr(lw, k, default))

        # Mesh branch
        w_m_rgb    = _w('lambda_mesh_rendering', 1.0)
        w_m_mask   = _w('lambda_mesh_mask',      1.0)
        w_m_lap    = _w('lambda_mesh_lap',       1.0)
        w_m_offset = _w('lambda_mesh_offset',    0.05)

        # GS branch
        w_gs_rgb    = float(self.op.lambda_rgb)
        w_gs_mask   = float(self.op.lambda_mask)
        w_gs_offset = float(self.op.lambda_delta_reg)

        total_steps = self.config.system.posttrain_step
        vis_every = int(getattr(lw, 'mesh_vis_every', 4))
        pbar = tqdm(range(total_steps + 1), desc='Mesh-GS')

        gs_image_last = None
        mesh_render_last = None
        for iter in pbar:
            idx = random.randint(0, num_frames - 1)
            viewpoint_cam = self.viewpoint_stack[idx]
            gt_image = viewpoint_cam.original_image.cuda()
            gt_mask  = viewpoint_cam.original_mask.cuda()
            t_val = torch.tensor([[viewpoint_cam.t]], dtype=torch.float32, device='cuda')

            loss = torch.zeros((), device='cuda')
            tb_scalars = {}

            # ── GS branch ─────────────────────────────────────────────────
            if include_gs:
                xyz_can = self.gaussians._xyz
                gs_delta = self.mesh_deform_net(xyz_can, t_val)[:, :3]
                gs_asset = {
                    'xyz':      xyz_can + gs_delta,
                    'scale':    self.gaussians.get_scaling,
                    'rotation': self.gaussians.get_rotation,
                    'feature':  self.gaussians.get_features,
                    'opacity':  self.gaussians.get_opacity,
                }
                render_pkg = SIMPLE_RENDER_FUNC(
                    viewpoint_cam, gs_asset,
                    self.piplin, self.background, self.scene,
                )
                gs_image = render_pkg['render']
                gs_mask  = render_pkg.get('mask', None)

                loss_gs_rgb    = l1_loss(gs_image, gt_image)
                loss_gs_mask   = (l1_loss(gs_mask, gt_mask)
                                  if (self.config.dataset.apply_mask and gs_mask is not None)
                                  else gs_image.new_zeros(()))
                loss_gs_offset = gs_delta.abs().mean()

                loss = loss + (
                    w_gs_rgb    * loss_gs_rgb
                    + w_gs_mask   * loss_gs_mask
                    + w_gs_offset * loss_gs_offset
                )
                tb_scalars['gs/rgb']    = loss_gs_rgb.item()
                tb_scalars['gs/mask']   = loss_gs_mask.item() if torch.is_tensor(loss_gs_mask) else float(loss_gs_mask)
                tb_scalars['gs/offset'] = loss_gs_offset.item()
                gs_image_last = gs_image.detach()

            # ── Mesh branch ───────────────────────────────────────────────
            if include_mesh:
                delta_v = self.mesh_deform_net(v_can, t_val)[:, :3]
                v_def   = v_can + delta_v

                render, _, _, pred_alpha, _ = render_mesh_with_nvdiffrast(
                    self.dr, self._dr_ctx, viewpoint_cam,
                    v_def, faces, self.background, v_rgb=v_rgb,
                )
                mask3 = gt_mask.expand(3, -1, -1)

                loss_m_rgb    = masked_l1_loss(render, gt_image, mask3)
                loss_m_mask   = l1_loss(pred_alpha, gt_mask)
                loss_m_lap    = laplacian_smoothness_loss(v_def, edges)
                loss_m_offset = delta_v.abs().mean()

                loss = loss + (
                    w_m_rgb    * loss_m_rgb
                    + w_m_mask   * loss_m_mask
                    + w_m_lap    * loss_m_lap
                    + w_m_offset * loss_m_offset
                )
                tb_scalars['mesh/rgb']    = loss_m_rgb.item()
                tb_scalars['mesh/mask']   = loss_m_mask.item()
                tb_scalars['mesh/lap']    = loss_m_lap.item()
                tb_scalars['mesh/offset'] = loss_m_offset.item()
                mesh_render_last = render.detach()

            # ── Backward / step ───────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f'loss:{loss.item():.4f}')

            # ── TensorBoard scalars ───────────────────────────────────────
            if self.tb_writer is not None:
                tb_step = int(self.global_epoch) + iter
                self.tb_writer.add_scalar('mesh_gs/total_loss', loss.item(), tb_step)
                self.tb_writer.add_scalar('mesh_gs/lr', optimizer.param_groups[0]['lr'], tb_step)
                for k, v in tb_scalars.items():
                    self.tb_writer.add_scalar(f'mesh_gs/{k}', v, tb_step)

            # ── Visualisation (gt | gs | mesh tile) ────────────────────────
            if vis_every > 0 and iter % vis_every == 0:
                tiles = [gt_image]
                if gs_image_last is not None:
                    tiles.append(gs_image_last.clamp(0, 1))
                if mesh_render_last is not None:
                    tiles.append(mesh_render_last.clamp(0, 1))
                vis = torch.cat(tiles, dim=2)
                vis_np = (vis.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(self.get_save_path('mesh_gs_preview.png'), vis_np[..., ::-1])
                if self.tb_writer is not None:
                    tb_step = int(self.global_epoch) + iter
                    self.tb_writer.add_image('mesh_gs/gt', gt_image.clamp(0, 1), tb_step)
                    if gs_image_last is not None:
                        self.tb_writer.add_image('mesh_gs/gs_render', gs_image_last.clamp(0, 1), tb_step)
                    if mesh_render_last is not None:
                        self.tb_writer.add_image('mesh_gs/mesh_render', mesh_render_last.clamp(0, 1), tb_step)

            # ── Periodic save ─────────────────────────────────────────────
            if iter > 0 and iter % 500 == 0:
                with torch.no_grad():
                    save_step = self.global_epoch + iter
                    if include_mesh:
                        self.export(iter, re_extract=False)
                    self.scene.save(save_step, time=0.0, only_gs=True)
                    self._save_mesh_deform_net(iter)

        # ── Final save ────────────────────────────────────────────────────────
        self.scene.save(self.global_epoch + total_steps, time=0.0, only_gs=True)
        self._save_mesh_deform_net(iter=None)


    def _canon_mesh_ckpt_path(self, ckpt_dir=None):
        if ckpt_dir is None:
            ckpt_dir = str(self.config.ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        return os.path.join(ckpt_dir, 'canon_mesh.pt')


    def _save_canon_mesh(self):
        if self.mesh is None:
            return
        path = self._canon_mesh_ckpt_path()
        payload = {
            'v_pos': self.mesh['v_pos'].detach().cpu() if torch.is_tensor(self.mesh['v_pos']) else torch.as_tensor(self.mesh['v_pos']),
            't_pos_idx': self.mesh['t_pos_idx'].detach().cpu() if torch.is_tensor(self.mesh['t_pos_idx']) else torch.as_tensor(self.mesh['t_pos_idx']),
        }
        v_rgb = self.mesh.get('v_rgb', None)
        if v_rgb is not None:
            payload['v_rgb'] = v_rgb.detach().cpu() if torch.is_tensor(v_rgb) else torch.as_tensor(v_rgb)
        torch.save(payload, path)
        print(f'[canon_mesh] saved → {path}')


    def _load_canon_mesh(self, path=None):
        """Populate self.mesh from a saved canonical mesh ckpt (skips NeuS re-extraction).

        Looks under dirname(--resume) by default. If the source path lives outside
        the current run's ckpt_dir, also copies the file into ckpt_dir for record.
        Returns True on success.
        """
        if self.mesh is not None:
            return True
        if path is None:
            resume = self.config.cmd_args.get('resume', None) if hasattr(self.config, 'cmd_args') else None
            ckpt_dir = os.path.dirname(resume) if resume else str(getattr(self.config, 'ckpt_dir', ''))
            if not ckpt_dir:
                return False
            path = os.path.join(ckpt_dir, 'canon_mesh.pt')
        if not os.path.isfile(path):
            print(f'[canon_mesh] not found: {path}')
            return False
        payload = torch.load(path, map_location='cpu')
        self.mesh = {
            'v_pos': payload['v_pos'],
            't_pos_idx': payload['t_pos_idx'],
        }
        if 'v_rgb' in payload and payload['v_rgb'] is not None:
            self.mesh['v_rgb'] = payload['v_rgb']
        print(f'[canon_mesh] loaded ← {path} (V={self.mesh["v_pos"].shape[0]}, F={self.mesh["t_pos_idx"].shape[0]})')

        cur_ckpt_dir = os.path.abspath(str(getattr(self.config, 'ckpt_dir', '') or ''))
        src_abs = os.path.abspath(path)
        if cur_ckpt_dir and os.path.dirname(src_abs) != cur_ckpt_dir:
            try:
                import shutil
                os.makedirs(cur_ckpt_dir, exist_ok=True)
                dst = os.path.join(cur_ckpt_dir, 'canon_mesh_pretrain.pt')
                shutil.copy2(src_abs, dst)
                print(f'[canon_mesh] copied pretrain canon → {dst}')
            except Exception as e:
                print(f'[canon_mesh] failed to copy pretrain canon: {e}')
        return True


    def _set_mesh_deform_net(self, net):
        """Attach mesh_deform_net WITHOUT registering it as an nn.Module child.

        Why: nn.Module.__setattr__ auto-registers nn.Module values as submodules,
        which then appear in self.state_dict(). Lightning's strict load over the
        original training checkpoint (which has no mesh_deform_net.* keys) would
        fail with "Missing key(s)". Bypassing registration keeps it fully usable
        but invisible to state_dict / load_state_dict.
        """
        self._modules.pop('mesh_deform_net', None)
        object.__setattr__(self, 'mesh_deform_net', net)


    def _mesh_deform_ckpt_path(self, iter=None):
        ckpt_dir = str(self.config.ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        if iter is None:
            return os.path.join(ckpt_dir, 'mesh_deform_net.pth')
        return os.path.join(ckpt_dir, f'mesh_deform_net_iter{int(iter)}.pth')


    def _save_mesh_deform_net(self, iter=None):
        if not hasattr(self, 'mesh_deform_net') or self.mesh_deform_net is None:
            return
        path = self._mesh_deform_ckpt_path(iter)
        payload = {
            'state_dict': self.mesh_deform_net.state_dict(),
            'global_epoch': int(self.global_epoch),
            'iter': int(iter) if iter is not None else None,
        }
        torch.save(payload, path)
        print(f'[mesh_gs_training] saved mesh_deform_net → {path}')


    def _load_mesh_deform_net(self, path=None, for_train=False):
        """Rebuild self.mesh_deform_net (deepcopy of trans_vox) and load weights from disk.

        path: explicit checkpoint path. If None, look under
              dirname(cmd_args.resume)/mesh_deform_net.pth (final), then any
              mesh_deform_net_iter*.pth in that dir.
        for_train: if True keep params trainable (train mode); else freeze (eval mode).
        Returns True on success, False if no checkpoint was found.

        When the source path lives outside the current run's ckpt_dir (e.g. finetuning
        from a different exp/<run>), the source file is also copied into
        self.config.ckpt_dir so the new run preserves its starting checkpoint.
        """
        if path is None:
            resume = self.config.cmd_args.get('resume', None) if hasattr(self.config, 'cmd_args') else None
            ckpt_dir = os.path.dirname(resume) if resume else str(getattr(self.config, 'ckpt_dir', ''))
            if not ckpt_dir:
                return False
            cand = os.path.join(ckpt_dir, 'mesh_deform_net.pth')
            if not os.path.isfile(cand):
                import glob
                snaps = sorted(glob.glob(os.path.join(ckpt_dir, 'mesh_deform_net_iter*.pth')))
                if not snaps:
                    print(f'[mesh_deform_net] no checkpoint found under {ckpt_dir}')
                    return False
                cand = snaps[-1]
            path = cand
        if not os.path.isfile(path):
            print(f'[mesh_deform_net] checkpoint not found: {path}')
            return False
        payload = torch.load(path, map_location='cuda')
        sd = payload['state_dict'] if isinstance(payload, dict) and 'state_dict' in payload else payload
        # Detect saved net class by state_dict keys: MeshDeformNet has top-level
        # `pos_enc.freqs` / `time_enc.freqs` / `net.*`; TransformField has
        # `front_net.*` / `back_net.*` / `unet.*` / `network.*`.
        keys = set(sd.keys())
        is_mesh_deform_net = any(k.startswith('pos_enc.') or k.startswith('time_enc.')
                                 or k.startswith('net.') for k in keys)
        is_trans_vox = any(k.startswith('front_net.') or k.startswith('back_net.')
                           or k.startswith('unet.') or k.startswith('network.') for k in keys)
        if is_mesh_deform_net and not is_trans_vox:
            from instant_nsr.systems.mesh_deform import MeshDeformNet
            net = MeshDeformNet().to('cuda')
        else:
            net = copy.deepcopy(self.gaussians.trans_vox).to('cuda')
        net.load_state_dict(sd)
        if for_train:
            net.train()
            for p in net.parameters():
                p.requires_grad_(True)
        else:
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
        self._set_mesh_deform_net(net)
        print(f'[mesh_deform_net] loaded ← {path}')

        # Copy source ckpt into current run's ckpt_dir (finetune record).
        cur_ckpt_dir = os.path.abspath(str(getattr(self.config, 'ckpt_dir', '') or ''))
        src_abs = os.path.abspath(path)
        if cur_ckpt_dir and os.path.dirname(src_abs) != cur_ckpt_dir:
            try:
                import shutil
                os.makedirs(cur_ckpt_dir, exist_ok=True)
                dst = os.path.join(cur_ckpt_dir, 'mesh_deform_net_pretrain.pth')
                shutil.copy2(src_abs, dst)
                print(f'[mesh_deform_net] copied pretrain ckpt → {dst}')
            except Exception as e:
                print(f'[mesh_deform_net] failed to copy pretrain ckpt: {e}')
        return True



    # ──────────────────────────────────────────────────────────────────────────












def render_mesh_with_nvdiffrast(dr, glctx, viewpoint_cam, verts_world, faces, background, v_rgb=None):
    device = verts_world.device
    faces = faces.to(device=device, dtype=torch.int32)

    ones = torch.ones((verts_world.shape[0], 1), device=device, dtype=verts_world.dtype)
    verts_h = torch.cat([verts_world, ones], dim=-1)

    full_proj = viewpoint_cam.full_proj_transform.to(device=device, dtype=verts_world.dtype)
    verts_clip = verts_h @ full_proj

    H = int(viewpoint_cam.image_height)
    W = int(viewpoint_cam.image_width)

    rast, rast_db = dr.rasterize(glctx, verts_clip[None], faces, resolution=[H, W])
    mask = (rast[..., 3:4] > 0).float()

    vert_normals_world = get_normals(verts_world, faces)[0]

    normal_img, _ = dr.interpolate(vert_normals_world[None], rast, faces, rast_db=rast_db)
    normal_img = F.normalize(normal_img, dim=-1, eps=1e-8)

    depth = rast[:, :, :, 2:3]
    valid_depth = depth[mask > 0]
    if valid_depth.numel() > 0:
        depth_img = 1 - (depth - valid_depth.min()) / (valid_depth.max() - valid_depth.min() + 1e-8)
        depth_img = depth_img * mask
    else:
        depth_img = torch.zeros_like(depth)

    bg = background.to(device=device, dtype=verts_world.dtype).view(1, 1, 1, 3)

    base_color = torch.tensor([0.7, 0.7, 0.7], dtype=verts_world.dtype, device=device).view(1, 1, 1, 3)
    N_z = normal_img[..., 2:3]
    diffuse = N_z.clamp(min=0.0)
    specular = 0.2 * torch.pow(diffuse, 16.0)
    mesh_color = (0.4 * base_color) + (0.6 * diffuse * base_color) + specular
    mesh_color = mesh_color.clamp(0.0, 1.0)
    mesh_img = mesh_color * mask + bg * (1.0 - mask)
    mesh_img = dr.antialias(mesh_img, rast, verts_clip[None], faces)

    if v_rgb is not None:
        v_rgb = v_rgb.to(device=device, dtype=verts_world.dtype)
        color_img, _ = dr.interpolate(v_rgb[None], rast, faces, rast_db=rast_db, diff_attrs='all')
        rgb = color_img * mask + bg * (1.0 - mask)
        rgb = dr.antialias(rgb, rast, verts_clip[None], faces)
    else:
        rgb = mesh_img

    # Differentiable silhouette: AA the hit mask so vertex motion produces
    # gradient along silhouette edges (hard mask has zero gradient).
    mask_soft = dr.antialias(mask, rast, verts_clip[None], faces).clamp(0.0, 1.0)

    render = rgb[0].permute(2, 0, 1).clamp(0.0, 1.0)
    depth_out = depth_img[0].permute(2, 0, 1)
    mask_out = mask_soft[0].permute(2, 0, 1)

    mask_hard = mask[0].permute(2, 0, 1)
    normal_mapped = (normal_img[0].permute(2, 0, 1) + 1.0) * 0.5 * mask_hard + 1 * (1 - mask_hard)

    mesh_out = mesh_img[0].permute(2, 0, 1).clamp(0.0, 1.0)

    return render, depth_out, normal_mapped, mask_out, mesh_out
