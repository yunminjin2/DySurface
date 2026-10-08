"""Train, validate, or render a DynamicSurface D-NeRF scene."""

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import re

import cv2
import imageio


def _write_video(path, images, fps=6):
    frames = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images if image is not None]
    if not frames:
        return
    imageio.mimwrite(str(path) + '.mp4', frames, fps=fps)
    imageio.mimsave(str(path) + '.gif', frames, duration=1 / fps, loop=0)


def _render_videos(save_dir, mesh_render, step):
    def natural_key(path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', path.name)]

    directory = Path(save_dir)
    for prefix in ('neus', 'gs', 'mesh') if mesh_render else ('neus', 'gs'):
        files = sorted(directory.glob(f'{prefix}{step}-*.png'), key=natural_key)
        _write_video(directory / prefix, [cv2.imread(str(path)) for path in files])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', default='0', help='visible GPU IDs, comma separated')
    parser.add_argument('--exp_dir', default='./exp')
    parser.add_argument('--resume', help='Lightning checkpoint')
    parser.add_argument('--resume_gs', help='GS output directory')
    parser.add_argument('--resume_iteration', default='0')
    parser.add_argument('--resume_gs_iteration', default='0')
    parser.add_argument('--resume_weights_only', action='store_true')
    parser.add_argument('--mesh_render', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--eval', action='store_true')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--predict', action='store_true')
    args, overrides = parser.parse_known_args()

    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy
    import instant_nsr.datasets
    import instant_nsr.systems
    from instant_nsr.utils.callbacks import ConfigSnapshotCallback, CustomProgressBar
    from instant_nsr.utils.misc import load_config

    config = load_config(args.config, cli_args=overrides)
    config.cmd_args = vars(args)
    config.trial_name = config.get('trial_name') or (config.get('tag', 'run') + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    run_dir = os.path.join(config.exp_dir, config.trial_name)
    config.save_dir = config.get('save_dir') or os.path.join(run_dir, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(run_dir, 'ckpt')
    config.config_dir = config.get('config_dir') or os.path.join(run_dir, 'config')

    if not config.dataset.root_dir or not os.path.isdir(config.dataset.root_dir):
        parser.error('dataset.root_dir must point to an existing D-NeRF scene directory')
    if args.validate or args.predict:
        if not args.resume or not args.resume_gs:
            parser.error('--resume and --resume_gs are required for validation and rendering')
        config.model.using_pretrain = True
        if args.validate and not any(item.startswith('trainer.limit_val_batches=') for item in overrides):
            config.trainer.limit_val_batches = 1.0
    if args.train and config.model.using_pretrain and not (args.resume_gs or config.model.using_pretrain_path):
        parser.error('GS_PRETRAIN_DIR is required when model.using_pretrain=true')

    logging.getLogger('pytorch_lightning').setLevel(logging.DEBUG if args.verbose else logging.INFO)
    pl.seed_everything(config.get('seed', 42))
    datamodule = instant_nsr.datasets.make(config.dataset.name, config.dataset)
    system = instant_nsr.systems.make(config.system.name, config, load_from_checkpoint=args.resume)

    callbacks = []
    loggers = []
    if args.train:
        callbacks = [
            ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint),
            LearningRateMonitor(logging_interval='step'),
            ConfigSnapshotCallback(config, config.config_dir, use_version=False),
            CustomProgressBar(refresh_rate=1),
        ]
        loggers = [
            TensorBoardLogger(run_dir, name='tensorboard', version=''),
            CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs'),
        ]
        if args.use_wandb:
            from pytorch_lightning.loggers import WandbLogger
            loggers.insert(0, WandbLogger(project='DynamicSurface', name=config.trial_name))

    n_gpus = len(args.gpu.split(','))
    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else 'auto',
        gradient_clip_val=1.0,
        **config.trainer,
    )
    if args.train:
        ckpt_path = args.resume if args.resume and not args.resume_weights_only else None
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
        if getattr(trainer, 'interrupted', False):
            return
        trainer.validate(system, datamodule=datamodule, ckpt_path=None)
        system.mesh_gs_training()
    elif args.validate:
        trainer.validate(system, datamodule=datamodule, ckpt_path=None)
    else:
        trainer.predict(system, datamodule=datamodule, ckpt_path=None)
        _render_videos(config.save_dir, args.mesh_render, system.global_epoch)


if __name__ == '__main__':
    main()
