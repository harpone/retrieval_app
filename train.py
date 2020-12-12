"""
This file runs the main training/val loop, etc... using Lightning Trainer 
"""
import os
from os.path import join
import random
import torch


from argparse import ArgumentParser
from core import utils


def main(args):
    from pytorch_lightning import Trainer
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    from core.models import TheEye
    model = TheEye(args)

    seed = 1
    # don't seed numpy because want random dataloader init, but distr training requires same seed for model inits
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # most basic trainer, uses good defaults
    # os.environ["WANDB_API_KEY"] = args.wandb_api_key

    logger = pl_loggers.TensorBoardLogger('./tensorboard_logs')
    checkpoint_callback = ModelCheckpoint(dirpath='/media/heka/TERA/Data/openimages_models/',  # TODO:  replace when VM
                                          filename=join(args.experiment, args.run_name),
                                          monitor='loss_val')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(logger=logger,
                      checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_monitor],
                      default_root_dir=None,
                      gradient_clip_val=0,
                      gpus=args.gpus,
                      auto_select_gpus=False,  # True will assume gpus present...
                      tpu_cores=args.tpu_cores,
                      log_gpu_memory=None,
                      progress_bar_refresh_rate=1,
                      overfit_batches=0.,
                      fast_dev_run=False,
                      accumulate_grad_batches=1,
                      max_epochs=args.max_epochs,
                      limit_train_batches=vars(args).get('limit_train_batches', 1.),
                      val_check_interval=args.val_check_interval,
                      limit_val_batches=args.limit_val_batches,
                      accelerator='ddp',
                      sync_batchnorm=False,
                      precision=args.precision,
                      weights_summary='top',
                      weights_save_path=None,
                      num_sanity_val_steps=args.num_sanity_val_steps,
                      resume_from_checkpoint=args.resume_from,
                      benchmark=False,
                      deterministic=False,
                      reload_dataloaders_every_epoch=False,
                      terminate_on_nan=False,  # do NOT use on TPUs, veeeery slow!!
                      prepare_data_per_node=True,
                      amp_backend='native',
                      profiler=args.profiler)
    trainer.logger.log_hyperparams(args)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # add CLI args:
    parser.add_argument(
        "--config-module",
        default='configs.openimages',
        metavar="FILE",
        help="path to config module (usually under ./configs)",
        type=str,
    )

    # parse params
    args = parser.parse_args()
    args_config = utils.load_args_module(args)
    vars(args).update(args_config)

    main(args)
