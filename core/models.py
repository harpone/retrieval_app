from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
from termcolor import colored
from google.cloud import storage
import os
import time
import pytorch_lightning as pl

import torch
import torch.nn as nn
from scipy.ndimage import zoom
from joblib import load as load_joblib
import numpy as np
import pandas as pd
from detectron2.data import MetadataCatalog, DatasetCatalog
from munch import Munch

from core.resnet_wider import resnet50x4
from core.config import RESIZE_TO
from core.augs import load_augs
from core.utils import visualize_segmentations, compute_visual_center, load_gcs_checkpoint
from core.dataio import blob_to_path

catalog = MetadataCatalog.get('coco_2017_train_panoptic_separated')
thing_classes = catalog.thing_classes
stuff_classes = catalog.stuff_classes
imagenet_classes = pd.read_csv('./misc/imagenet_classes.txt', header=None, index_col=[0])


class CombineOne(pl.LightningModule):
    """Model with SimCLR backbone, bottleneck layer as well as segmentation and BB regression heads.

    By default, SimCLR is *detached* from the rest since it's very good and slow to train.

    Bottleneck layer is a linear map from 8192 -> 128 with MaxEnt, i.e. essentially PCA. Optimize by minimizing
    the loss = -variance + lambda * (W.T.dot(W) - id) for each feature pixel (i.e. W is shared).

    To be used with OpenImages only for now.

    """

    def __init__(self, args=None):
        super().__init__()
        self.args = Munch(vars(args))  # Namespace to Munch dict

        with torch.no_grad():
            self.backbone = resnet50x4()  # TODO: check that really no grad during training!!
        self.bottleneck = PCALayer(8192, 128)  # TODO
        self.seg_head = ConvHead(channels_in=128,
                                 channels=args.segmentation_width,
                                 channels_out=350,
                                 depth=args.segmentation_depth,
                                 out_activation=nn.Tanh())  # <-- pos/neg gt is -1, +1 or None
        self.bbox_head = ConvHead(channels_in=128,
                                  channels=args.segmentation_width,
                                  channels_out=601,
                                  depth=args.segmentation_depth,
                                  out_activation=nn.Tanh())  # <-- pos/neg gt is -1, +1 or None

    def forward(self, x):

        features = self.backbone(x)
        codes = self.bottleneck(features)

        seg_pred = self.seg_head(codes.detach())  # codes will be trained with PCA loss
        bbox_pred = self.bbox_head(codes.detach())

        return codes, seg_pred, bbox_pred

    def training_step(self, batch, batch_idx):

        batch = batch[0]  # NOTE: batch[0] because effectively dataloader batch_size = 1 due to wds batching
        images = batch['images']
        targets = batch['targets']

        codes, seg_pred, bbox_pred = self(images)

        loss_trn, metrics_trn = self.compute_losses(heads_out, targets)

        self.log('loss_trn', loss_trn, sync_dist=True)
        if self.args.get('classifier_head', False):
            self.log('loss_xent_trn', metrics_trn['loss_xent'], sync_dist=True)
            self.log('acc_trn', metrics_trn['acc'], sync_dist=True, prog_bar=True)
        if self.args.get('sim_head', False):
            self.log('loss_sim_trn', metrics_trn['loss_sim'], sync_dist=True, prog_bar=True)
            if self.args.get('train_kl', False):
                self.log('loss_kl', metrics_trn['loss_kl'], sync_dist=True)

            z_mean = metrics_trn['z_mean']
            self.log('z_mean', z_mean, sync_dist=True)
            z_std = metrics_trn['z_std']
            self.log('z_std', z_std, sync_dist=True)

            # log similarities as image:
            if batch_idx % 50 == 0:  # because can't use result.log TODO: will this screw up distributed training??
                cosine_sims = metrics_trn['cosine_sims'].detach().cpu().numpy()
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    s = ax.imshow(cosine_sims)
                    fig.colorbar(s, ax=ax)
                    self.logger.experiment.log({'cosine_sims': fig})
                    plt.close('all')
                except:
                    pass  # probably tensorboard logging issue so pass

        # Log anomalous batch size:  TODO not working
        # b_this = len(ys)
        # if b_this != self.args.batch_size:
        #     result.log('anomalous_batch_size', b_this)

        # Log batch diversity:
        num_unique_ids = targets.get('num_unique_ids', False)
        if num_unique_ids:
            fraction_of_unique_ids = 2 * num_unique_ids[0].float() / targets['class_id'].shape[0]
            self.log('fraction_of_unique_ids', fraction_of_unique_ids, sync_dist=True)

        # Completely nonsensical logging to make sure each node and process have unique batches:
        # TODO: disable when AOK
        # TODO: not logging from each device with sync_dist=False
        # test_shuffle = ys['class_id'].sum()
        # self.log('test_shuffle', test_shuffle, sync_dist=True, reduce_fx=torch.cat)

        return loss_trn

    def validation_step(self, batch, batch_idx):

        batch = batch[0]  # NOTE: batch[0] because effectively dataloader batch_size = 1 due to wds batching
        images = batch['images']
        targets = batch['targets']

        features, heads_out = self(images)

        # Validation losses:
        loss_val, metrics_val = self.compute_losses(heads_out, targets)

        # Logging:
        fig = visualize_openimages(images, targets, heads_out)

        self.log('val_loss', loss_val, sync_dist=True, on_epoch=True)
        if self.args.get('classifier_head', False):
            # Log classification acc:
            # acc_val = metrics_val['acc']
            # print(f'\n ACC_VAL={acc_val.item()}')
            # self.log('acc_val', acc_val, sync_dist=True, on_epoch=True)

            # Log xent:
            self.log('loss_xent_val', metrics_val['loss_xent'], sync_dist=True, on_epoch=True)
        if self.args.get('sim_head', False):
            loss_sim_val = metrics_val['loss_sim']
            self.log('loss_sim_val', loss_sim_val, sync_dist=True, on_epoch=True)

            cosine_sims = metrics_val['cosine_sims'].detach().cpu().numpy()
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                s = ax.imshow(cosine_sims)
                fig.colorbar(s, ax=ax)
                self.logger.experiment.log({'cosine_sims': fig})
                plt.close('all')
            except:
                pass  # probably tensorboard logging issue so pass

    def compute_losses(self, heads_out_, targets_):

        loss = 0.
        metrics = dict()

        if 'openimages' in self.args[
            'dataset']:  # TODO: or rather get this from the `targets_` since different datasets
            loss_this, metrics_this = compute_openimages_loss(heads_out_, targets_, self.args)
            loss = loss + loss_this
            metrics.update(metrics_this)

        return loss, metrics

    def configure_optimizers(self):
        if self.args.optimizer_name == "adam":
            print("Using Adam optimizer.")
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.args.learning_rate,
                betas=(0.9, 0.99))
        elif self.args.optimizer_name == "sgd":
            print("Using SGD optimizer.")
            optimizer = optim.SGD(params=self.parameters(),
                                  lr=self.args.learning_rate,
                                  momentum=0.9,
                                  weight_decay=self.args.weight_decay,
                                  nesterov=True)
            # scheduler = CosineAnnealingLR(optimizer, self.args.warmup_peak)
            # TODO: warmup also maybe
        elif self.args.optimizer_name == 'lamb':
            print('Using LAMB optimizer.')
            optimizer = Lamb(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-6,
                             weight_decay=self.args.weight_decay)
        elif self.args.optimizer_name == 'lars':
            print('Using Lars optimizer.')
            optimizer = LARS(self.parameters(), lr=self.args.learning_rate, momentum=0.9, eta=1e-3, dampening=0,
                             weight_decay=self.args.weight_decay, nesterov=False, epsilon=1e-8)
        else:
            optimizer = None  # let PL raise the appropriate error
        scheduler = MultiStepLR(optimizer, milestones=self.args.decay_at_epochs, gamma=self.args.lr_decay_gamma)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataloader = get_dataloader(self.args, phase='train')
        return dataloader

    def val_dataloader(self):
        dataloader = get_dataloader(self.args, phase='validate')
        return dataloader

    def get_progress_bar_dict(self):
        # hack to get rid of progbar v_num
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


class SuperModel(nn.Module):

    def __init__(self):
        """Joint model combining SimCLR, detectron2 and the PCA.

        """
        super(SuperModel, self).__init__()

        # repnet:
        with torch.no_grad():
            self.repnet = resnet50x4()
            repnet_pth = '/home/heka/model_data/resnet50-4x.pth'
            try:
                state_dict = torch.load(repnet_pth)['state_dict']
            except Exception as e:
                print(colored('Local repnet checkpoint not found... downloading from GCS.', 'red'))
                if not os.path.exists('/home/heka/model_data/'):
                    os.makedirs('/home/heka/model_data/', exist_ok=True)
                blob_to_path('mldata-westeu', 'models/resnet50-4x.pth', repnet_pth)
                time.sleep(1)
                state_dict = torch.load(repnet_pth)['state_dict']
            self.repnet.load_state_dict(state_dict)
            self.repnet.eval()
            self.repnet.cuda()

        # segnet:
        cfg_fname = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'
        cfg = get_cfg()
        # cfg.MODEL.DEVICE = 'cpu'  # force CPU
        cfg.merge_from_file(model_zoo.get_config_file(cfg_fname))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.INPUT.MIN_SIZE_TEST = RESIZE_TO  # default is 800 and is pretty slow... NOTE this should be same as in augs!!
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fname)
        self.segnet = DefaultPredictor(cfg)

        # PCA
        # pca model (for per item codes):
        pca_path = '/home/heka/model_data/pca_simclr_8192.joblib'
        try:
            self.pca = load_joblib(pca_path)
        except Exception as e:
            print(colored('Local pca checkpoint not found... downloading from GCS.', 'red'))
            if not os.path.exists('/home/heka/model_data/'):
                os.makedirs('/home/heka/model_data/', exist_ok=True)
            blob_to_path('mldata-westeu', 'models/pca_simclr_8192.joblib', pca_path)
            self.pca = load_joblib(pca_path)

        # Load augs:
        self.augs = load_augs(resize_to=RESIZE_TO)

    def forward(self, img):
        """

        :param img: PIL image, ORIGINAL shape
        :return: dict where index=0 is the global entity and index>0 are the local ones
        """
        # TODO: something's just FUCKED in item preds!!
        # Segmentation:
        img_aug = self.augs['augs_base'](img)  # safer to do transform here than in Dataset

        shape_orig = np.array(list(img.size))  # w, h
        shape_current = np.array(list(img_aug.size))  # w', h'

        # Compute rescaling factor from original and current image sizes:
        aspect_orig = shape_orig[0] / shape_orig[1]
        aspect_current = shape_current[0] / shape_current[1]
        rescaler = aspect_orig / aspect_current

        outputs_seg = self.segnet(self.augs['augs_seg'](img_aug))
        seg, segments_info = outputs_seg["panoptic_seg"]
        # visualize_segmentations(augs['augs_seg'](img), seg, segments_info)  # TODO: comment when done debugging
        seg_masks = np.eye(seg.max() + 1)[seg.cpu().numpy()].transpose(2, 0, 1).astype(bool)  # [num_segs, H_, W_]

        # Drop background:
        seg_masks = seg_masks[1:]

        # Representation:
        logits, codes = self.repnet(
            self.augs['augs_rep'](img_aug)[None].cuda())  # e.g. inp: [256, 416] out: [8192, 8, 13]
        pred_img = int(logits[0].argmax().cpu())  # TODO: sanity check several of these
        codes = codes[0].detach().cpu().numpy()  # e.g. shape [8192, 8, 13]

        #### Get global code and meta:
        code_global = codes.mean(-1).mean(-1)  # [8192, ]
        code_global = self.pca.transform(code_global[None])  # [1, 128]

        # Resize segmentation to repnet shape:
        seg_masks_small = zoom(seg_masks, [1, 1 / 32, 1 / 32], order=0)  # e.g. shape [num_segs, 8, 13]
        result_dict = dict()
        pred_img = imagenet_classes.loc[pred_img][1]
        result_dict[0] = dict(code=code_global,
                              h=-1,
                              w=-1,
                              pred=pred_img,
                              is_thing=False,
                              seg_mask=None)
        for i, (seg_mask_small, seg_mask, seg_info) in enumerate(zip(seg_masks_small, seg_masks, segments_info)):

            is_thing = seg_info['isthing']

            #### Get local code and meta:
            seg_mask_area = seg_mask_small.sum()
            code_local = (seg_mask_small[None] * codes).sum(-1).sum(-1) / (seg_mask_area + 1e-8)  # [8192, ]
            code_local = self.pca.transform(code_local[None])  # [1, 128]
            pred_item = seg_info['category_id']  # corresponds to `catalog` categories (thing or stuff)
            pred_item = thing_classes[pred_item] if is_thing else stuff_classes[pred_item]

            # Visual center from large seg_mask:
            h_center, w_center = compute_visual_center(seg_mask)

            # Adjust visual center to original image coords:
            # Note that now shorter side is always size RESIZE_TO and therefore not cropped
            if shape_orig[0] / shape_orig[1]:  # landscape: cropped in w
                w_center /= rescaler
            else:  # portrait: cropped in h
                h_center *= rescaler

            result_dict[i + 1] = dict(code=code_local,
                                      h=h_center,
                                      w=w_center,
                                      pred=pred_item,
                                      is_thing=is_thing,
                                      seg_mask=seg_mask)

        return result_dict
