from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
from termcolor import colored
import os
import time
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from scipy.ndimage import zoom
from joblib import load as load_joblib
import numpy as np
import pandas as pd
from detectron2.data import MetadataCatalog
from munch import Munch

from core.resnet_wider import resnet50x4
from core.config import RESIZE_TO
from core.augs import load_augs
from core.utils import visualize_segmentations, compute_visual_center, load_gcs_checkpoint
from core.dataio import blob_to_path, get_dataloader
from core.losses import compute_loss

catalog = MetadataCatalog.get('coco_2017_train_panoptic_separated')
thing_classes = catalog.thing_classes
stuff_classes = catalog.stuff_classes
imagenet_classes = pd.read_csv('./misc/imagenet_classes.txt', header=None, index_col=[0])

# TODO: check ConvHead out scale


class TheEye(pl.LightningModule):
    """Model with SimCLR or other fully conv backbone, bottleneck layer and segmentation layer on top.

    By default, SimCLR is *detached* from the rest since it's very good and slow to train.

    Bottleneck layer is a *linear* map from 8192 -> 128 with MaxEnt, i.e. essentially PCA. Optimize by minimizing
    the loss = -variance + lambda * (W.T.dot(W) - id) for each feature pixel (i.e. W is shared), or max knn entropy.
    - Or maybe linear + tanh and knn-ent? Manhattan/Hamming dist. and easy generalization to hashing
    - Or linear + layer norm and knn-ent, i.e. on S^{D-1}? Then cosine dist as metric

    To be used with OpenImages only for now.

    TODOs:
    - dataset create script
    - TB logger
    - just pred class or pred difference?

    """

    def __init__(self, args=None):
        super().__init__()
        self.args = Munch(vars(args))  # Namespace to Munch dict

        # TODO: eventually merge backbone and bottleneck to Encoder
        with torch.no_grad():
            self.backbone = resnet50x4()  # TODO: check that really no grad during training!!
        self.bottleneck = nn.Conv2d(8192, 128, kernel_size=1)  # TODO: check out scale/ normalization
        self.seg_head = ConvHead(channels_in=128,
                                 channels=self.args.segmentation_width,
                                 channels_out=350,
                                 depth=self.args.segmentation_depth,
                                 out_activation=nn.Tanh())  # <-- pos/neg gt is -1, +1 or None

    def forward(self, images):

        class_preds, features = self.backbone(images)  # carrying class_pred for sanity checks
        codes = self.bottleneck(features)

        seg_preds = self.seg_head(codes.detach())  # codes will be trained with PCA/MaxEnt loss

        return codes, seg_preds, class_preds

    def codify(self, images):
        """Takes in an image tensor x, obtains codes, seg_pred and uses the predicted segmentations to generate
        a code per detected item. Outputs also the segmentation and possibly other info.

        :param images: torch.tensor shape [B, C, H, W]
        :return:
        """
        codes, seg_preds, class_preds = self(images)
        # TODO

    def training_step(self, batch, batch_idx):

        batch = batch[0]  # NOTE: batch[0] because effectively dataloader batch_size = 1 due to wds batching
        images = batch['images']
        targets = batch['targets']

        codes, seg_preds, class_preds = self(images)

        loss_trn, metrics_trn = self.compute_losses(seg_preds, targets)

        self.log('loss_trn', loss_trn, sync_dist=True)
        for key, val in metrics_trn.items():
            self.log(key + '_trn', val, sync_dist=True, on_epoch=False)

        return loss_trn

    def validation_step(self, batch, batch_idx):

        batch = batch[0]  # NOTE: batch[0] because effectively dataloader batch_size = 1 due to wds batching
        images = batch['images']
        targets = batch['targets']

        codes, seg_preds, class_preds = self(images)

        loss_val, metrics_trn = self.compute_losses(seg_preds, targets)

        self.log('loss_val', loss_val, sync_dist=True)
        for key, val in metrics_trn.items():
            self.log(key + '_val', val, sync_dist=True, on_epoch=False)

        return loss_val

    def compute_losses(self, seg_preds_, targets_):

        loss = 0.
        metrics = dict()

        if 'openimages' in self.args['dataset']:
            loss_this, metrics_this = compute_loss(seg_preds_, targets_, self.args)
            loss = loss + loss_this
            metrics.update(metrics_this)

        return loss, metrics

    def configure_optimizers(self):
        print("Using Adam optimizer.")
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.99))
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


class ConvHead(nn.Module):
    """Head on top of a fully convolutional net. A fully convolutional residual net with 1x1 convs.

    """

    def __init__(self, channels_in=128, channels=1024, channels_out=128, depth=1, out_activation=None):
        super().__init__()
        out_activation = nn.Identity if out_activation is None else out_activation

        self.pre_layer = ConvLayer(channels_in, channels)
        self.res_layers = nn.ModuleList([ConvLayer(channels, channels) for _ in range(depth)])
        self.post_layer = ConvLayer(channels, channels_out, act=out_activation)

    def forward(self, x):
        """
        :param x: shape (B, channels, H / k, H / k) for int k
        :return: segmap predictions of shape [B, num_classes, H / k, H / k];
        """
        x = self.pre_layer(x)
        for fn in self.res_layers:
            x = x + fn(x)
        x = self.post_layer(x)

        return x


class ConvLayer(nn.Module):

    def __init__(self, f_in, f_out, kernel_size=1, padding=0, stride=1, init_scale=1., groups=1, act=nn.ReLU()):
        """Basic convolutional layer with ReLU, BN layers.

        :param f_in:
        :param f_out:
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        conv = nn.Conv2d(f_in, f_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        norm = nn.BatchNorm2d(f_out)
        norm.bias.data.zero_()
        norm.weight.data = init_scale * norm.weight.data
        self.fn = nn.Sequential(conv, norm, act)

    def forward(self, x):
        x = self.fn(x)
        return x


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
                print(e)  # TODO: catch and use
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
        cfg.INPUT.MIN_SIZE_TEST = RESIZE_TO  # default is 800 and is pretty slow... NOTE should be same as in augs!!
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fname)
        self.segnet = DefaultPredictor(cfg)

        # PCA
        # pca model (for per item codes):
        pca_path = '/home/heka/model_data/pca_simclr_8192.joblib'
        try:
            self.pca = load_joblib(pca_path)
        except Exception as e:
            print(e)  # TODO: catch and use
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
