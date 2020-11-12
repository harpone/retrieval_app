from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
from termcolor import colored
from google.cloud import storage

import torch
import torch.nn as nn
from scipy.ndimage import zoom
from joblib import load as load_joblib
import numpy as np
import pandas as pd
from detectron2.data import MetadataCatalog, DatasetCatalog

from core.resnet_wider import resnet50x4
from core.config import RESIZE_TO
from core.augs import load_augs
from core.utils import visualize_segmentations, compute_visual_center, load_gcs_checkpoint, blob_to_path

catalog = MetadataCatalog.get('coco_2017_train_panoptic_separated')
thing_classes = catalog.thing_classes
stuff_classes = catalog.stuff_classes
imagenet_classes = pd.read_csv('./model_data/imagenet_classes.txt', header=None, index_col=[0])

class SuperModel(nn.Module):

    def __init__(self):
        """Joint model combining SimCLR, detectron2 and the PCA.

        """
        super(SuperModel, self).__init__()

        # repnet:
        with torch.no_grad():
            self.repnet = resnet50x4()
            repnet_pth = './model_data/resnet50-4x.pth'
            try:
                state_dict = torch.load(repnet_pth)['state_dict']
            except Exception as e:
                print(colored('Local repnet checkpoint not found... downloading from GCS.', 'red'))
                blob_to_path('mldata-westeu', 'models/resnet50-4x.pth', repnet_pth)
                state_dict = torch.load(repnet_pth)['state_dict']
            self.repnet.load_state_dict(state_dict)
            self.repnet.eval()
            self.repnet.cuda()

        # segnet:
        cfg_fname = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'
        cfg = get_cfg()
        #cfg.MODEL.DEVICE = 'cpu'  # force CPU
        cfg.merge_from_file(model_zoo.get_config_file(cfg_fname))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.INPUT.MIN_SIZE_TEST = RESIZE_TO  # default is 800 and is pretty slow... NOTE this should be same as in augs!!
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fname)
        self.segnet = DefaultPredictor(cfg)

        # PCA
        # pca model (for per item codes):
        pca_path = './model_data/pca_simclr_8192.joblib'
        try:
            self.pca = load_joblib(pca_path)
        except Exception as e:
            print(colored('Local pca checkpoint not found... downloading from GCS.', 'red'))
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
        logits, codes = self.repnet(self.augs['augs_rep'](img_aug)[None].cuda())  # e.g. inp: [256, 416] out: [8192, 8, 13]
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

            result_dict[i+1] = dict(code=code_local,
                                    h=h_center,
                                    w=w_center,
                                    pred=pred_item,
                                    is_thing=is_thing,
                                    seg_mask=seg_mask)

        return result_dict



def load_models():
    #### load segnet (detectron2), repnet, PCA model (already trained with imagenet data):
    cfg_fname = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'  # force CPU
    cfg.merge_from_file(model_zoo.get_config_file(cfg_fname))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.INPUT.MIN_SIZE_TEST = 256  # default is 800 and is pretty slow... NOTE this should be same as in augs!!
    #cfg.INPUT.MAX_SIZE_TEST = 640
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fname)

    # segnet:
    segnet = DefaultPredictor(cfg)

    with torch.no_grad():
        # repnet:
        repnet = resnet50x4()  # TODO: no longer CUDA because I want CPU
        repnet_pth = './model_data/resnet50-4x.pth'  # TODO: from GCS or cache
        state_dict = torch.load(repnet_pth)['state_dict']
        repnet.load_state_dict(state_dict)
        repnet.eval()
        repnet.cuda()

    # pca model (for per item codes):
    pca = load('./model_data/pca_simclr_8192.joblib')  # TODO: from GCS or cache

    return dict(segnet=segnet, repnet=repnet, pca=pca)
