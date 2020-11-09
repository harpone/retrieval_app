from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from joblib import load
import numpy as np

from core.resnet_wider import resnet50x4
from core.config import RESIZE_TO
from core.augs import load_augs
from core.utils import visualize_segmentations, compute_visual_center



class SuperModel(nn.Module):

    def __init__(self):
        """Joint model combining SimCLR, detectron2 and the PCA.

        """
        super(SuperModel, self).__init__()

        # repnet:
        with torch.no_grad():
            self.repnet = resnet50x4()
            repnet_pth = './model_data/resnet50-4x.pth'
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
        self.pca = load('./model_data/pca_simclr_8192.joblib')  # TODO: from GCS or cache

        # Load augs:
        self.augs = load_augs(resize_to=RESIZE_TO)

    def forward(self, img):
        """

        :param img: PIL image, transformed
        :return:
        """
        # Segmentation:
        outputs_seg = self.segnet(self.augs['augs_seg'](img))
        seg, segments_info = outputs_seg["panoptic_seg"]
        # visualize_segmentations(augs['augs_seg'](img), seg, segments_info)  # TODO: comment when done debugging
        seg_masks = np.eye(seg.max() + 1)[seg.cpu().numpy()].transpose(2, 0, 1).astype(bool)  # [num_segs, H_, W_]

        # Representation:
        logits, codes = self.repnet(self.augs['augs_rep'](img)[None].cuda())  # e.g. inp: [256, 416] out: [8192, 8, 13]
        pred_img = logits[0].argmax().cpu()  # TODO: sanity check several of these
        codes = codes[0].detach().cpu().numpy()  # e.g. shape [8192, 8, 13]

        #### Get global code and meta:
        code_global = codes.mean(-1).mean(-1)  # [8192, ]
        code_global = self.pca.transform(code_global[None])  # [1, 128]

        # Resize segmentation to repnet shape:
        seg_masks_small = zoom(seg_masks, [1, 1 / 32, 1 / 32], order=0)  # e.g. shape [num_segs, 8, 13]
        local_results = list()
        for seg_mask_small, seg_mask, seg_info in zip(seg_masks_small[1:], seg_masks[1:], segments_info):

            #### Get local code and meta:
            seg_mask_area = seg_mask_small.sum()
            code_local = (seg_mask_small[None] * codes).sum(-1).sum(-1) / (seg_mask_area + 1e-8)  # [8192, ]
            code_local = self.pca.transform(code_local[None])  # [1, 128]
            pred_item = seg_info['category_id']  # corresponds to `catalog` categories (thing or stuff)

            # Visual center from large seg_mask:
            h_center, w_center = compute_visual_center(seg_mask)

            local_results.append([code_local, h_center, w_center, pred_item])

        return code_global, pred_img, local_results



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
