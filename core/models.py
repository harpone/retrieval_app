from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
import torch.nn as nn
from joblib import load

from core.resnet_wider import resnet50x4


class JointModel(nn.Module):

    def __init__(self):
        """Joint model combining SimCLR and detectron2.

        """
        # repnet:
        self.repnet = resnet50x4()
        repnet_pth = './resnet50-4x.pth'
        state_dict = torch.load(repnet_pth)['state_dict']
        self.repnet.load_state_dict(state_dict)

        # segnet:
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer

        cfg_fname = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'  # force CPU
        cfg.merge_from_file(model_zoo.get_config_file(cfg_fname))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.INPUT.MIN_SIZE_TEST = 256  # default is 800 and is pretty slow... NOTE this should be same as in augs!!
        # cfg.INPUT.MAX_SIZE_TEST = 640
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_fname)
        self.segnet = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def forward(self, x):
        # TODO: check BGR vs RGB segmentation quality and which done by default!


        return



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
