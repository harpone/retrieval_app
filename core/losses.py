import torch


def compute_loss(heads_out, targets, args):

    metrics = dict()
    loss = 0.
    if args.get('classifier_head', False):

        # loss, basically pos/neg regression neglecting missing labels:
        class_gt = targets['LabelVec']  # [B, n_classes]
        class_pred = heads_out['classifier_head']  # [B, n_classes]
        assert not torch.isnan(class_pred.sum())  # just in case because using nanmean...
        loss_class = nanmean(torch.abs(class_gt - class_pred))
        loss = loss + args.loss_scale_classifier * loss_class
        metrics['loss_class'] = loss_class

    if args.get('segmentation_head', False):  # TODO
        seg_gt = targets['masks']  # [B, n_classes, H_out, H_out]
        seg_pred = heads_out['segmentation_head']  # [B, n_classes, H_out, H_out]; in [-1, 1]
        assert not torch.isnan(seg_pred.sum())  # just in case because using nanmean...
        loss_seg = nanmean(torch.abs(seg_gt - seg_pred))
        loss = loss + args.loss_scale_segmentation * loss_seg
        metrics['loss_seg'] = loss_seg

    if args.get('fcos_head', False):  # TODO
        fcos_gt = targets['masks_bbox']
        fcos_pred = heads_out['fcos_head']
        assert not torch.isnan(fcos_pred.sum())  # just in case because using nanmean...
        loss_fcos = nanmean(torch.abs(fcos_gt - fcos_pred))
        loss = loss + args.loss_scale_fcos * loss_fcos
        metrics['loss_fcos'] = loss_fcos

    if args.get('sim_head', False):  # TODO
        pass

    return loss, metrics