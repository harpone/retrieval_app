from itertools import islice
from munch import Munch
import torch
import sys
import os
from torch.utils.data import DataLoader
import time
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import webdataset as wds
from webdataset.filters import batched, shuffle

sys.path.append(os.getcwd())
from core.dataio import TransformOpenImages, collate_openimages

num_iters = 1000
num_workers = 1


path_to_tar = "/media/heka/TERA/Data/openimages/val/openimages-z-val-0.tar"

aug = alb.Compose([  # alb.Equalize(always_apply=True),
    alb.SmallestMaxSize(224),
    alb.CenterCrop(224, 224, always_apply=True),
    alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
    ToTensorV2()
],
    p=1,
    bbox_params=alb.BboxParams(
        format='albumentations', label_fields=['bbox_labels'])
)
transform_openimages = TransformOpenImages(aug=aug)

dataset = (
    wds.Dataset(
        path_to_tar,
        length=None,
        tarhandler=None,
    )
    # .pipe(none_filter)
    .pipe(transform_openimages)  # still in image, mask, target format; bboxes are numpy vectors
    .pipe(batched(batchsize=16, partial=True, collation_fn=collate_openimages))
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,  # because batching done in dataset
    shuffle=False,
    num_workers=num_workers,
    drop_last=True,
    collate_fn=None,
    pin_memory=False,
    worker_init_fn=None,
)

start_time = time.time()
for i, sample in enumerate(islice(dataloader, 0, num_iters)):
    print(f"\r{i}", end="")
print(f"\nend time={time.time() - start_time}")
