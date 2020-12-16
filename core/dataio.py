from PIL import Image
import torch
import io
import numpy as np
import tables as tb
import os
import cv2
import json
from os.path import join
import collections
from google.cloud import storage
from termcolor import colored
from scipy.ndimage import zoom
import pandas as pd
import time
import warnings
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import webdataset as wds
from webdataset.filters import batched, shuffle


from core.config import CODE_LENGTH
import core.utils as utils


def collate_openimages(batch):
    """Packs `imgs` and `masks` to a minibatch tensor.
    :param batch: list of (image, target)
    :return:
    """
    # Unpack if lists:
    imgs = list()
    masks = list()
    masks_bbox = list()
    targets = collections.defaultdict(list)

    keys_used = ['LabelVec']  # has pos (+1), neg (-1) or not present (NaN) labels

    for image, target in batch:
        if image is None or target is None:  # maybe there was a read error/ corrupt example so skip
            continue
        imgs.append(torch.as_tensor(image, dtype=torch.float32))
        masks.append(torch.as_tensor(target['mask'], dtype=torch.float32))
        masks_bbox.append(torch.as_tensor(target['mask_bbox'], dtype=torch.float32))
        [targets[key].append(torch.as_tensor(target[key], dtype=torch.float32)) for key in keys_used]

    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack(masks, dim=0)
    masks_bbox = torch.stack(masks_bbox, dim=0)
    targets['masks'] = masks
    targets['masks_bbox'] = masks_bbox

    for key in keys_used:
        targets[key] = torch.stack(targets[key], dim=0)

    return dict(images=imgs, targets=targets)


def read_csv_url(url, cache=True):
    return


def authenticate_gcs_urls(url):
    """Adds authentication string to an `url`. `url` can be in brace form.

    Args:
        url (str): [description]

    Returns:
        [str]: authenticated url
    """
    url = replace_gcs_endpoint(url)
    url = f'pipe:curl -L -s -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) '
    +url
    +" || true"

    return url


def replace_gcs_endpoint(url):
    """Replace the GCS `url` with the JSON API object download endpoint, as per here:
    https://cloud.google.com/storage/docs/request-endpoints#typical
    Also appends '?alt=media'.

    :param url: e.g. gs://imidatasets/cat.jpg
    :return: https://storage.googleapis.com/storage/v1/b/imidatasets/o/cat.jpg?alt=media
    """

    path = url.split("//")[1]
    bucket = path.split("/")[0]
    rest = "%2F".join(path.split("/")[1:])
    url = (
        "https://storage.googleapis.com/storage/v1/b/"
        + bucket
        + "/o/"
        + rest
        + "?alt=media"
    )
    return url


class TransformOpenImages:
    def __init__(self, aug=None):
        super().__init__()
        # TODO: maybe cache or url/uri/path as kwarg
        bbox_class_names_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
        seg_class_names_url = (
            "https://storage.googleapis.com/openimages/v5/classes-segmentation.txt"
        )
        class_names_bbox = pd.read_csv(bbox_class_names_url, header=None).to_dict()
        class_names_seg = pd.read_csv(seg_class_names_url, header=None).to_dict()
        self.idx2name_bbox = class_names_bbox[0]
        self.name2idx_bbox = {val: key for key, val in self.idx2name_bbox.items()}
        self.idx2description_bbox = class_names_bbox[1]
        self.description2idx_bbox = {
            val: key for key, val in self.idx2description_bbox.items()
        }
        self.idx2name_seg = class_names_seg[0]
        self.name2idx_seg = {val: key for key, val in self.idx2name_seg.items()}
        self.num_classes_bbox = len(self.idx2name_bbox)
        self.num_classes_seg = len(class_names_seg[0])
        self.aug = aug

    def __call__(self, src):
        """Apply Albumentations transformations to `image`, `mask` and bboxes in `target`.
        Bounding boxes are also transformed to numpy array vectors.

        scratchpad for debugging:
        num_ids_present = np.arange(len(mask_bbox))[mask_bbox.sum(-1).sum(-1) > 0]

        :param src:
        :return:
        """
        # decode:
        src = self.decode_openimages(src)
        for image, mask, target in src:
            if self.aug:
                # Gather bboxes as list of [x_min, y_min, x_max, y_max], relative coords:
                x_mins = target.get("XMin", [])
                y_mins = target.get("YMin", [])
                x_maxs = target.get("XMax", [])
                y_maxs = target.get("YMax", [])
                x_min1s = target.get("XMin1", [])
                y_min1s = target.get("YMin1", [])
                x_max1s = target.get("XMax1", [])
                y_max1s = target.get("YMax1", [])
                x_min2s = target.get("XMin2", [])
                y_min2s = target.get("YMin2", [])
                x_max2s = target.get("XMax2", [])
                y_max2s = target.get("YMax2", [])
                bboxes = list(
                    zip(x_mins, y_mins, x_maxs, y_maxs, ["bbox"] * len(x_mins))
                )
                bboxes += list(
                    zip(x_min1s, y_min1s, x_max1s, y_max1s, ["bbox1"] * len(x_min1s))
                )
                bboxes += list(
                    zip(x_min2s, y_min2s, x_max2s, y_max2s, ["bbox2"] * len(x_min2s))
                )

                # Need to include bbox labels in aug since bboxes may be dropped:
                labels_bbox = target["LabelNameBB"]
                labels_1 = target["LabelName1"]
                labels_2 = target["LabelName2"]
                bbox_labels = labels_bbox + labels_1 + labels_2

                augmented = self.aug(image=image, mask=mask, bboxes=bboxes, bbox_labels=bbox_labels)
                image = augmented["image"]
                mask = augmented["mask"]
                bboxes = augmented["bboxes"]
                bbox_labels = augmented["bbox_labels"]
                bboxes_base = np.array(
                    [bbox[:4] for bbox in bboxes if bbox[-1] == "bbox"]
                )
                bboxes_1 = np.array(
                    [bbox[:4] for bbox in bboxes if bbox[-1] == "bbox1"]
                )
                bboxes_2 = np.array(
                    [bbox[:4] for bbox in bboxes if bbox[-1] == "bbox2"]
                )

                # Process bounding box coords:
                if len(bboxes_base) > 0:
                    target["XMin"] = bboxes_base[:, 0]
                    target["YMin"] = bboxes_base[:, 1]
                    target["XMax"] = bboxes_base[:, 2]
                    target["YMax"] = bboxes_base[:, 3]
                else:
                    target["XMin"] = np.array([0.0])
                    target["YMin"] = np.array([0.0])
                    target["XMax"] = np.array([0.0])
                    target["YMax"] = np.array([0.0])

                # Maybe process visual relation bbox coords:
                if len(bboxes_1) > 0 and len(bboxes_2) > 0:
                    target["XMin1"] = bboxes_1[:, 0]
                    target["YMin1"] = bboxes_1[:, 1]
                    target["XMax1"] = bboxes_1[:, 2]
                    target["YMax1"] = bboxes_1[:, 3]
                    target["XMin2"] = bboxes_2[:, 0]
                    target["YMin2"] = bboxes_2[:, 1]
                    target["XMax2"] = bboxes_2[:, 2]
                    target["YMax2"] = bboxes_2[:, 3]

                # Get label indices for xent loss:
                labels_img = np.array(target["LabelNameImage"])
                labels_img_int = np.array(
                    [self.name2idx_bbox[lbl] for lbl in labels_img]
                )
                label_presence = np.array(target["LabelPresence"])
                positive_labels_int = labels_img_int[label_presence == 1]
                negative_labels_int = labels_img_int[label_presence == 0]
                negative_labels = labels_img[label_presence == 0]  # will be used in centerness masks also
                labels_img_vec = np.zeros([len(self.idx2name_bbox), ], dtype=np.float32)
                labels_img_vec.fill(np.nan)  # NaN means not present
                labels_img_vec[positive_labels_int] = 1
                labels_img_vec[negative_labels_int] = -1

                target["LabelIntImage"] = labels_img_int
                target["LabelVec"] = labels_img_vec

                # Get label indices for bboxes:
                labels_bb_int = np.array(
                    [self.name2idx_bbox[lbl] for lbl in bbox_labels[: len(bboxes_base)]]
                )
                target["LabelIntBB"] = labels_bb_int

                # Get label indices for relations:
                # TODO oops these are the relation tags... finish up later if needed
                # TODO note need bbox_labels[len(bboxes_base):len(bboxes_base) + len(bboxes_1)] etc
                # labels_1_int = np.array([name2idx[lbl] for lbl in labels_1])
                # labels_2_int = np.array([name2idx[lbl] for lbl in labels_2])
                # target['LabelInt1'] = labels_1_int
                # target['LabelInt2'] = labels_2_int

                # Generate downsampled segmentation mask:
                mask = self.get_openimages_segmask(mask, negative_labels)

                # Generate downsampled bbox mask:
                mask_bbox = self.get_openimages_bbmask(
                    bboxes_base,
                    labels_bb_int,
                    mask.shape[1:],
                    negative_labels_int,
                    self.num_classes_bbox,
                )

                target["mask"] = mask  # [H, W], int valued
                target["mask_bbox"] = mask_bbox  # [num_classes, H, W], float valued

            yield image, target

    @staticmethod
    def decode_openimages(src):
        """Decode openimages data in the form of (image.jpg, mask.png, target.json) to
        (uint8 numpy array of shape [H, W, C], uint8 numpy array of shape [H, W], dict) respectively.

        Decode to uint8 numpy arrays because that's what albumentations works with.

        :param src:
        :return img, mask, target: img: uint8 ndarray shape [H, W, 3]; mask: int32 ndarray shape [H, W]; target: dict
        """
        # TODO: maybe replace PIL.Image.open with libjpg-turbo? Or pytorch native open?
        for sample in src:
            try:
                img = sample['image.jpg']
                mask = sample['mask.png']
                target = sample['targets.json']
            except KeyError:  # not found
                continue
            with io.BytesIO(img) as stream:
                img = Image.open(stream)
                #img.load()
                img = np.array(img.convert('RGB'))

            with io.BytesIO(mask) as stream:
                mask = Image.open(stream)
                #mask.load()
                mask = np.array(mask)

            target = json.loads(target)

            # Filter nones now that all are loaded:
            if (img is None) or (mask is None) or (target is None):
                continue

            yield img, mask, target

    def get_openimages_segmask(self, mask, negative_labels_):
        """Get positives/ negatives segmentation mask by using global negative labels.

        :param mask: original image shape int mask, shape [H, W]; values are the positive labels
        :param negative_labels_: list/array of ints
        :return:
        """
        mask = zoom(
            mask, 1 / 32, order=0
        )  # TODO: maybe 32 to args although prolly will be constant for all eternity

        # Mask to pos/neg:
        mask_seg = np.zeros((self.num_classes_seg,) + mask.shape, dtype=np.float32)
        mask_seg.fill(np.nan)  # NaN = missing by default

        # Fill negative labels:
        neg_labels_int = [
            self.name2idx_seg.get(name, None) for name in negative_labels_
        ]
        for neg_label in neg_labels_int:
            if neg_label is not None:
                mask_seg[neg_label] = -1

        # Fill positive labels:
        seg_labels = np.unique(mask)
        for seg_label in seg_labels:
            if seg_label > 0:
                mask_seg[seg_label][
                    np.isnan(mask_seg[seg_label])
                ] = -1  # -1 outside of segmentation
                mask_seg[seg_label][mask == seg_label] = 1

        return mask_seg

    def get_openimages_bbmask(
        self, bboxes, labels, shape, negative_labels, num_classes
    ):
        """Form a bbox centerness mask from a list/array of bboxes, accompanying labels and output mask shape.

        NOTE: I'm actually not generating an FCOS detection bbox regression targets to bbox edges but just
        a mask with distance to boundary per class. For each label and (feature-) pixel in [H, W], we have NaN if no
        positive or negative labels, -1 if there is an image level negative label, a value `centerness` > 0 if the pixel
        center is at distance `centerness` from the bbox boundary, or -1 if the feature pixel is outside of a
        bounding box.

        NOTE: think if it's better to actually discard values 0 outside bbox, since not all bboxes may be present?
        OpenImages does seem to have lots of bboxes though, so maybe I can keep them.

        NOTE 2: negative labels may not be needed, since I'm doing regression to centerness, which can't be guessed
        to be one as for binary xent.

        NOTE 3: or should I just take all nonexisting bboxes as negatives? FCOS did that! For rep learning, the former
        may be better though...

        :param bboxes: shape [num_bboxes, 4] where the last dim are the `XMin`, `YMin`, `XMax`, `YMax` coords.
        :param labels: int, shape [num_bboxes, ]
        :param shape: tuple (H, W)
        :param negative_labels: if not None or len() > 0, non-bbox pixels will be assigned these negative labels.
        :param num_classes:
        :return: shape [num_classes, H, W] centerness mask; values are distances to the nearest edge measured
            from center of a (feature) pixel.
        """
        # TODO: I could easily replace NaNs with -1 to force not present to be negatives!
        assert len(bboxes) == len(labels)

        height, width = shape
        h_centers = np.arange(height) + 0.5
        w_centers = np.arange(width) + 0.5
        mask_bbox = np.zeros((num_classes,) + shape, dtype=np.float32)
        mask_bbox.fill(np.nan)  # nan = does not contribute

        # add negative labels as -1:
        for neg_label in negative_labels:
            mask_bbox[neg_label] = -1

        for label, bbox in zip(
            labels, bboxes
        ):  # note: can be multiple same label bboxes!
            left = np.broadcast_to(w_centers[None] - bbox[0] * width, shape)  # [H, W]
            right = np.broadcast_to(bbox[2] * width - w_centers[None], shape)
            bottom = np.broadcast_to(
                bbox[3] * height - h_centers[:, None], shape
            )  # note bottom is last row!
            top = np.broadcast_to(h_centers[:, None] - bbox[1] * height, shape)
            distances2edges = np.stack([left, right, top, bottom], axis=0)  # [4, H, W]
            distances2edges[distances2edges < 0.0] = 0  # discard distances outside bbox
            centerness = np.min(distances2edges, axis=0)  # [H, W]
            mask_bbox[label][
                np.isnan(mask_bbox[label])
            ] = -1  # tag as negative by default
            mask_bbox[label][centerness > 0.0] = centerness[
                centerness > 0.0
            ]  # replace only positive parts
            # TODO: how about overlapping, same label bboxes? Pretty rare occurrence...

        return mask_bbox


def filter_nones(src, has_keys=None):
    """Some images, targets may be missing or None because of corrupt data, so filter them out.

    For webdatasets only.

    :param src: generator outputting a dict with keys `__key__`, `jpg`, `json`
    :return:
    """
    # TODO: maybe keys as kwarg
    for sample in src:
        no_nones = not any([value is None for _, value in sample.items()])
        if has_keys:
            has_all_keys = all(val in sample.keys() for val in has_keys)
        else:
            has_all_keys = True
        if has_all_keys and no_nones:
            yield sample


def decode_openimages(src):
    """Decode openimages data in the form of (image.jpg, mask.png, target.json) to
    (uint8 numpy array of shape [H, W, C], uint8 numpy array of shape [H, W], dict) respectively.

    Decode to uint8 numpy arrays because that's what albumentations works with.

    :param src:
    :return img, mask, target: img: uint8 ndarray shape [H, W, 3]; mask: int32 ndarray shape [H, W]; target: dict
    """
    for sample in src:
        try:
            img_bytes = sample['image.jpg']  # still bytes
            mask_bytes = sample['mask.png']  # still bytes
            target_bytes = sample['targets.json']  # still bytes
        except KeyError:  # not found  # TODO: maybe catch & log
            continue
        try:
            img = utils.turbodecoder(img_bytes)
        except OSError:  # "Unsupported color conversion request"?
            img = utils.pildecoder(img_bytes)
        mask = utils.decompress_from_bytes(mask_bytes).astype(np.int16)  # alb can't handle uint16
        target = json.loads(target_bytes)

        # add __key__ (good for debugging):
        target['__key__'] = sample['__key__']

        #if target['__key__'] == 'c96be80d96718e10':  # TODO: debugging
        #    print()

        if len(target['LabelNameImage']) == 0:  # discard no labels for now
            continue

        # Filter nones now that all are loaded:
        if (img is None) or (mask is None) or (target is None):
            continue

        yield img, mask, target


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloader(args, phase="train", method='gsutil'):
    if phase == "train":
        transform = alb.Compose(
            [
                alb.RandomResizedCrop(
                    args.input_size,
                    args.input_size,
                    scale=(0.2, 1),
                    ratio=(3 / 4, 4 / 3),
                    always_apply=True,
                ),
                # alb.Rotate(limit=45, p=0.75, border_mode=0),
                alb.MotionBlur(p=0.5),
                alb.HorizontalFlip(p=0.5),
                alb.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.8, p=0.8
                ),
                alb.ToGray(p=0.25),
                alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
                ToTensorV2(),  # TODO: need this?
            ],
            p=1,
            bbox_params=alb.BboxParams(format="albumentations"),
        )
        urls = args.urls_train

    elif phase == "validate":
        transform = alb.Compose(
            [  # alb.Equalize(always_apply=True),
                alb.SmallestMaxSize(args.input_size),
                alb.CenterCrop(args.input_size, args.input_size, always_apply=True),
                alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
                ToTensorV2(),
            ],
            p=1,
            bbox_params=alb.BboxParams(
                format="albumentations", label_fields=["bbox_labels"]
            ),
        )
        urls = args.urls_val
    else:
        raise NotImplementedError

    if method == 'gsutil':
        urls = f"pipe:gsutil cp {urls} -"  # or 'gsutil cat {urls}' but cp has some checksum stuff I think
    else:  # curl with authentication
        urls = authenticate_gcs_urls(urls)

    # tarhandler=warn_and_continue because sometimes rarely corrupt jpg
    def warn_and_cont(exn):
        """Called in an exception handler to ignore any exception, isssue a warning, and continue.
        Same as in wds but no sleep.
        """
        warnings.warn(repr(exn))
        return True

    shuffle_buffer = args.shuffle_buffer if phase == "train" else 10

    transform_openimages = TransformOpenImages(aug=transform)

    def augment(src):
        return transform_openimages(src)

    def none_filter(src):  # TODO: check
        return filter_nones(src, has_keys=["image.jpg", "targets.json"])

    dataset = (
        wds.Dataset(
            urls,
            length=None,
            # tarhandler=None,
            tarhandler=warn_and_cont,
        )
        # .pipe(none_filter)
        .pipe(decode_openimages)
        .pipe(augment)  # still in image, mask, target format; bboxes are numpy vectors
        .pipe(shuffle(shuffle_buffer, initial=100, rng=utils.NumpyRNG()))
        .pipe(
            batched(
                batchsize=args.batch_size,
                partial=True,
                collation_fn=collate_openimages,
            )
        )
    )
    # nominal = args.nominal_dataset_len // args.batch_size // (args.tpu_cores or args.gpus)
    # dataset = wds.ResizedDataset(dataset, length=5000000, nominal=nominal)
    # no need for ResizedDataset if Multidataset

    if 1:  # basic DataLoader

        def collate_id(x_):
            return x_

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # because batching done in dataset
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_id,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
        )
    if 0:  # try to get multidataset working, because then no need to worry about dataset size

        def unbatch_dct(data):
            """
            :param data: MultiDatasetIterator with (x_batch, y_batch=dict)
            :yield: one example pair (x, y)
            """
            for sample in data:
                assert isinstance(sample, (tuple, list)), sample
                assert len(sample) > 0
                for i in range(len(sample[0])):
                    yield sample[0][i], {key: val[i] for key, val in sample[1].items()}

        loader = (
            wds.MultiDataset(
                dataset,
                workers=args.num_workers
                if phase == "train"
                else 1,  # 1 for val because OOM easily
                pin_memory=False,  # problems with dict targets
                output_size=10000,
            )
            # .pipe(unbatch_dct)
            # .pipe(shuffle(shuffle_buffer, initial=shuffle_buffer, rng=NumpyRNG()))
            # .pipe(batched(batchsize=args.batch_size, partial=True, collation_fn=recollate_ytbb))
        )

    return loader


def blob_to_path(bucketname, blob_path=None, local_path=None):

    store = storage.Client()
    bucket = store.bucket(bucketname)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def capture_webcam():
    """Takes one photo with webcam.

    :return:
    """
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")

    _, frame = video_capture.read()  # Read picture. ret === True on success

    # Close device
    video_capture.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def upload_to_gcs(bucketname, blob_path=None, local_path=None):
    """Upload a file to GCS.

    """
    store = storage.Client()
    bucket = store.bucket(bucketname)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    return


class Database:
    def __init__(
        self,
        database_name,
        url_max_len=128,
        mode="w",
        title=None,
        expected_rows=1000000,
    ):

        self.data_root = "/home/heka/model_data/"
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root, exist_ok=True)

        # if 'gs:' in database_name:  # get from cloud storage  # TODO: shit, refactor or delete
        #     store = storage.Client()
        #     bucket = store.bucket('mldata-westeu')
        #     database_name = database_name.split('mldata-westeu')[-1]  # TODO fuck this is ugly
        #     blob = bucket.blob(database_name[1:])
        #     blob.download_to_filename(database_name)

        class Entity(tb.IsDescription):
            """Metadata for a given item. Aligned with `code_arr` and `segmask_arr`.

            url: image url
            h_center: visual center height in relative coords (in [0, 1]) of item segmentation
            w_center:
            global_code: bool; True if image level global code
            """

            url = tb.StringCol(url_max_len)
            h = tb.Float16Col()
            w = tb.Float16Col()
            pred = tb.StringCol(32)
            is_thing = tb.BoolCol()

        if mode == "w":
            self.h5file = tb.open_file(
                join(self.data_root, database_name), mode=mode, title=title
            )

            # Schema:
            self.table = self.h5file.create_table(
                self.h5file.root,
                "entities",
                Entity,
                "Entity metadata",
                expectedrows=expected_rows,
            )
            self.entities = self.table.row
            self.codes = self.h5file.create_earray(
                self.h5file.root,
                "codes",
                atom=tb.Float16Atom(),
                shape=(0, CODE_LENGTH),
                expectedrows=expected_rows,
            )

        else:
            try:
                print(
                    colored(
                        f"Trying to load database from {join(self.data_root, database_name)}"
                    )
                )
                self.h5file = tb.open_file(
                    join(self.data_root, database_name), mode=mode
                )
            except Exception as e:  # TODO: catch? OSError at least maybe also read error
                print(e)  # TODO catch and use
                print(
                    colored("Local database not found... downloading from GCS.", "red")
                )
                blob_to_path(
                    "mldata-westeu",
                    blob_path=join("databases", database_name),
                    local_path=join(self.data_root, database_name),
                )
                time.sleep(1)
                self.h5file = tb.open_file(
                    join(self.data_root, database_name), mode=mode
                )

            self.table = self.h5file.root.entities
            self.entities = self.table.row
            self.codes = self.h5file.root.codes

        self.table_keys = list(self.table.coldescrs.keys())

    def append_to_store(self, url=None, code=None, h=0, w=0, pred=None, is_thing=False):
        self.codes.append(code.astype(np.float16))
        self.entities["url"] = url
        self.entities["h"] = float(h)
        self.entities["w"] = float(w)
        self.entities["pred"] = pred
        self.entities["is_thing"] = is_thing
        self.entities.append()

    def cat(self, other):
        # TODO: about 1 min for 600k codes... faster way?
        # TODO: how does perf suffer because expectedrows?
        i = 0
        while True:
            try:
                code, entity = other[i]
                self.append_to_store(
                    str(entity["url"], encoding="utf-8"),
                    code[None],
                    h=entity["h"],
                    w=entity["w"],
                    pred=entity["pred"],
                    is_thing=entity["is_thing"],
                )
                i += 1
            except TypeError:  # TODO: catch
                raise
            except Exception as e:
                if "out of range" in e.args[0]:  # TODO: ugly
                    print("Done.")
                    break
                else:
                    raise e

    def flush(self):
        self.h5file.flush()

    def close(self):
        self.h5file.close()

    def __getitem__(self, i):

        code = self.codes[i]
        entity_list = list(self.table[i])
        entity = {key: val for key, val in zip(self.table_keys, entity_list)}

        return code, entity
