import io
import os
import shutil
import uuid
from detectron2.data import MetadataCatalog
import blosc
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing import Pool
import numpy as np
from munch import Munch
import skimage
import torch
import PIL
from PIL import Image
import requests
import importlib
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from google.cloud import storage
from scipy.ndimage import zoom
from skimage.filters import gaussian
from skimage.morphology import medial_axis
import types

from core.config import N_RETRIEVED_RESULTS
#import core.dataio as dataio

catalog = MetadataCatalog.get('coco_2017_train_panoptic_separated')
thing_classes = catalog.thing_classes
stuff_classes = catalog.stuff_classes

try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()  # TODO: refactor once confirmed working
except ImportError:
    print('Some dependencies not imported...')


def image_from_url(url):
    """Load image from `url`

    :param url: str
    :return:
    """
    r = requests.get(url, stream=True)
    if r.status_code == 200:  # AOK
        r.raw.decode_content = True
        img = Image.open(r.raw)
    else:
        img = None

    return img


def images_from_urls(urls, num_processes=None):
    """Load multiple images from a list of urls in parallel.

    :param urls: list of strings
    :param num_processes: 1 or None; will use all available processes if None
    :return:
    """

    if num_processes == 1:
        images = [image_from_url(url) for url in urls]
    elif num_processes is None:
        with Pool() as pool:
            images = pool.map(image_from_url, urls)
    else:
        raise NotImplementedError

    return images


def load_args_module(args_module_):

    args = importlib.import_module(args_module_.config_module).__dict__

    args_new = Munch()

    for key, val in args.items():
        if '__' not in key:  # drop module specific stuff
            if not isinstance(val, types.BuiltinFunctionType):  # drop functions etc
                args_new[key] = val
    return args_new


def image_bytes_from_url(url):
    """Load image from `url`

    :param url: str
    :return: bytes
    """
    r = requests.get(url, stream=True)
    if r.status_code == 200:  # AOK
        r.raw.decode_content = True
        img_bytes = r.raw.read()
    else:  # TODO: maybe retry logic here?
        img_bytes = None

    return img_bytes


def compress_to_bytes(arr):
    """Compresses a numpy array `arr` by using blosc and returns bytes. Inverse operation to `decompress_from_bytes`.

    :param arr: numpy.ndarray
    :return: byte string
    """
    compressed_arr = blosc.compress_ptr(arr.__array_interface__['data'][0],
                                        arr.size,
                                        arr.dtype.itemsize,
                                        clevel=3,
                                        cname='zstd',
                                        shuffle=blosc.SHUFFLE)
    arr_bytes = pickle.dumps((arr.shape, arr.dtype, compressed_arr))
    return arr_bytes


def decompress_from_bytes(arr_bytes):
    """Decompresses `arr_bytes` compressed by `compress_to_bytes` and returns a numpy array.

    :param arr_bytes:
    :return:
    """
    shape, dtype, compressed_arr = pickle.loads(arr_bytes)
    arr = np.empty(shape, dtype)
    blosc.decompress_ptr(compressed_arr, arr.__array_interface__['data'][0])

    return arr


def turbodecoder(img_bytes):
    """libjpeg-turbo based decoder for JPEG image bytes.

    :param img_bytes: e.g. open(pth_img, 'rb').read() or from `image_bytes_from_url()`
    :return: numpy uint8 array, RGB
    """
    bgr_array = jpeg.decode(img_bytes)
    return bgr_array[:, :, ::-1]  # RGB


def pildecoder(img_bytes, to_rgb=True):
    """Standard PIL based decoder for JPEG bytes.

    :param img_bytes:
    :param to_rgb:
    :return: numpy uint8 array, RGB
    """
    with io.BytesIO(img_bytes) as stream:
        img = PIL.Image.open(stream)
        img.load()  # Image.open is lazy so need this for benchmarks!
        img = np.array(img.convert('RGB')) if to_rgb else np.array(img)
    return img


class NumpyRNG:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)

    def shuffle(self, lst):
        lst = self.rng.shuffle(lst)
        return lst

    def randint(self, low, high):
        if high > low:
            rnd_int = self.rng.integers(low, high)
        else:
            rnd_int = 0
        return rnd_int

    def seed(self, seed_value):
        self.rng = np.random.default_rng(int(seed_value * 1e+8))


def nanmean(x):
    """Uses torch.nansum to compute mean over all non-NaN values.

    :param x:
    :return:
    """
    norm = (~torch.isnan(x)).sum()
    mean = torch.nansum(x) / norm
    return mean


def load_gcs_checkpoint(bucketname, blob_path):

    store = storage.Client()
    bucket = store.bucket(bucketname)
    blob = bucket.blob(blob_path)

    checkpoint_bytes = blob.download_as_string()  # about 5s
    stream = io.BytesIO(checkpoint_bytes)

    checkpoint = torch.load(stream, map_location='cpu')

    return checkpoint


def load_bytes_from_gcs(bucket_name, blob_name):
    """Remember to close!

    :param bucket:
    :param blob:
    :return: BytesIO stream
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    assert blob.exists(), f'Blob {blob_name} does not exist!'
    blob_bytes = blob.download_as_string()

    return blob_bytes


def delete_plot_cache():
    try:
        shutil.rmtree('./static/cache')
        os.makedirs('./static/cache')
    except Exception as ex:
        print(ex)
        pass


def get_mask_around_center(mask, center, smooth_scale=1.):
    """Chooses the connected component around visual `center` and smoothens it with Gaussian filter.

    About 7.61 ms @home on (256, 352) shape mask.

    :param mask: dtype=bool square numpy array
    :param center: tuple (w, h) of the visual center coordinates
    :param smooth_scale: float > 0
    :return:
    """
    # Get connected components:
    w, h = center
    labels = skimage.measure.label(mask, return_num=False)

    # Choose connected component containing visual center:
    label_at_center = labels[h, w]  # TODO: sometimes IndexError: index 768 is out of bounds for axis 1 with size 768
    mask_around_center = labels == label_at_center

    # Smoothen:
    mask_area = mask_around_center.sum()
    mask_smooth = gaussian(mask_around_center, sigma=smooth_scale * np.sqrt(mask_area) / 8)
    mask_smooth = (mask_smooth > 0.5).astype(float)

    # Connected comps again and select center one:
    labels_smooth = skimage.measure.label(mask_smooth, return_num=False)
    label_smooth_at_center = labels_smooth[h, w]
    mask_smooth_around_center = labels_smooth == label_smooth_at_center

    return mask_smooth_around_center


def get_query_plot(img_orig, img_aug, results, debug_mode=False):
    """

    :param img_orig: PIL image *before* augmentation
    :param img_aug: PIL image *after* augmentation
    :param results: results dict; output from SuperModel
    :return:
    """
    shape_orig = np.array(list(img_orig.size))  # w, h
    shape_current = np.array(list(img_aug.size))  # w, h
    scale = shape_orig.min() / shape_current.min()
    pad_amount = shape_orig.max() - int(scale * shape_current.max())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    img_np = np.array(img_orig)
    ax.imshow(img_np)  # TODO: very large images => problems...
    ax.set_xticks([])
    ax.set_yticks([])

    seg_mask_canvas = None
    for key, vals in results.items():
        if key == 0:
            continue  # display nothing for global code
        #_, h_center, w_center, pred_item, is_thing, seg_mask = vals

        # Resize seg_mask to `img` shape:
        seg_mask = zoom(vals['seg_mask'], [scale, scale], order=0)
        # Pad longer side:
        landscape = seg_mask.shape[1] > seg_mask.shape[0]
        seg_mask = np.pad(seg_mask, ((0, pad_amount if not landscape else 0), (0, pad_amount if landscape else 0)))

        #seg_mask = find_boundaries(seg_mask, mode='thick').astype(np.float32)
        #seg_mask[seg_mask < 1] = np.nan  # NaN is transparent

        w_center = vals['w']
        h_center = vals['h']
        w_center = int(w_center * img_orig.width)
        h_center = int(h_center * img_orig.height)

        seg_mask = get_mask_around_center(seg_mask, (w_center, h_center)).astype(float)

        if seg_mask_canvas is None:
            seg_mask_canvas = seg_mask
        else:
            seg_mask_canvas[seg_mask > 0.5] = key  # TODO: test!!!

        # pred_item = thing_classes[pred_item] if is_thing else stuff_classes[pred_item]
        # ax.scatter(w_center, h_center, s=500, c='r', marker='o', alpha=0.3)
        #seg_mask = seg_mask * key / (len(results) - 1)  # for display
        #ax.imshow(seg_mask, alpha=.3, cmap='hsv', norm=None)  # if is_thing else None
        text_dict = dict(boxstyle="round", fc="white", ec="green")
        if debug_mode:
            #pred_item = thing_classes[pred_item] if is_thing else stuff_classes[pred_item]
            key = str(key) + ': ' + vals['pred']
        ax.annotate(key, (w_center - 2, h_center + 2), bbox=text_dict)
        ax.set_axis_off()  # get rid of padding in figure
        # ax.annotate(pred_item, (w_center - 2, h_center + 2), bbox = text_dict)

    # plot segmentations:
    seg_mask_canvas[seg_mask_canvas < 0.5] = np.nan  # nan is transparent
    ax.imshow(seg_mask_canvas, alpha=.3, cmap='hsv', norm=None)  # if is_thing else None

    # make bytesIO:
    #buf = io.BytesIO()
    rnd_string = uuid.uuid1().hex[:16]  # need unique filename to avoid browser using cache
    query_img_path = f'./static/cache/query_img_{rnd_string}.jpg'
    os.makedirs('./static/cache/', exist_ok=True)
    fig.savefig(query_img_path, format='jpg', bbox_inches='tight', pad_inches=0)
    query_img_path = query_img_path[2:]  # need this for teh HTML
    # buf.seek(0)
    # if encode_for_html:
    #     buf = base64.b64encode(buf.getvalue()).decode('ascii')

    return query_img_path


def get_retrieval_plot(indices, entities, debug_mode=False):
    # Get corresponding entities from database:  # TODO: refactor to get_retrieval_plot etc.
    # 1) get all paths, download images to PIL in parallel
    urls = list()
    h_centers = list()
    w_centers = list()
    preds_item = list()
    is_things = list()
    for idx in indices:
        entity = entities[idx]
        h_centers.append(entity['h'])
        w_centers.append(entity['w'])
        urls.append(str(entity['url'], encoding='utf-8'))
        preds_item.append(entity['pred'])
        is_things.append(entity['is_thing'])
    images_ret = images_from_urls(urls)

    # 2) form 2 col, 3 row matplotlib plot with h_center, w_center scatter
    fig, ax = plt.subplots(N_RETRIEVED_RESULTS // 2, 2, figsize=(13, 13))
    for n in range(len(images_ret)):
        img_ret = images_ret[n]
        img_ret = np.array(img_ret)
        h, w = img_ret.shape[:-1]
        h_center = h_centers[n]
        w_center = w_centers[n]
        pred_item = preds_item[n]
        #is_thing = is_things[n]

        i = n // 2
        j = n % 2

        ax[i, j].imshow(np.array(img_ret))
        if h_center >= 0:
            ax[i, j].scatter(w_center * w, h_center * h, s=500, c='r', marker='o', alpha=0.3)
            if debug_mode:
                text_dict = dict(boxstyle="round", fc="white", ec="green")
                ax[i, j].annotate(pred_item, (w_center * w - 2, h_center * h + 2), bbox=text_dict)

        ax[i, j].set_axis_off()

    rnd_string = uuid.uuid1().hex[:16]  # need unique filename to avoid browser using cache
    retrieval_img_path = f'./static/cache/retrieval_img_{rnd_string}.jpg'
    os.makedirs('./static/cache/', exist_ok=True)
    plt.tight_layout()
    fig.savefig(retrieval_img_path, format='jpg', bbox_inches='tight', pad_inches=0)
    retrieval_img_path = retrieval_img_path[2:]

    return retrieval_img_path


def visualize_openimages(images, targets, heads_out, num_figs=1):
    """

    :param images: torch.tensor shape [B, C, H, W] in [0, 1]
    :param targets:
    :param heads_out:
    :return:
    """
    # just take the first example for now:
    figs = list()
    for i in range(num_figs):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        mask_seg_gt = targets['masks'][i].cpu().numpy()  # [n_classes_seg, H_out, W_out]
        mask_bbox_gt = targets['masks_bbox'][i].cpu().numpy()  # [n_classes_bbox, H_out, W_out]
        #labels = targets['LabelVec'][0].cpu().numpy()
        #labels = labels[~np.isnan(labels)]

        # Collect non NaNs:
        mask_seg_isnan = np.isnan(mask_seg_gt.sum(-1).sum(-1))
        mask_seg_notnan = mask_seg_gt[~mask_seg_isnan]
        mask_seg_pos = mask_seg_notnan[mask_seg_notnan.mean(-1).mean(-1) > -1]  # [num_pos_seg, H_out, W_out]
        #mask_seg_idx = np.arange(350)[~mask_seg_isnan]

        mask_bb_isnan = np.isnan(mask_bbox_gt.sum(-1).sum(-1))
        mask_bb_notnan = mask_bbox_gt[~mask_bb_isnan]
        mask_bb_pos = mask_bb_notnan[mask_bb_notnan.mean(-1).mean(-1) > -1]  # [num_pos_bb, H_out, W_out]
        mask_bb_idx = np.arange(601)[~mask_bb_isnan]

        # Plot figure
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].set_axis_off()
        ax[0, 1].set_axis_off()
        ax[1, 0].set_axis_off()
        ax[1, 1].set_axis_off()

        ax[0, 0].set_title('GT seg')
        ax[0, 0].imshow(image)
        for seg_mask in mask_seg_pos:
            seg_mask = zoom(seg_mask, [32, 32], order=0)
            seg_mask[seg_mask < 0] = np.nan  # making transparent
            ax[0, 0].imshow(seg_mask, alpha=0.5)

        # Show most "confident" prediction:
        ax[0, 1].set_title('PRED seg')
        ax[0, 1].imshow(image)
        seg_preds = heads_out['segmentation_head'][i].cpu().numpy().astype(np.float32)  # [n_seg_classes, H_out, W_out]
        seg_preds = np.argmax(seg_preds, axis=0)
        ax[0, 1].imshow(seg_preds, alpha=0.5, cmap='RdYlGn')
        # for idx in mask_seg_idx:  # Not a good idea because often no gt seg mask...
        #     seg_pred = seg_preds[idx]  # [H_out, W_out]
        #     seg_pred = (seg_pred + 1) / 2  # because in [-1, 1]
        #     seg_pred = zoom(seg_pred, [32, 32], order=0)
        #     ax[0, 1].imshow(seg_pred, alpha=0.5, cmap='RdYlGn')

        ax[1, 0].set_title('GT bb')
        ax[1, 0].imshow(image)
        for bb_mask in mask_bb_pos:
            rows = np.any(bb_mask > 0, axis=1)
            cols = np.any(bb_mask > 0, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            rmin = (rmin - 0.5) * 32
            rmax = (rmax + 0.5) * 32
            cmin = (cmin - 0.5) * 32
            cmax = (cmax + 0.5) * 32
            rect = patches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, linewidth=1, edgecolor='r',
                                     facecolor='none')
            # Add the patch to the Axes
            ax[1, 0].add_patch(rect)

        ax[1, 1].set_title('PRED bb')
        ax[1, 1].imshow(image)
        bb_preds = heads_out['fcos_head'][i].cpu().numpy().astype(np.float32)  # [n_bb_classes, H_out, W_out]
        for idx in mask_bb_idx:
            bb_pred = bb_preds[idx]  # [H_out, W_out]
            bb_pred = (bb_pred + 1) / 2  # because in [-1, 1]
            bb_pred = zoom(bb_pred, [32, 32], order=0)
            ax[1, 1].imshow(bb_pred, alpha=0.5, cmap='RdYlGn')

        figs.append(fig)

    return figs


def visualize_segmentations(img, seg, seg_info):

    cfg = get_cfg()
    cfg_fname = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_fname))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    v = Visualizer(img[:, :, ::-1], catalog, scale=.8)
    out = v.draw_panoptic_seg_predictions(seg.cpu(), seg_info)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(out.get_image()[:, :, ::-1])
    plt.show()


def compute_visual_center(mask):
    """First find the distances to nearest boundary point, then highest point.

    :param mask:
    :return:
    """
    # Boundaries to 0 to avoid centers at edges:
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    _, distance = medial_axis(mask, return_distance=True)
    h_center, w_center = np.unravel_index(distance.argmax(), np.array(distance).shape)

    h_center /= mask.shape[0]
    w_center /= mask.shape[1]

    return h_center, w_center
