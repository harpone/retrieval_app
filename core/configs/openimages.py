
"""
TODOs:

#### UserWarning: ReadError('unexpected end of data'):
exception occurs at webdataset/dataset.py in raw_samples, maybe for a mask png...
at tardata actually!
openimages-c-train-1.tar unexpected end of file
occurs apparently for json, png,
anyway, happens at item level, not entire tar so probably not a big deal atm

SPEED ISSUE:
- getting max 40 img/s with single process, 150 img/s with 8 processes - got around 1200 img/s earlier!
- gsutil -m cp still as fast as before

"""
# TODO: warning shuffle disabled

# TODO: /opt/conda/lib/python3.7/site-packages/webdataset/dataset.py:56: UserWarning: ReadError('unexpected end of data')

# TODO: check that shuffle is really random, i.e. entire dataset seen
# TODO: warning: reload_dataloaders_every_epoch=False!!!

from math import sqrt

gpus = 8
num_nodes = 1
run_name = 'testrun'
batch_size = 8 * 8
num_workers = 4  # per device process
#limit_train_batches = 100  # TODO: note testing only!!
limit_val_batches = 5  # TODO: testing
val_check_interval = 1000
experiment = 'testruns'
resume_from = None
#resume_from = 'cuda-v100-1: efficientnet-b3, sim only nthalles, LAMB, 16 prec B=64'

# debugging switches:
debug_mode = 0  # use single TPU => no multiprocessing
use_profiler = False
num_validation_visualizations = 10

precision = 32

if 0:  # LARS
    optimizer_name = 'lars'  # 'sgd', 'adam', 'lars' or 'lamb'
    warmup_peak = 1500000 / (batch_size * (tpu_cores or gpus))  #  peak at about 9 epochs with this; ~SimCLR with LARS
    base_lr = 0.075  # default for LARS SimCLR v1 at B_total=64
if 1:  # LAMB
    optimizer_name = 'lamb'  # 'sgd', 'adam', 'lars' or 'lamb'
    warmup_peak = 1000 if resume_from is None else 0
    #base_lr = 0.00025  # from LAMB paper, ImageNet resnet50; seems too high or warmup too fast...
    base_lr = 0.00015
if 0:  # Adam
    optimizer_name = 'adam'  # 'sgd', 'adam', 'lars' or 'lamb'
    warmup_peak = 0
    base_lr = 1e-5

learning_rate = base_lr * sqrt(2 * batch_size * gpus * num_nodes)  # 2 because pairs

lr_decay_gamma = 0.2
decay_at_epochs = [100, 200]
weight_decay = 1e-4

# Dataset:
dataset = 'openimages'
shuffle_buffer = 1  # collect this many examples then shuffle; larger is better but can lead to OOM
max_epochs = 1000
num_sanity_val_steps = 0
check_val_every_n_epoch = 1
database_size_train = 1000000000  # TODO
database_size_val = 20000
shared_memory_size = '100G'
seed = 1  # TODO
# use_apex=0,  # use mixed precision training

#### Which tasks to train:
# Similarity (SimCLR):
sim_head = 0
sim_head_width = 1024
sim_head_depth = 2
sim_head_out_dim = 128
temperature = 0.5
loss_scale_sim = 1.

# KL divergence: note: not a separate head but will apply KL loss to the representations
train_kl = 0
k_knn = 3  # k:th nearest neighbor(s)
target_uniformity = .1  # 1 is uniform, 0 is binary
loss_scale_kl = 0.1

# Classifier:
classifier_head = 0
pos_neg_classifier = 1  # use pos (+1),neg (-1) or missing (NaN) labels if 1, standard smax otherwise
classifier_head_detach = 0  # whether or not to detach before classifier
classifier_width = 1024
classifier_depth = 1
classifier_out_dim = 601
loss_scale_classifier = 1.

# Segmentation:
segmentation_head = 1
segmentation_head_detach = 0
segmentation_width = 1024
segmentation_depth = 3
segmentation_out_dim = 350
loss_scale_segmentation = 1.

# FCOS bbox:
fcos_head = 0
fcos_head_detach = 0
fcos_width = 1024
fcos_depth = 1
fcos_out_dim = 601
loss_scale_fcos = 1.


detach_readout = 0  # train readout (classifier or regressor) independently of the rest of the network if 1


# Net params:
input_size = 224
hash_dim = 128  # note that this is typically very high in SimCLR
target_dimension = 128  # TODO: now this is used for similarity loss as well as in SimCLRv2
