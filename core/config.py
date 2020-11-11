

# Globals:
MIN_MASK_SIDE = 7  # derived from SimCLR standard 224x224 input size
MAX_MASK_SIDE = 12  # pad all segmaps to (7, MAX_MASK_PAD)
CODE_LENGTH = 128  # retrieval code lenght, reduced by PCA from SimCLR out dim = 8192
RESIZE_TO = 256  # resize shorter side to this
N_RETRIEVED_RESULTS = 6  # number of images retrieved per query

