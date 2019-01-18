""" config template 
"""

import os 
from yacs.config import CfgNode as CN 


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------

_C.INPUT = CN() 
_C.INPUT.SIZE_TRAIN = 800
_C.INPUT.SIZE_TEST = 800


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.IMS_PER_BATCH = 8


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
