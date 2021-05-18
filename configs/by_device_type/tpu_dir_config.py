from configs.common_constants import EXP_NAME

CHECKPOINT_DIR = "gs://transformedd/experiments/" + EXP_NAME + "/checkpoint/"
CHECKPOINT_NAME = "cp-{epoch:04d}.ckpt"

TENSORBOARD_DIR = "gs://transformedd/experiments/" + EXP_NAME + "/tb"
TENSORBOARD_NAME = 'epoch-{}'

# DATASET_PATH = "/content/drive/MyDrive/RAMAS"
DATASET_TF_RECORDS_PATH = "gs://transformedd/transformer"

PRETRAINED_MODELS = "/content/drive/MyDrive/pretrained/"

