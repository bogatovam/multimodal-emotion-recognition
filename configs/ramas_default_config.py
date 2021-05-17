from configs.dataset.ramas_config import *
from configs.face_c3d_transformer_config import *

NAME = "emotion-transformer"

EXP_NAME = "C3D"

SAMPLES_PER_TFRECORD = 128

BATCH_SIZE = 1
CHECKPOINT_DIR = "..\\experiments\\" + EXP_NAME + "\\checkpoint\\"
CHECKPOINT_NAME = "cp-{epoch:04d}.ckpt"

TENSORBOARD_DIR = "..\\experiments\\" + EXP_NAME + "\\tb"
TENSORBOARD_NAME = 'epoch-{}'

LOG_AND_SAVE_FREQ_BATCH = 10

DATASET_SIZE = 5639
NUM_EPOCHS = 30
NUM_ITER_PER_EPOCH = int(DATASET_SIZE / BATCH_SIZE)
VALIDATION_STEPS = int(DATASET_SIZE / BATCH_SIZE)
LEARNING_RATE = 0.0001
