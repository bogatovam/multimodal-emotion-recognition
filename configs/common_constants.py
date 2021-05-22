from configs.dataset.modality import VideoModalityConfig

NAME = "emotion-transformer"

EXP_NAME = "multimodal-sum"

FEATURES_COUNT = 40

SAMPLES_PER_TFRECORD = 128

BATCH_SIZE = 2

LOG_AND_SAVE_FREQ_BATCH = 10

DATASET_SIZE = 5639
NUM_EPOCHS = 30
NUM_ITER_PER_EPOCH = int(DATASET_SIZE / BATCH_SIZE)
VALIDATION_STEPS = int((DATASET_SIZE * 0.2) / BATCH_SIZE)
LEARNING_RATE = 0.00001

INPUT_FACE_SIZE = (224, 224)
FRAMES_STEP = 2

DATASET_NAME = "RAMAS"

classes = ["Angry", "Sad", "Disgusted", "Happy", "Scared", "Surprised", "Neutral"]
