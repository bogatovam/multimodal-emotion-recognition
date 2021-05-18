from configs.dataset.modality import VideoModalityConfig
from dataset.preprocessor.feature_extractors_metadata import VideoFeatureExtractor

NAME = "emotion-transformer"

EXP_NAME = "C3D"

FEATURES_COUNT = 40

SAMPLES_PER_TFRECORD = 128

BATCH_SIZE = 2

LOG_AND_SAVE_FREQ_BATCH = 10

DATASET_SIZE = 5639
NUM_EPOCHS = 30
NUM_ITER_PER_EPOCH = int(DATASET_SIZE / BATCH_SIZE)
VALIDATION_STEPS = int((DATASET_SIZE * 0.2) / BATCH_SIZE)
LEARNING_RATE = 0.0001

EXTRACTOR: VideoFeatureExtractor = VideoFeatureExtractor.C3D
INPUT_FACE_SIZE = (VideoModalityConfig().SHAPE, VideoModalityConfig().SHAPE)
FRAMES_STEP = 2

DATASET_NAME = "RAMAS"

classes = ["Angry", "Sad", "Disgusted", "Happy", "Scared", "Surprised", "Neutral", "Shame", "Tiredness"]
