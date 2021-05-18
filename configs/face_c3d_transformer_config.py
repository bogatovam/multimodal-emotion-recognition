from configs.dataset.modality import VideoModalityConfig
from dataset.preprocessor.feature_extractors_metadata import VideoFeatureExtractor

EXTRACTOR: VideoFeatureExtractor = VideoFeatureExtractor.C3D
INPUT_FACE_SIZE = (VideoModalityConfig().SHAPE, VideoModalityConfig().SHAPE)
FRAMES_STEP = 2

PRETRAINED_MODEL_PATH = "../models/pretrained/c3d.h5"
# PRETRAINED_MODEL_PATH = "/content/drive/MyDrive/pretrained/r2plus1d_34_clip32_ft_kinetics_from_ig65m-10f4c3bf"

FEATURES_COUNT = 40
