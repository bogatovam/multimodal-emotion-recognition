from enum import Enum
import tensorflow as tf
from abc import ABC


class TimeDependentModality(ABC):
    WINDOW_STEP_IN_SECS = 1
    WINDOW_WIDTH_IN_SECS = 5


class VideoModalityConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.FRAMES_PERIOD = 5
        self.SHAPE = 224
        self.EXTRACTED_FRAMES_COUNT = 64
        self.FILE_EXT = '_Video.mp4'


class VideoSceneModalityConfig(VideoModalityConfig):
    def __init__(self):
        super().__init__()
        self.SHAPE = 224
        self.EXTRACTED_FRAMES_COUNT = 64
        self.FILE_EXT = '_web.mp4'


class AudioModalityConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.FILE_EXT = '_mic.wav'


class Modality(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config):
        self.config = config

    AUDIO = AudioModalityConfig()
    VIDEO_FACE = VideoModalityConfig()
    VIDEO_SCENE = VideoSceneModalityConfig()
    SHIMMERS = TimeDependentModality()
    KINECT_SKELETON = TimeDependentModality()


class ByteEncoder:
    def transform(self, feature):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))


class IntEncoder:
    def transform(self, feature):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))


class TensorEncoder(ByteEncoder):
    def transform(self, feature):
        feature = tf.io.serialize_tensor(tf.constant(feature)).numpy()
        return super().transform(feature)


class DatasetFeature(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, encoder):
        self.encoder = encoder

    L3 = TensorEncoder()
    VIDEO_SCENE_RAW = TensorEncoder()
    VIDEO_FACE_RAW = TensorEncoder()
    CLASS = TensorEncoder()
