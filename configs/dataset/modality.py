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
        self.FPS = 32
        self.FILE_EXT = '_Video.mp4'


class ShimmersConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.EXAMPLES_PER_SECOND = 10
        self.FILE_EXT = '_Shimmer.csv'


class SkeletonConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.EXAMPLES_PER_SECOND = 10
        self.FILE_EXT = '_Kinect.csv'


class VideoSceneModalityConfig(VideoModalityConfig):
    def __init__(self):
        super().__init__()
        self.FILE_EXT = '_web.mp4'


class AudioModalityConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.SR = 48000
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
    SHIMMERS = ShimmersConfig()
    KINECT_SKELETON = SkeletonConfig()


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


class FeaturesSetConfig:
    def __init__(self, shape, input_shape=None):
        self.shape = shape
        self.input_shape = input_shape if input_shape is not None else shape


class DatasetFeaturesSet(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, encoder, config):
        self.encoder = encoder
        self.config = config

    AUDIO = TensorEncoder(), FeaturesSetConfig(shape=(10, 32, 24, 512), input_shape=(10, 393216))
    OPENSMILE_ComParE_2016 = TensorEncoder(), FeaturesSetConfig(shape=(10, 1, 6373), input_shape=(10, 6373))
    VIDEO_SCENE_RAW = TensorEncoder(), FeaturesSetConfig(shape=0)
    VIDEO_FACE_RAW = TensorEncoder(), FeaturesSetConfig(shape=0)
    CLASS = TensorEncoder(), FeaturesSetConfig(shape=7)
    VIDEO_FACE_R2PLUS1_FEATURES = TensorEncoder(), FeaturesSetConfig(shape=(39, 512))
    VIDEO_SCENE_R2PLUS1_FEATURES = TensorEncoder(), FeaturesSetConfig(shape=(39, 512))
    VIDEO_SCENE_IV3_FEATURES = TensorEncoder(), FeaturesSetConfig(shape=(160, 2, 2, 2048), input_shape=(160, 8192))
    VIDEO_FACE_VGG_FEATURES = TensorEncoder(), FeaturesSetConfig(shape=(160, 512))
    VIDEO_FACE_IR50_FEATURES = TensorEncoder(), FeaturesSetConfig(shape=(160, 512))
    SKELETON = TensorEncoder(), FeaturesSetConfig(shape=(9, 28))
    SHIMMERS = TensorEncoder(), FeaturesSetConfig(shape=(50, 17))
    SHIMMERS_SHAPE = TensorEncoder(), FeaturesSetConfig(shape=0)
    SKELETON_SHAPE = TensorEncoder(), FeaturesSetConfig(shape=0)
