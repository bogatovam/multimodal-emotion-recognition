from enum import Enum


class AudioFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, *output_shape):
        self.output_shape = output_shape

    # L3 = (46, 32, 24, 512),
    L3 = (1, 48000)  # just after net preprocessor
    OPENSMILE_GeMAPSv01b = (1, 62)
    OPENSMILE_eGeMAPSv02 = (1, 88)
    OPENSMILE_ComParE_2016 = (1, 6373)


class VideoFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, *output_shape):
        self.output_shape = output_shape

    C3D = (32, 112, 112, 3),
    VGG_FACE = (32, 112, 112, 3)