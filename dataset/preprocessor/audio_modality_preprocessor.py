from enum import Enum
from numbers import Real

import resampy
import tensorflow as tf
import numpy as np

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeature


class AudioFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, output_shape):
        self.output_shape = output_shape

    L3 = (46, 32, 24, 512),
    OPENSMILE_GeMAPSv01b = (1, 62)
    OPENSMILE_eGeMAPSv02 = (1, 88)
    OPENSMILE_ComParE_2016 = (1, 6373)
    # SOUNDNET = ""


class AudioModalityPreprocessor(BaseDatasetProcessor):

    def __init__(self, extractor: AudioFeatureExtractor,
                 center: bool = True,
                 target_sr: int = 48000,
                 hop_size: float = 0.1):
        self._extractor = extractor
        self.output_shape = extractor.output_shape

        self._center = center
        self._hop_size = hop_size
        self._target_sr = target_sr

        self._feature_description = {
            DatasetFeature.AUDIO_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.AUDIO_RATE.name: tf.io.FixedLenFeature([1], tf.int64),
            DatasetFeature.VIDEO_SCENE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_SHAPE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_FACE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.CLASS.name: tf.io.FixedLenFeature([], tf.string)
        }

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self.preprocess_according_to_extractor, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self.extract_features, num_parallel_calls=parallel_calls)
        return dataset

    @tf.function
    def map_record_to_dictionary_of_tensors(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    @tf.function
    def preprocess_according_to_extractor(self, serialized_example: tf.train.Example) -> dict:
        if self._extractor.name.startswith("OPENSMILE"):
            return serialized_example
        elif self._extractor == AudioFeatureExtractor.L3:
            return self._l3_preprocess_audio(serialized_example)

    @tf.function
    def extract_features(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        example = tf.io.parse_single_example(serialized_example, self._feature_description)

        audio_raw = tf.io.parse_tensor(example[DatasetFeature.AUDIO_RAW.name], tf.float32)
        audio_raw = tf.RaggedTensor.from_tensor(audio_raw)

        audio_rate = example[DatasetFeature.AUDIO_RATE.name]

        clazz = tf.io.parse_tensor(example[DatasetFeature.CLASS.name], tf.float32)
        clazz = tf.ensure_shape(clazz, (1, 3))

        return {
            DatasetFeature.AUDIO_RAW: audio_raw,
            DatasetFeature.AUDIO_RATE: audio_rate,
            DatasetFeature.CLASS: clazz
        }

    TARGET_SR = 48000

    def _center_audio(self, audio, frame_len):
        """Center audio so that first sample will occur in the middle of the first frame"""
        return np.pad(self, audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)

    def _pad_audio(self, audio, frame_len, hop_len):
        """Pad audio if necessary so that all samples are processed"""
        audio_len = audio.size
        if audio_len < frame_len:
            pad_length = frame_len - audio_len
        else:
            pad_length = int(np.ceil((audio_len - frame_len) / float(hop_len))) * hop_len \
                         - (audio_len - frame_len)

        if pad_length > 0:
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

        return audio

    def _l3_preprocess_audio(self, serialized_example):

        sr = serialized_example[DatasetFeature.AUDIO_RATE]
        audio = serialized_example[DatasetFeature.AUDIO_RAW]

        if audio.size == 0:
            raise RuntimeError('Got empty audio')

        # Warn user if audio is all zero
        if np.all(audio == 0):
            print('[WARN] Provided audio is all zeros')

        # Check audio array dimension
        if audio.ndim > 2:
            raise RuntimeError('Audio array can only be be 1D or 2D')

        elif audio.ndim == 2:
            # Downmix if multichannel
            audio = np.mean(audio, axis=1)

        if not isinstance(sr, Real) or sr <= 0:
            raise RuntimeError('Invalid sample rate {}'.format(sr))

        # Resample if necessary
        if sr != self._target_sr:
            audio = resampy.resample(audio, sr_orig=sr, sr_new=self._target_sr, filter='kaiser_best')

        audio_len = audio.size
        frame_len = self._target_sr

        hop_len = int(self._hop_size * self._target_sr)

        if audio_len < frame_len:
            print('[WARN] Duration of provided audio is shorter than window size (1 second). Audio will be padded.')

        if self._center:
            # Center audio
            audio = self._center_audio(audio, frame_len)

        # Pad if necessary to ensure that we process all samples
        audio = self._pad_audio(audio, frame_len, hop_len)

        # Split audio into frames, copied from librosa.util.frame
        n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
        x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
                                            strides=(audio.itemsize, hop_len * audio.itemsize)).T

        # Add a channel dimension
        x = x.reshape((x.shape[0], 1, x.shape[-1]))

        new_example = {}
        new_example.update(serialized_example)
        new_example.pop(DatasetFeature.AUDIO_RATE)
        new_example.pop(DatasetFeature.AUDIO_RAW)

        new_example[DatasetFeature.AUDIO] = x
        return new_example
