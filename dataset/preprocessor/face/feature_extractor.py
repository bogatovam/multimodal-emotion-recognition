import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeature
from dataset.preprocessor.feature_extractors_metadata import VideoFeatureExtractor
import numpy as np


class FeatureExtractor(BaseDatasetProcessor):

    def __init__(self,
                 pretrained_model_path,
                 input_shape,
                 output_shape,
                 frames_step,
                 window_width_in_sec=5,
                 window_step_in_sec=1, fps=32):
        self._output_shape = output_shape
        self._frames_step = frames_step
        self._input_shape = input_shape

        self._fps = fps
        self._window_step_in_sec = window_step_in_sec
        self._window_width_in_sec = window_width_in_sec

        self._feature_extractor = tf.keras.models.load_model(pretrained_model_path, compile=False)
        self._feature_extractor.trainable = False
        self._feature_extractor = self._feature_extractor.signatures["serving_default"]

        self._feature_description = {
            DatasetFeature.VIDEO_FACE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_FACE_SHAPE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.CLASS.name: tf.io.FixedLenFeature([], tf.string)
        }

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors, num_parallel_calls=parallel_calls)
        dataset = dataset.flat_map(self._split_by_windows)
        dataset = dataset.map(self._C3D_preprocess_frames, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self._extract_features_tensor_frames, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self.concat_with_labels, num_parallel_calls=parallel_calls)
        return dataset

    @tf.function
    def map_record_to_dictionary_of_tensors(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    @tf.function
    def concat_with_labels(self, example: tf.train.Example):
        return example[DatasetFeature.VIDEO_FACE_C3D_FEATURES.name], example[DatasetFeature.CLASS.name]

    @tf.function
    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        example = tf.io.parse_single_example(serialized_example, self._feature_description)

        video_fragment_shape = tf.io.parse_tensor(example[DatasetFeature.VIDEO_FACE_SHAPE.name], tf.int32)
        video_fragment = tf.io.parse_tensor(example[DatasetFeature.VIDEO_FACE_RAW.name], tf.uint8)
        clazz = tf.io.parse_tensor(example[DatasetFeature.CLASS.name], tf.double)

        clazz = tf.cast(clazz, dtype=tf.float32)
        video_fragment = tf.cast(video_fragment, dtype=tf.float32)

        clazz = tf.ensure_shape(clazz, 7)
        # clazz = tf.reshape(clazz, (1, 9))

        video_fragment_shape = tf.ensure_shape(video_fragment_shape, (4,))
        video_fragment_shape = tf.expand_dims(video_fragment_shape, 1)

        return {
            DatasetFeature.VIDEO_FACE_SHAPE.name: video_fragment_shape,
            DatasetFeature.VIDEO_FACE_RAW.name: video_fragment,
            DatasetFeature.CLASS.name: clazz
        }

    @tf.function
    def _split_by_windows(self, example: tf.Tensor) -> dict:
        clazz = example[DatasetFeature.CLASS.name]
        shape_tensor = example[DatasetFeature.VIDEO_FACE_SHAPE.name]
        frames_count = shape_tensor[0]
        frames_count = tf.expand_dims(frames_count, 1)
        pad = tf.pad(frames_count, [[0, 0], [0, 1]], constant_values=self._window_width_in_sec * self._fps)
        pad = tf.concat([pad, tf.constant(np.full((3, 2), self._window_width_in_sec * self._fps), dtype=tf.int32)],
                        axis=0)
        new_pad = tf.math.abs(tf.math.subtract(pad, self._window_width_in_sec * self._fps))
        # clazz.shape = (1, 9)
        video_frames = example[DatasetFeature.VIDEO_FACE_RAW.name]
        video_frames = tf.pad(video_frames, new_pad)
        # video_frames.shape = (None, 112, 112, 3)
        # 160 frames = 5 * 35 = 5 sec = 5 * 1 sec
        # window = 5 sec
        #
        # video_frames = self._pad_frames_according_window(video_frames)
        video_frames = tf.signal.frame(video_frames, self._window_width_in_sec * self._fps,
                                       self._window_step_in_sec * self._fps, axis=0)
        video_frames += (tf.cast(tf.math.equal(video_frames, 0), tf.float32) * -1e9)
        # video_frames.shape = (None, 160, 112, 112, 3)
        return tf.data.Dataset.from_tensor_slices(video_frames).map(lambda x: self._encode_as_dict(x, clazz))

    @tf.function
    def _C3D_preprocess_frames(self, example: tf.train.Example):
        video_frames = example[DatasetFeature.VIDEO_FACE_RAW.name]  # None, 224, 224, 3
        video_frames = tf.map_fn(self._decode_image, video_frames)
        video_frames = tf.ensure_shape(video_frames, shape=(None, 3, 112, 112))  # None, 3, 112, 112
        video_frames_tensors = tf.signal.frame(video_frames, 8, self._frames_step, axis=0)  # None, 8, 3, 112, 112
        print(f'video_frames_tensors.shape:={video_frames_tensors.shape}')
        video_frames_tensors = tf.transpose(video_frames_tensors, (0, 2, 3, 4, 1))  # None, 3, 112, 112, 8
        return {DatasetFeature.VIDEO_FACE_RAW.name: video_frames_tensors,
                DatasetFeature.CLASS.name: example[DatasetFeature.CLASS.name]}

    @tf.function
    def _extract_features_tensor_frames(self, example: tf.train.Example):
        video_frames_tensors = example[DatasetFeature.VIDEO_FACE_RAW.name]  # None, 3, 112, 112, 8
        features = tf.map_fn(lambda el: self._feature_extractor(tf.expand_dims(el, 0))['635'], video_frames_tensors)
        features = tf.map_fn(lambda el: tf.squeeze(el), features)
        features = tf.ensure_shape(features, shape=(None, 512))  # None, 3, 112, 112
        return {DatasetFeature.VIDEO_FACE_C3D_FEATURES.name: features,
                DatasetFeature.CLASS.name: example[DatasetFeature.CLASS.name]}

    @tf.function
    def _decode_image(self, image: tf.Tensor):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.ensure_shape(image, shape=(*self._input_shape, 3))
        image = tf.image.resize(image, self._output_shape)
        image = tf.transpose(image, (2, 0, 1))
        return image

    def _encode_as_dict(self, x: tf.Tensor, label: tf.Tensor) -> dict:
        return {DatasetFeature.VIDEO_FACE_RAW.name: x,
                DatasetFeature.CLASS.name: label}
