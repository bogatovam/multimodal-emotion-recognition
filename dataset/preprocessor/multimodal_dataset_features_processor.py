import sys

import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeaturesSet


class MultimodalDatasetFeaturesProcessor(BaseDatasetProcessor):

    def __init__(self, modalities_list):
        self._modalities_list = modalities_list

        self._feature_description = {
            DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_FACE_IR50_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            # DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.AUDIO.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SHIMMERS.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SHIMMERS_SHAPE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SKELETON.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SKELETON_SHAPE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.CLASS.name: tf.io.FixedLenFeature([], tf.string)
        }

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors, num_parallel_calls=parallel_calls)
        if DatasetFeaturesSet.SHIMMERS in self._modalities_list:
            dataset = dataset.filter(self.is_shape_exists)
        if DatasetFeaturesSet.SKELETON in self._modalities_list:
            dataset = dataset.filter(self.is_shape_skeleton_exists)

        dataset = dataset.map(self._extract_specified_modalities_and_ensure_shape, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self.concat_with_labels, num_parallel_calls=parallel_calls)
        return dataset

    @tf.function
    def map_record_to_dictionary_of_tensors(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    @tf.function
    def is_shape_exists(self, features_dict: tf.train.Example) -> dict:
        return tf.math.reduce_any(tf.not_equal(features_dict[DatasetFeaturesSet.SHIMMERS_SHAPE.name], 0))

    @tf.function
    def is_shape_skeleton_exists(self, features_dict: tf.train.Example) -> dict:
        return tf.math.reduce_any(tf.not_equal(features_dict[DatasetFeaturesSet.SKELETON_SHAPE.name], 0))

    @tf.function
    def concat_with_labels(self, example: tf.train.Example):
        inputs = []
        for modality in self._modalities_list:
            inputs.append(example[str(modality.name)])

        return tuple(inputs), example[DatasetFeaturesSet.CLASS.name]

    @tf.function
    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        example = tf.io.parse_single_example(serialized_example, self._feature_description)

        video_face_vgg_features = example[DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES.name]
        video_face_vgg_features = tf.io.parse_tensor(video_face_vgg_features, tf.float32)
        video_face_vgg_features = tf.cast(video_face_vgg_features, tf.float32)

        video_face_ir50_features = example[DatasetFeaturesSet.VIDEO_FACE_IR50_FEATURES.name]
        video_face_ir50_features = tf.io.parse_tensor(video_face_ir50_features, tf.float32)
        video_face_ir50_features = tf.cast(video_face_ir50_features, tf.float32)

        # video_face_r2plus1_features = example[DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES.name]
        # video_face_r2plus1_features = tf.io.parse_tensor(video_face_r2plus1_features, tf.float32)
        # video_face_r2plus1_features = tf.cast(video_face_r2plus1_features, tf.float32)

        video_scene_r2plus1_features = example[DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name]
        video_scene_r2plus1_features = tf.io.parse_tensor(video_scene_r2plus1_features, tf.float32)
        video_scene_r2plus1_features = tf.cast(video_scene_r2plus1_features, tf.float32)

        video_scene_iv3_features = example[DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name]
        video_scene_iv3_features = tf.io.parse_tensor(video_scene_iv3_features, tf.float32)
        video_scene_iv3_features = tf.cast(video_scene_iv3_features, tf.float32)

        audio = example[DatasetFeaturesSet.AUDIO.name]
        audio = tf.io.parse_tensor(audio, tf.float32)
        audio = tf.cast(audio, tf.float32)

        shimmers_shape = tf.io.parse_tensor(example[DatasetFeaturesSet.SHIMMERS_SHAPE.name], tf.int64)
        skeleton_shape = tf.io.parse_tensor(example[DatasetFeaturesSet.SKELETON_SHAPE.name], tf.int64)

        clazz = tf.io.parse_tensor(example[DatasetFeaturesSet.CLASS.name], tf.double)
        clazz = tf.cast(clazz, dtype=tf.float32)
        clazz = tf.ensure_shape(clazz, 7)

        return {
            DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES.name: video_face_vgg_features,
            DatasetFeaturesSet.VIDEO_FACE_IR50_FEATURES.name: video_face_ir50_features,
            # DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES.name: video_face_r2plus1_features,
            DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name: video_scene_r2plus1_features,
            DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name: video_scene_iv3_features,
            DatasetFeaturesSet.AUDIO.name: audio,
            DatasetFeaturesSet.SHIMMERS.name: example[DatasetFeaturesSet.SHIMMERS.name],  # raw
            DatasetFeaturesSet.SHIMMERS_SHAPE.name: shimmers_shape,
            DatasetFeaturesSet.SKELETON.name: example[DatasetFeaturesSet.SKELETON.name],  # raw
            DatasetFeaturesSet.SKELETON_SHAPE.name: skeleton_shape,
            DatasetFeaturesSet.CLASS.name: clazz
        }

    @tf.function
    def _extract_specified_modalities_and_ensure_shape(self, example: tf.Tensor) -> dict:
        tf_features_dict = {}
        for modality in self._modalities_list:
            data = example[str(modality.name)]
            if modality == DatasetFeaturesSet.SHIMMERS:
                shimmers = tf.io.parse_tensor(data, tf.float32)
                shimmers = tf.cast(shimmers, tf.float32)
                tf_features_dict[str(modality.name)] = tf.ensure_shape(shimmers, modality.config.shape)
            elif modality == DatasetFeaturesSet.SKELETON:
                skeleton = tf.io.parse_tensor(example[str(modality.name)], tf.float32)
                skeleton = tf.cast(skeleton, tf.float32)
                tf_features_dict[str(modality.name)] = tf.ensure_shape(skeleton, modality.config.shape)
            else:
                tf_features_dict[str(modality.name)] = tf.ensure_shape(data, modality.config.shape)

        if DatasetFeaturesSet.AUDIO.name in tf_features_dict:
            audio = tf_features_dict[DatasetFeaturesSet.AUDIO.name]
            # fixme (need more generic)
            tf_features_dict[DatasetFeaturesSet.AUDIO.name] = tf.reshape(audio, shape=(10, 393216))

        if DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name in tf_features_dict:
            video = tf_features_dict[DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name]
            # fixme (need more generic)
            tf_features_dict[DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name] = tf.reshape(video,
                                                                                            shape=(160, 2 * 2 * 2048))

        tf_features_dict[DatasetFeaturesSet.CLASS.name] = example[DatasetFeaturesSet.CLASS.name]
        return tf_features_dict
