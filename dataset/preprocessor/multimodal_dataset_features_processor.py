import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeaturesSet


class MultimodalDatasetFeaturesProcessor(BaseDatasetProcessor):

    def __init__(self, modalities_list):
        self._modalities_list = modalities_list

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
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

        if DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name in tf_features_dict:
            video = tf_features_dict[DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name]
            # fixme (need more generic)
            tf_features_dict[DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name] = tf.reshape(video,
                                                                                            shape=(160, 2 * 2 * 2048))

        tf_features_dict[DatasetFeaturesSet.CLASS.name] = example[DatasetFeaturesSet.CLASS.name]
        return tf_features_dict
