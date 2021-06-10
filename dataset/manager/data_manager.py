import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeaturesSet
from utils.dirs import get_files_from_dir
from sklearn.model_selection import train_test_split


class DataManager:

    def __init__(self,
                 tf_record_path: str,
                 repeat: int = None,
                 batch_size: int = 1,
                 use_cache: bool = True,
                 use_prefetch: bool = True):
        self._tf_record_path = tf_record_path
        self._use_cache = use_cache
        self._batch_size = batch_size

        self._use_prefetch = use_prefetch
        self._repeat = repeat

        files = get_files_from_dir(tf_record_path)
        print("Dataset files: {}".format(files))
        self.PARALLEL_CALLS = tf.data.experimental.AUTOTUNE

        self.train_files, self.val_files = train_test_split(files, test_size=0.3)

        print("Train files size: {}".format(len(self.train_files)))
        print("Valid files size: {}".format(len(self.val_files)))

        self._val_ds = tf.data.TFRecordDataset(self.val_files)
        self._train_ds = tf.data.TFRecordDataset(self.train_files)

        self._feature_description = {
            DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.OPENSMILE_ComParE_2016.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.AUDIO.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SHIMMERS.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SHIMMERS_SHAPE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SKELETON.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.SKELETON_SHAPE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.CLASS.name: tf.io.FixedLenFeature([], tf.string)
        }

        self._val_ds = self.delete_unused_data(self._val_ds)
        self._train_ds = self.delete_unused_data(self._train_ds)

        if self._use_cache:
            self._val_ds = self._val_ds.cache()
            self._train_ds = self._train_ds.cache()

    def build_training_dataset(self, dataset_processor) -> tf.data.Dataset:
        return self._preprocess_dataset(dataset_processor, self._train_ds)

    def build_validation_dataset(self, dataset_processor) -> tf.data.Dataset:
        return self._preprocess_dataset(dataset_processor, self._val_ds)

    def _preprocess_dataset(self, dataset_processor, ds: tf.data.Dataset, ) -> tf.data.Dataset:
        ds = dataset_processor.pre_process(ds, self.PARALLEL_CALLS)

        ds = ds.batch(self._batch_size)
        #
        # if self._repeat:
        #     ds = ds.repeat(self._repeat)
        # else:
        #     ds = ds.repeat()

        if self._use_prefetch:
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def delete_unused_data(self, dataset):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    @tf.function
    def map_record_to_dictionary_of_tensors(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    @tf.function
    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        example = tf.io.parse_single_example(serialized_example, self._feature_description)

        video_face_vgg_features = example[DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES.name]
        video_face_vgg_features = tf.io.parse_tensor(video_face_vgg_features, tf.float32)
        video_face_vgg_features = tf.cast(video_face_vgg_features, tf.float32)

        video_face_r2plus1_features = example[DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES.name]
        video_face_r2plus1_features = tf.io.parse_tensor(video_face_r2plus1_features, tf.float32)
        video_face_r2plus1_features = tf.cast(video_face_r2plus1_features, tf.float32)

        video_scene_r2plus1_features = example[DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name]
        video_scene_r2plus1_features = tf.io.parse_tensor(video_scene_r2plus1_features, tf.float32)
        video_scene_r2plus1_features = tf.cast(video_scene_r2plus1_features, tf.float32)

        video_scene_iv3_features = example[DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name]
        video_scene_iv3_features = tf.io.parse_tensor(video_scene_iv3_features, tf.float32)
        video_scene_iv3_features = tf.cast(video_scene_iv3_features, tf.float32)

        audio = example[DatasetFeaturesSet.AUDIO.name]
        audio = tf.io.parse_tensor(audio, tf.float32)
        audio = tf.cast(audio, tf.float32)

        opemsmile = example[DatasetFeaturesSet.OPENSMILE_ComParE_2016.name]
        opemsmile = tf.io.parse_tensor(opemsmile, tf.float32)
        opemsmile = tf.cast(opemsmile, tf.float32)
        opemsmile = tf.squeeze(opemsmile)

        shimmers_shape = tf.io.parse_tensor(example[DatasetFeaturesSet.SHIMMERS_SHAPE.name], tf.int64)
        skeleton_shape = tf.io.parse_tensor(example[DatasetFeaturesSet.SKELETON_SHAPE.name], tf.int64)

        clazz = tf.io.parse_tensor(example[DatasetFeaturesSet.CLASS.name], tf.double)
        clazz = tf.cast(clazz, dtype=tf.float32)
        clazz = tf.ensure_shape(clazz, 7)

        return {
            DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES.name: video_face_vgg_features,
            DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES.name: video_face_r2plus1_features,
            DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name: video_scene_r2plus1_features,
            DatasetFeaturesSet.OPENSMILE_ComParE_2016.name: opemsmile,
            DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES.name: video_scene_iv3_features,
            DatasetFeaturesSet.AUDIO.name: audio,
            DatasetFeaturesSet.SHIMMERS.name: example[DatasetFeaturesSet.SHIMMERS.name],  # raw
            DatasetFeaturesSet.SHIMMERS_SHAPE.name: shimmers_shape,
            DatasetFeaturesSet.SKELETON.name: example[DatasetFeaturesSet.SKELETON.name],  # raw
            DatasetFeaturesSet.SKELETON_SHAPE.name: skeleton_shape,
            DatasetFeaturesSet.CLASS.name: clazz
        }
