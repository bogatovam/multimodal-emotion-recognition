import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from utils.dirs import get_files_from_dir


class DataManager:

    def __init__(self,
                 dataset_processor: BaseDatasetProcessor,
                 dataset_size: int,
                 tf_record_path: str,
                 repeat: int = None,
                 batch_size: int = 1,
                 use_cache: bool = True,
                 use_prefetch: bool = True):
        self._tf_record_path = tf_record_path
        self._dataset_processor = dataset_processor
        self._use_cache = use_cache
        self._batch_size = batch_size

        self._use_prefetch = use_prefetch
        self._repeat = repeat

        files = get_files_from_dir(tf_record_path)
        print("Dataset files: {}".format(files))

        self._ds_size = dataset_size

        self._ds = tf.data.TFRecordDataset(files)

        self.PARALLEL_CALLS = tf.data.experimental.AUTOTUNE

        print("Dataset size:={}".format(self._ds_size))
        self._val_size = int(0.20 * self._ds_size)
        self._test_size = int(0.10 * self._ds_size)
        self._train_size = int(0.7 * self._ds_size)

        self._ds = self._dataset_processor.pre_process(self._ds, self.PARALLEL_CALLS)
        self._ds = self._ds.shuffle(self._ds_size//6, reshuffle_each_iteration=True)

        self._train_ds = self._test_ds = self._val_ds = None

    def build_training_dataset(self) -> tf.data.Dataset:
        self._train_ds = self._ds.take(self._train_size)
        self._ds.skip(self._train_size)
        return self._preprocess_dataset(self._train_ds)

    def build_testing_dataset(self) -> tf.data.Dataset:
        self._test_ds = self._ds.take(self._test_size)
        self._ds.skip(self._test_size)
        return self._preprocess_dataset(self._test_ds)

    def build_validation_dataset(self) -> tf.data.Dataset:
        self._val_ds = self._ds.take(self._val_size)
        self._ds.skip(self._val_size)
        return self._preprocess_dataset(self._val_ds)

    def _preprocess_dataset(self, ds: tf.data.Dataset, ) -> tf.data.Dataset:
        # todo
        # ds = self._dataset_processor.pre_process(ds, self.PARALLEL_CALLS)
        if self._use_cache:
            ds = ds.cache()

        ds = ds.batch(self._batch_size)

        if self._repeat:
            ds = ds.repeat(self._repeat)
        else:
            ds = ds.repeat()

        if self._use_prefetch:
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds