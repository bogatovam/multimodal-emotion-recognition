import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from utils.dirs import get_files_from_dir


class DataManager:

    def __init__(self,
                 dataset_processor: BaseDatasetProcessor,
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

        ds_size = 0
        for fn in files:
            for _ in tf.compat.v1.io.tf_record_iterator(fn):
                ds_size += 1

        self._ds = tf.data.TFRecordDataset(files)

        val_size = int(0.15 * ds_size)
        test_size = int(0.15 * ds_size)
        train_size = int(0.7 * ds_size)

        self._ds = self._ds.shuffle(ds_size)

        self._val_ds = self._ds.take(val_size)
        self._ds.skip(val_size)

        self._test_ds = self._ds.take(test_size)
        self._ds.skip(test_size)

        self._train_ds = self._ds.take(train_size)
        self._ds.skip(train_size)

        self.PARALLEL_CALLS = tf.data.experimental.AUTOTUNE

    def build_training_dataset(self) -> tf.data.Dataset:
        return self._preprocess_dataset(self._train_ds)

    def build_testing_dataset(self) -> tf.data.Dataset:
        return self._preprocess_dataset(self._test_ds)

    def build_validation_dataset(self) -> tf.data.Dataset:
        return self._preprocess_dataset(self._val_ds)

    def _preprocess_dataset(self, ds: tf.data.Dataset, ) -> tf.data.Dataset:
        ds = self._dataset_processor.pre_process(ds, self.PARALLEL_CALLS)
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
