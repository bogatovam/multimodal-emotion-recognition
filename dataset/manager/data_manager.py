import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
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
