import os
import time

import tensorflow as tf
import logging as log


class TfExampleWriter:
    def __init__(self, dataset_name, base_filename: str, records_per_file: int):
        self._file_writer = tf.io.TFRecordWriter
        self._filename_format = base_filename + "\\" + dataset_name + "-{}.tfrecords"

        self._records_per_file = records_per_file
        self._example_counter = 0
        self._file_counter = 1
        self._start_file()

    def __enter__(self):
        return self

    def _start_file(self):
        self.start_file_time = time.process_time()

        self._filename = self._filename_format.format(self._file_counter)
        self._writer = self._file_writer(self._filename)

    def _finish_file(self):
        self.end_file_time = time.process_time()

        self._writer.flush()
        self._writer.close()
        print("Writing file: {}\tTime: {}s".format(self._filename, self.end_file_time - self.start_file_time))

    def _next_file(self):
        self._finish_file()
        self._file_counter += 1
        self._example_counter = 0
        self._start_file()

    def write(self, item):
        self._writer.write(item)

        self._example_counter += 1
        if not self._example_counter % self._records_per_file:
            self._next_file()

    def __exit__(self, *args):
        self._finish_file()
