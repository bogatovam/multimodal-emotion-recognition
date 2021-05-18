import os
import tensorflow as tf

_PATH_DELIMITER = "/"


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_files_from_dir(dir_name: str) -> list:
    try:
        files = tf.io.gfile.glob(dir_name + _PATH_DELIMITER + "*")
        files.sort()
        return files
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
