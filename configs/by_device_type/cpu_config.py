import csv
import os
import numpy as np
from random import shuffle
from configs.common_constants import *

from configs.dataset.modality import VideoModalityConfig, Modality
from configs.by_device_type.cpu_dir_config import *

PRETRAINED_MODEL_PATH = PRETRAINED_MODELS + "/rplus1"

ANNOTATIONS_FILE = "D:/2021/hse/course-work" + "/annotationsClass.csv"

MODALITY_TO_DATA: dict = {
    Modality.AUDIO: DATASET_PATH + "/Data/Audio",
    Modality.SHIMMERS: DATASET_PATH + "/Data/Shimmers",
    Modality.VIDEO_SCENE: DATASET_PATH + "/Data/Video_web",
    Modality.VIDEO_FACE: DATASET_PATH + "/Data/Video_close",
    Modality.KINECT_SKELETON: DATASET_PATH + "/Data/Kinect_skeleton"
}


class EmotionAnnotation:
    def __init__(self, emotions_vector, time_interval):
        self.offset = time_interval[0]
        self.emotions_vector = emotions_vector
        self.duration = time_interval[1] - time_interval[0]


def read_annotations(path):
    file_to_annotation = {}
    target_dataset_length = 0
    with open(path, newline='') as annotations:
        reader = csv.DictReader(annotations)
        for row in reader:
            file_name = row['file']
            if file_name not in file_to_annotation:
                file_to_annotation[file_name] = []

            emotion_interval = (float(row['start']), float(row['end']))
            emotions_vector = np.zeros(len(classes), dtype=np.float)
            for i, clazz in enumerate(classes):
                emotions_vector[i] = float(row[clazz])

            file_to_annotation[file_name].append(EmotionAnnotation(emotions_vector, emotion_interval))
            target_dataset_length += 1

    file_to_annotation_list = list(file_to_annotation.items())
    filenames = list(map(lambda x: x[0], file_to_annotation_list))

    start_index = filenames.index('15dec_K11_2')

    res = []
    target_dataset_length = 0
    for i, file_to_emotions in enumerate(file_to_annotation_list):
        if i < start_index:
            continue
        res.append(file_to_emotions)
        target_dataset_length += len(file_to_emotions[1])
    # shuffle(file_to_annotation_list)
    return res, target_dataset_length


# ANNOTATIONS, TARGET_DATASET_LENGTH = read_annotations(ANNOTATIONS_FILE)
