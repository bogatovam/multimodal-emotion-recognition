import csv
import os
import librosa
from random import shuffle

import numpy as np
from configs.dataset.modality import Modality

DATASET_NAME = "deception_dataset"
DATASET_PATH = "D:/2021/hse/lie-detector/deception_dataset"
DATASET_TF_RECORDS_PATH = "E:/2021/hse/lie-detector/deception_dataset/tf-records"

MODALITY_TO_DATA: dict = {
    Modality.AUDIO: DATASET_PATH + "/Audio",
    Modality.VIDEO_FACE: DATASET_PATH + "/Clips",
    Modality.VIDEO_SCENE: DATASET_PATH + "/Clips"
}

ANNOTATIONS_FILE = DATASET_PATH + "/Annotation/annotations.csv"


def _get_file_name_to_label_from_annotations():
    file_name_to_label = []
    with open(ANNOTATIONS_FILE, newline='') as annotations:
        reader = csv.DictReader(annotations)
        for row in reader:
            class_vector = np.array([row['class'] == 'deceptive'], dtype=int)

            filename = os.path.splitext(row['id'])[0]
            file_to_extract_duration = os.path.join(MODALITY_TO_DATA[Modality.AUDIO],
                                                    filename + Modality.AUDIO.config.FILE_EXT)
            duration = librosa.get_duration(filename=file_to_extract_duration)
            file_name_to_label.append((filename, duration, class_vector))
    shuffle(file_name_to_label)
    return file_name_to_label


ANNOTATION_BY_FILE = _get_file_name_to_label_from_annotations()
