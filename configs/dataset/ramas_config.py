from configs.dataset.modality import Modality

DATASET_NAME = "RAMAS"
DATASET_PATH = "E:/RAMAS/RAMAS"
DATASET_TF_RECORDS_PATH = "D:/RAMAS/tf-records"

MODALITY_TO_DATA: dict = {
    Modality.AUDIO: DATASET_PATH + "/Data/Audio",
    Modality.SHIMMERS: DATASET_PATH + "/Data/Shimmers",
    Modality.VIDEO_SCENE: DATASET_PATH + "/Data/Video_web",
    Modality.VIDEO_FACE: DATASET_PATH + "/Data/Video_close",
    Modality.KINECT_SKELETON: DATASET_PATH + "/Data/Kinect_skeleton"
}

ANNOTATIONS_BY_FILES = DATASET_PATH + "/Annotations_by_files"
ANNOTATIONS_BY_EMOTIONS = DATASET_PATH + "/Annotations_by_emotions"
