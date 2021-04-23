import os
import cv2
import numpy as np
import librosa as lb
import tensorflow as tf

import configs.deception_default_config as config

from configs.dataset.modality import Modality, DatasetExampleFeature, TimeDependentModality
from dataset.tf_example_writer import TfExampleWriter
from utils.dirs import create_dirs

face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
face_cords_hist = {}


def _process_audio_modality(filename: str, offset: float, duration: float) -> dict:
    # почистить данные от шума
    # попробовать выделить тлоько самый сильный голос - это все дальше, здесь только вытаскивание данных
    # как передавать дальше? - байты аудио, надо только нарезать по фреймам
    features_by_name = {}
    audio_raw, audio_raw_rate = lb.load(filename, offset=offset, duration=duration)
    features_by_name[DatasetExampleFeature.AUDIO_RAW] = audio_raw
    features_by_name[DatasetExampleFeature.AUDIO_RATE] = audio_raw_rate
    return features_by_name


def _process_video_scene_modality(filename: str, offset: float, duration: float) -> dict:
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("[ERROR][TF RECORDS BUILDING][VIDEO SCENE] It's impossible to open file '{}'".format(filename))
        cap.release()
        return {}

    example_frames_list = []
    offset_frames_count = int(cap.get(cv2.CAP_PROP_FPS) * offset)
    extracted_frames_count = int(cap.get(cv2.CAP_PROP_FPS) * duration)

    captured_frames_count = 0
    while cap.isOpened() and captured_frames_count - offset_frames_count <= extracted_frames_count:
        read, frame = cap.read()
        if not read:
            continue
        if 0 <= captured_frames_count - offset_frames_count <= extracted_frames_count:
            example_frames_list.append(frame)
        captured_frames_count += 1

    features_by_name = {
        DatasetExampleFeature.VIDEO_SCENE_RAW: np.asarray(example_frames_list),
        DatasetExampleFeature.VIDEO_SHAPE: np.asarray(
            [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 3]),
        DatasetExampleFeature.VIDEO_FRAMES: extracted_frames_count
    }

    cap.release()
    return features_by_name


def _select_face(frame, face):
    # increase face window
    x = face[0] - 1 * face[0] // 10
    y = face[1] - 1 * face[1] // 10
    w = face[2] + 2 * face[2] // 10
    h = face[3] + 2 * face[3] // 10

    return frame[y:y + h, x:x + w]


def _process_video_face_modality(filename: str, offset: float, duration: float) -> dict:
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("[ERROR][TF RECORDS BUILDING][VIDEO FACE] It's impossible to open file '{}'".format(filename))
        cap.release()
        return {}

    example_frames_list = []
    offset_frames_count = cap.get(cv2.CAP_PROP_FPS) * offset
    extracted_frames_count = cap.get(cv2.CAP_PROP_FPS) * duration

    captured_frames_count = 0
    max_face_h = 1
    max_face_w = 1
    while cap.isOpened() and captured_frames_count - offset_frames_count <= extracted_frames_count:
        read, frame = cap.read()
        if not read:
            continue
        if 0 <= captured_frames_count - offset_frames_count <= extracted_frames_count:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 1:
                frame = _select_face(frame, faces[0])
                face_cords_hist[filename] = [faces[0][0], faces[0][1], faces[0][2], faces[0][3]]
            elif len(faces) > 1:
                face_size = 0
                nearest_face = None
                for face in faces:
                    current_face_size = faces[0][2] * faces[0][3]
                    if current_face_size > face_size:
                        face_size = current_face_size
                        nearest_face = face
                frame = _select_face(frame, nearest_face)
            elif face_cords_hist.get(filename) and len(face_cords_hist.get(filename)) != 0:
                frame = _select_face(frame, face_cords_hist.get(filename))
            else:
                frame = np.zeros((max_face_h, max_face_w, 3))

            max_face_h = max(max_face_h, frame.shape[0])
            max_face_w = max(max_face_w, frame.shape[1])

            example_frames_list.append(frame)
        captured_frames_count += 1

    example_frames_list_resized = []
    for frame in example_frames_list:
        example_frames_list_resized.append(cv2.resize(frame, (max_face_h, max_face_w)))

    features_by_name = {
        DatasetExampleFeature.VIDEO_FACE_RAW: np.asarray(example_frames_list_resized)
    }

    cap.release()
    return features_by_name


def _encode_example(features_by_name, clazz):
    tf_features_dict = {}
    for modality, feature in features_by_name.items():
        tf_features_dict[str(modality.name)] = modality.encoder.transform(feature)

    tf_features_dict[DatasetExampleFeature.CLASS.name] = DatasetExampleFeature.CLASS.encoder.transform(clazz)
    example = tf.train.Example(features=tf.train.Features(feature=tf_features_dict))
    return example.SerializeToString()


def _process_multimodal_dataset(name: str, modality_to_data: dict, output_folder: str, samples_per_tfrecord: int):
    create_dirs([output_folder, os.path.join(output_folder, name)])

    # Train/Test/Val split will be performed later, before learning
    with TfExampleWriter(name, "", samples_per_tfrecord) as writer:
        j = 1
        for example_name, source_duration, clazz in config.ANNOTATION_BY_FILE:
            print("[INFO][{}] Example [{}/{}]: duration:={}"
                  .format(example_name, j, len(config.ANNOTATION_BY_FILE), source_duration))
            offset: float = 0
            frame_duration: float = TimeDependentModality.WINDOW_WIDTH_IN_SECS
            while offset + frame_duration < source_duration:
                print("[INFO][{}] Frame position: offset:={},\tframe_duration:={}"
                      .format(example_name, offset, frame_duration))
                features_by_name = {}
                for modality, data_path in modality_to_data.items():
                    filename = os.path.join(data_path, example_name) + modality.config.FILE_EXT
                    if modality == Modality.AUDIO:
                        features_by_name.update(_process_audio_modality(filename, offset, frame_duration))
                    elif modality == Modality.VIDEO_SCENE:
                        features_by_name.update(_process_video_scene_modality(filename, offset, frame_duration))
                    elif modality == Modality.VIDEO_FACE:
                        features_by_name.update(_process_video_face_modality(filename, offset, frame_duration))

                writer.write(_encode_example(features_by_name, clazz))
                offset += TimeDependentModality.WINDOW_STEP_IN_SECS
            j += 1


if __name__ == "__main__":
    _process_multimodal_dataset(config.NAME, config.MODALITY_TO_DATA, config.DATASET_TF_RECORDS_PATH,
                                config.SAMPLES_PER_TFRECORD)
