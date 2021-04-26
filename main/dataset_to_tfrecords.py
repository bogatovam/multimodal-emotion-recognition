import math
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


def _open_video(filename: str) -> tuple:
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("[ERROR][TF RECORDS BUILDING][VIDEO FACE] It's impossible to open file '{}'".format(filename))
        cap.release()
        raise RuntimeError

    fps = cap.get(cv2.CAP_PROP_FPS)

    example_frames = []
    while cap.isOpened():
        read, frame = cap.read()
        if not read:
            break
        example_frames.append(frame)

    cap.release()
    return example_frames, fps


def _select_face(frame, face):
    # increase face window
    x = face[0] - 1 * face[0] // 10
    y = face[1] - 1 * face[1] // 10
    w = face[2] + 2 * face[2] // 10
    h = face[3] + 2 * face[3] // 10

    return frame[y:y + h, x:x + w]


def _process_face_on_frame(example: str, frame: np.ndarray, max_face_h, max_face_w):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 1:
        frame = _select_face(frame, faces[0])
        face_cords_hist[example] = [faces[0][0], faces[0][1], faces[0][2], faces[0][3]]
        return frame
    elif len(faces) > 1:
        face_size = 0
        nearest_face = None
        for face in faces:
            current_face_size = faces[0][2] * faces[0][3]
            if current_face_size > face_size:
                face_size = current_face_size
                nearest_face = face
        return _select_face(frame, nearest_face)
    elif face_cords_hist.get(example) and len(face_cords_hist.get(example)) != 0:
        return _select_face(frame, face_cords_hist.get(example))
    else:
        return np.zeros((max_face_h, max_face_w, 3))


def _process_audio_modality(filename: str, offset: float, duration: float) -> dict:
    # почистить данные от шума
    # попробовать выделить тлоько самый сильный голос - это все дальше, здесь только вытаскивание данных
    # как передавать дальше? - байты аудио, надо только нарезать по фреймам
    features_by_name = {}
    audio_raw, audio_raw_rate = lb.load(filename, offset=offset, duration=duration)
    features_by_name[DatasetExampleFeature.AUDIO_RAW] = audio_raw
    features_by_name[DatasetExampleFeature.AUDIO_RATE] = audio_raw_rate
    return features_by_name


def _process_video_scene_modality(frames: list, fps, example: str, offset: float, duration: float, modality) -> dict:
    offset_frames_count = math.floor(fps * offset)
    extracted_frames_count = math.floor(fps * duration)

    example_frames_list = frames[offset_frames_count:offset_frames_count + extracted_frames_count]

    example_total_frames = len(example_frames_list)

    frames_period = math.ceil(example_total_frames / modality.config.EXTRACTED_FRAMES_COUNT)
    period_res = modality.config.EXTRACTED_FRAMES_COUNT - (math.ceil(example_total_frames / frames_period))

    print("[INFO][{}][VIDEO SCENE] frames_period:={}, period_res:={}".format(example, frames_period, period_res))

    # to get N frames from the video we extract frames with period and then randomly add
    counter_1 = 0
    result_list = []
    for i in range(example_total_frames):
        if i % frames_period == 0:
            result_list.append(example_frames_list[i])
        elif (i - frames_period // 2) % frames_period == 0 and counter_1 < period_res:
            counter_1 += 1
            result_list.append(example_frames_list[i])

    result_list = np.asarray(result_list)
    features_by_name = {
        DatasetExampleFeature.VIDEO_SCENE_RAW: result_list,
        DatasetExampleFeature.VIDEO_SHAPE: result_list.shape,
    }

    print(
        "[INFO][{}][VIDEO SCENE] Complete video scene processing: frames_count:={}".format(os.path.basename(example),
                                                                                           len(result_list)))
    if len(result_list) != modality.config.EXTRACTED_FRAMES_COUNT:
        raise RuntimeError("len(result_list_resized) != modality.config.EXTRACTED_FRAMES_COUNT")
    return features_by_name


def _process_video_face_modality(frames: list, fps, example: str, offset: float, duration: float, modality) -> dict:
    offset_frames_count = math.floor(fps * offset)
    extracted_frames_count = math.floor(fps * duration)

    example_frames_list = frames[offset_frames_count:offset_frames_count + extracted_frames_count]

    example_total_frames = len(example_frames_list)

    frames_period = math.ceil(example_total_frames / modality.config.EXTRACTED_FRAMES_COUNT)
    period_res = modality.config.EXTRACTED_FRAMES_COUNT - (math.ceil(example_total_frames / frames_period))

    max_face_h = 1
    max_face_w = 1
    print("[INFO][{}][VIDEO FACE] frames_period:={}, period_res:={}".format(example, frames_period, period_res))
    # to get N frames from the video we extract frames with period and then randomly add
    counter_1 = 0
    result_list = []
    for i in range(example_total_frames):
        if i % frames_period == 0:
            frame = _process_face_on_frame(example, example_frames_list[i], max_face_h, max_face_w)
            result_list.append(frame)
            max_face_h = max(max_face_h, frame.shape[0])
            max_face_w = max(max_face_w, frame.shape[1])
        elif (i - frames_period // 2) % frames_period == 0 and counter_1 < period_res:
            counter_1 += 1
            frame = _process_face_on_frame(example, example_frames_list[i], max_face_h, max_face_w)
            result_list.append(frame)
            max_face_h = max(max_face_h, frame.shape[0])
            max_face_w = max(max_face_w, frame.shape[1])

    result_list_resized = []
    for frame in result_list:
        result_list_resized.append(cv2.resize(frame, (max_face_h, max_face_w)))

    features_by_name = {
        DatasetExampleFeature.VIDEO_FACE_RAW: np.asarray(result_list_resized)
    }

    if len(result_list_resized) != modality.config.EXTRACTED_FRAMES_COUNT:
        raise RuntimeError("len(result_list_resized) != modality.config.EXTRACTED_FRAMES_COUNT")

    print(
        "[INFO][{}][VIDEO FACE] Complete video face processing: frames_count:={}".format(os.path.basename(example),
                                                                                         len(result_list_resized)))
    return features_by_name


def _encode_example(features_by_name, clazz):
    tf_features_dict = {}
    for modality, feature in features_by_name.items():
        tf_features_dict[str(modality.name)] = modality.encoder.transform(feature)

    tf_features_dict[DatasetExampleFeature.CLASS.name] = DatasetExampleFeature.CLASS.encoder.transform(clazz)
    example = tf.train.Example(features=tf.train.Features(feature=tf_features_dict))
    return example.SerializeToString()


def _process_multimodal_dataset(name: str, modality_to_data: dict, output_folder: str, samples_per_tfrecord: int):
    tf_dataset_path = os.path.join(output_folder, name)
    create_dirs([output_folder, os.path.join(output_folder, name)])

    # Train/Test/Val split will be performed later, before learning
    with TfExampleWriter(name, tf_dataset_path, samples_per_tfrecord) as writer:
        j = 1
        for example, source_duration, clazz in config.ANNOTATION_BY_FILE:
            print("\n[INFO][{}] Example [{}/{}]: duration:={}"
                  .format(example, j, len(config.ANNOTATION_BY_FILE), source_duration))
            offset: float = 0
            source_duration = int(source_duration)
            frame_duration: float = min(TimeDependentModality.WINDOW_WIDTH_IN_SECS, source_duration)

            video_frames, fps = None, 0
            while offset + frame_duration <= source_duration:
                print("[INFO][{}] Frame position: offset:={},\tframe_duration:={}"
                      .format(example, offset, frame_duration))
                features_by_name = {}
                for modality, data_path in modality_to_data.items():
                    filename = os.path.join(data_path, example) + modality.config.FILE_EXT

                    if modality == Modality.AUDIO:
                        features_by_name.update(_process_audio_modality(filename, offset, frame_duration))

                    elif modality == Modality.VIDEO_SCENE:
                        if not video_frames:
                            video_frames, fps = _open_video(filename)
                        features_by_name.update(
                            _process_video_scene_modality(video_frames, fps, example, offset, frame_duration, modality))

                    elif modality == Modality.VIDEO_FACE:
                        if not video_frames:
                            video_frames, fps = _open_video(filename)
                        features_by_name.update(
                            _process_video_face_modality(video_frames, fps, example, offset, frame_duration, modality))

                writer.write(_encode_example(features_by_name, clazz))
                offset += TimeDependentModality.WINDOW_STEP_IN_SECS
            j += 1


if __name__ == "__main__":
    _process_multimodal_dataset(config.NAME, config.MODALITY_TO_DATA, config.DATASET_TF_RECORDS_PATH,
                                config.SAMPLES_PER_TFRECORD)
