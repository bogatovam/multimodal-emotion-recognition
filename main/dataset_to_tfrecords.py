import gc
import math
import os
from numbers import Real

import cv2
import numpy as np
import librosa as lb
import opensmile
import resampy
import tensorflow as tf

import configs.ramas_default_config as config

from configs.dataset.modality import Modality, DatasetFeature, TimeDependentModality
from dataset.tf_example_writer import TfExampleWriter
from utils.dirs import create_dirs

face_cascade = cv2.CascadeClassifier('../models/pretrained/haarcascade_frontalface_default.xml')
face_cords_hist = {}

def _open_video(filename: str) -> tuple:
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("[ERROR][TF RECORDS BUILDING][VIDEO FACE] It's impossible to open file '{}'".format(filename))
        cap.release()

        filename = os.path.splitext(filename)[0] + ".avi"
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            cap.release()
            print("[ERROR][TF RECORDS BUILDING][VIDEO FACE] It's impossible to open file '{}'".format(filename))
            filename = os.path.splitext(filename)[0] + ".mov"
            cap = cv2.VideoCapture(filename)
            if not cap.isOpened():
                cap.release()
                print("[ERROR][TF RECORDS BUILDING][VIDEO FACE] It's impossible to open file '{}'".format(filename))

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
    x = face[0] - 1 * face[0] // 5
    y = face[1] - 1 * face[1] // 5
    w = face[2] + 2 * face[2] // 5
    h = face[3] + 2 * face[3] // 5

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


def _extract_opensmile_features(audio, sr, features_set):
    smile = opensmile.Smile(
        feature_set=features_set,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    return smile.process_signal(audio, sr)


def _center_audio(audio, frame_len):
    """Center audio so that first sample will occur in the middle of the first frame"""
    return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)


def _pad_audio(audio, frame_len, hop_len):
    """Pad audio if necessary so that all samples are processed"""
    audio_len = audio.size
    if audio_len < frame_len:
        pad_length = frame_len - audio_len
    else:
        pad_length = int(np.ceil((audio_len - frame_len) / float(hop_len))) * hop_len \
                     - (audio_len - frame_len)

    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    return audio


def _l3_preprocess_audio(audio, sr, target_sr: int = 48000, center=True, hop_size=0.1):
    if audio.size == 0:
        raise RuntimeError('Got empty audio')

    # Warn user if audio is all zero
    if np.all(audio == 0):
        print('[WARN] Provided audio is all zeros')

    # Check audio array dimension
    if audio.ndim > 2:
        raise RuntimeError('Audio array can only be be 1D or 2D')

    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    if not isinstance(sr, Real) or sr <= 0:
        raise RuntimeError('Invalid sample rate {}'.format(sr))

    # Resample if necessary
    if sr != target_sr:
        audio = resampy.resample(audio, sr_orig=sr, sr_new=target_sr, filter='kaiser_best')

    audio_len = audio.size
    frame_len = target_sr
    hop_len = int(hop_size * target_sr)

    if audio_len < frame_len:
        print('[WARN] Duration of provided audio is shorter than window size (1 second). Audio will be padded.')

    if center:
        # Center audio
        audio = _center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = _pad_audio(audio, frame_len, hop_len)

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
                                        strides=(audio.itemsize, hop_len * audio.itemsize)).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))
    return x


def _process_audio_modality(filename: str, offset: float, duration: float) -> dict:
    features_by_name = {}
    audio_raw, audio_raw_rate = lb.load(filename, offset=offset, duration=duration)
    features_by_name[DatasetFeature.L3] = _l3_preprocess_audio(audio_raw, audio_raw_rate)
    print("[INFO][{}][Audio] Complete audio: shape:={}".format(os.path.basename(filename),
                                                               features_by_name[DatasetFeature.L3].shape))
    return features_by_name


def _process_video_scene_modality(frames: list, fps, example: str, offset: float, duration: float, modality) -> dict:
    offset_frames_count = math.floor(fps * offset)
    extracted_frames_count = math.floor(fps * duration)

    delta = math.floor(fps / 2)
    offset_frames_count = math.floor(max(0, offset_frames_count - delta))
    extracted_frames_count = math.floor(min(len(frames) - 1, extracted_frames_count + fps))

    example_frames_list = frames[offset_frames_count:offset_frames_count + extracted_frames_count]

    example_total_frames = len(example_frames_list)

    frames_period = modality.config.FRAMES_PERIOD

    # to get N frames from the video we extract frames with period and then randomly add
    result_list = []
    for i in range(example_total_frames):
        if i % frames_period == 0:
            result_list.append(example_frames_list[i])

    result_list_resized = []
    for frame in result_list:
        result_list_resized.append(cv2.resize(frame, (modality.config.SHAPE, modality.config.SHAPE)))

    features_by_name = {
        DatasetFeature.VIDEO_SCENE_RAW: np.asarray(result_list_resized)
    }

    print(
        "[INFO][{}][VIDEO SCENE] Complete video scene processing: frames_count:={}".format(os.path.basename(example),
                                                                                           len(result_list)))
    return features_by_name


def _process_video_face_modality(frames: list, fps, example: str, offset: float, duration: float, modality) -> dict:
    offset_frames_count = math.floor(fps * offset)
    extracted_frames_count = math.floor(fps * duration)

    # add some frames from neighborhood
    delta = math.floor(fps / 2)
    offset_frames_count = math.floor(max(0, offset_frames_count - delta))
    extracted_frames_count = math.floor(min(len(frames) - 1, extracted_frames_count + fps))

    example_frames_list = frames[offset_frames_count:offset_frames_count + extracted_frames_count]

    example_total_frames = len(example_frames_list)

    frames_period = modality.config.FRAMES_PERIOD

    print("[INFO][{}][VIDEO FACE] frames_period:={}".format(example, frames_period))
    # to get N frames from the video we extract frames with period and then randomly add

    max_face_h = 1
    max_face_w = 1
    result_list = []
    for i in range(example_total_frames):
        if i % frames_period == 0:
            frame = _process_face_on_frame(example, example_frames_list[i], max_face_h, max_face_w)
            result_list.append(frame)
            max_face_h = max(max_face_h, frame.shape[0])
            max_face_w = max(max_face_w, frame.shape[1])

    result_list_resized = []
    for frame in result_list:
        result_list_resized.append(cv2.resize(frame, (modality.config.SHAPE, modality.config.SHAPE)))

    features_by_name = {
        DatasetFeature.VIDEO_FACE_RAW: np.asarray(result_list_resized)
    }

    print(
        "[INFO][{}][VIDEO FACE] Complete video face processing: frames_count:={}".format(os.path.basename(example),
                                                                                         len(result_list_resized)))
    return features_by_name


def _encode_example(features_by_name, clazz):
    tf_features_dict = {}
    for modality, feature in features_by_name.items():
        tf_features_dict[str(modality.name)] = modality.encoder.transform(feature)

    tf_features_dict[DatasetFeature.CLASS.name] = DatasetFeature.CLASS.encoder.transform(clazz)
    example = tf.train.Example(features=tf.train.Features(feature=tf_features_dict))
    return example.SerializeToString()


def _process_multimodal_dataset(name: str, modality_to_data: dict, output_folder: str, samples_per_tfrecord: int):
    tf_dataset_path = os.path.join(output_folder, name)
    create_dirs([output_folder, os.path.join(output_folder, name)])

    # Train/Test/Val split will be performed later, before learning
    with TfExampleWriter(name, tf_dataset_path, samples_per_tfrecord) as writer:
        j = 1
        for file_name, annotations_list in config.ANNOTATIONS:
            for emotion in annotations_list:

                print("\n[INFO][{}] Example [{}/{}]: duration:={}, classes:={}"
                      .format(file_name, j, config.TARGET_DATASET_LENGTH, emotion.duration, emotion.emotions_vector))

                offset = emotion.offset
                duration = emotion.duration

                features_by_name = {}
                for modality, data_path in modality_to_data.items():
                    filename = os.path.join(data_path, file_name) + modality.config.FILE_EXT
                    if modality == Modality.AUDIO:
                        features_by_name.update(
                            _process_audio_modality(filename, offset, emotion.duration))
                    elif modality == Modality.VIDEO_SCENE:
                        scene_video_frames, scene_video_fps = _open_video(filename)
                        features_by_name.update(
                            _process_video_scene_modality(scene_video_frames, scene_video_fps, file_name, offset,
                                                          duration, modality))
                    elif modality == Modality.VIDEO_FACE:
                        face_video_frames, face_video_fps = _open_video(filename)
                        features_by_name.update(
                            _process_video_face_modality(face_video_frames, face_video_fps, file_name, offset,
                                                         duration, modality))
                    gc.collect()
                writer.write(_encode_example(features_by_name, emotion.emotions_vector))
                j += 1


if __name__ == "__main__":
    _process_multimodal_dataset(config.NAME, config.MODALITY_TO_DATA, config.DATASET_TF_RECORDS_PATH,
                                config.SAMPLES_PER_TFRECORD)
