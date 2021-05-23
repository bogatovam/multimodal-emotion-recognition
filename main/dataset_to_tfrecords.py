import os
import numpy as np
import librosa as lb
import opensmile
import tensorflow as tf

import configs.by_device_type.cpu_config as config

from configs.dataset.modality import Modality, DatasetFeaturesSet, AudioModalityConfig
from dataset.tf_example_writer import TfExampleWriter
from main.preprocess.feature_extractors import extract_r2_plus1_features, get_ir50_face_model, get_vgg_face_model, \
    get_r2_plus_1_model, extract_vgg_features, extract_ir50_face_features, get_inception_v3_model, extract_iv3_features, \
    get_l3_model, extract_l3_features, process_shimmers_features, extract_pose_features, _extract_opensmile_features, \
    _l3_preprocess_audio, _preprocess_audio_batch
from main.preprocess.pose_feature_extraction import open_skeleton
from main.preprocess.shimmers_feature_extractors import open_shimmers
from main.preprocess.video_preprocessing import extract_frames_from_video_with_fps, open_video
from utils.dirs import create_dirs

with tf.device('/device:GPU:0'):
    vgg_face_model = get_vgg_face_model()
    audio_model = get_l3_model(config.PRETRAINED_MODELS)
    ir50_model = get_ir50_face_model(config.PRETRAINED_MODELS)
    r2_plus_1_model = get_r2_plus_1_model(config.PRETRAINED_MODELS)
    inception_v3_model = get_inception_v3_model(config.PRETRAINED_MODELS)

WINDOW_LEN_IN_SEC = 5
STEP_LEN_IN_SEC = 2


def _process_audio_modality(filename: str, offset: float, duration: float, modality) -> dict:
    features_by_name = {}
    audio_raw, audio_raw_rate = lb.load(filename, offset=offset, duration=duration, sr=48000)
    audio = _apply_window_to_signal(audio_raw, 48000, WINDOW_LEN_IN_SEC, STEP_LEN_IN_SEC)
    features_by_name[DatasetFeaturesSet.AUDIO] = audio
    print(f"[INFO][{filename}][Audio] Complete audio: shape:={features_by_name[DatasetFeaturesSet.AUDIO].shape}")
    return features_by_name


def _process_video_face_modality(frames: list, fps, example: str, offset: float, duration: float,
                                 modality) -> dict:
    print(f"[INFO][{example}] Preprocess video input")
    sparse_frames = extract_frames_from_video_with_fps(frames, fps, example, offset, duration, modality)
    sparse_frames = _apply_window_to_signal(sparse_frames, modality.config.FPS, WINDOW_LEN_IN_SEC, STEP_LEN_IN_SEC)
    return {
        DatasetFeaturesSet.VIDEO_FACE_RAW: sparse_frames
    }


def _process_video_scene_modality(frames: list, fps, example: str, offset: float, duration: float,
                                  modality) -> dict:
    print(f"[INFO][{example}] Preprocess video input")
    sparse_frames = extract_frames_from_video_with_fps(frames, fps, example, offset, duration, modality)
    sparse_frames = _apply_window_to_signal(sparse_frames, modality.config.FPS, WINDOW_LEN_IN_SEC, STEP_LEN_IN_SEC)
    return {
        DatasetFeaturesSet.VIDEO_SCENE_RAW: sparse_frames
    }


def _process_skeleton_modality(example: str, file: np.ndarray, examples_per_second: int, offset: float,
                               duration: float) -> dict:
    offset_examples = int(offset * examples_per_second)
    duration_examples = int(duration * examples_per_second)
    if file.shape[0] > offset_examples + duration_examples:
        skeleton = file[offset_examples:offset_examples + duration_examples]
    else:
        delta = offset_examples + duration_examples - file.shape[0]
        skeleton = file[offset_examples - delta:offset_examples + duration_examples - delta]

    print(f"[INFO][{example}] Preprocess skeleton input: examples_count:{skeleton.shape}")
    skeleton = _apply_window_to_signal(skeleton, examples_per_second, WINDOW_LEN_IN_SEC, STEP_LEN_IN_SEC)
    return {
        DatasetFeaturesSet.SKELETON: skeleton
    }


def _process_shimmers_modality(example: str, file: np.ndarray, examples_per_second: int, offset: float,
                               duration: float) -> dict:
    offset_examples = int(offset * examples_per_second)
    duration_examples = int(duration * examples_per_second)
    if file.shape[0] > offset_examples + duration_examples:
        shimmers = file[offset_examples:offset_examples + duration_examples]
    else:
        delta = offset_examples + duration_examples - file.shape[0]
        shimmers = file[offset_examples - delta:offset_examples + duration_examples - delta]

    print(f"[INFO][{example}] Preprocess shimmers input: examples_count:{shimmers.shape}")
    shimmers = _apply_window_to_signal(shimmers, examples_per_second, WINDOW_LEN_IN_SEC, STEP_LEN_IN_SEC)
    return {
        DatasetFeaturesSet.SHIMMERS: shimmers
    }


def _apply_window_to_signal(signal, elements_per_sec, window_len_sec, step_len_sec) -> np.ndarray:
    min_allowed_elements = window_len_sec * elements_per_sec

    if signal.shape[0] < min_allowed_elements:
        pad = np.zeros((len(signal.shape), 2), dtype=int)
        pad[0][1] = min_allowed_elements - signal.shape[0]
        signal = np.pad(signal.astype(np.float32), pad, 'constant')
    return tf.signal.frame(signal, elements_per_sec * window_len_sec, elements_per_sec * step_len_sec, axis=0).numpy()


def _extract_features_from_data(filename, data_from_window) -> dict:
    with tf.device('/device:GPU:0'):
        features_by_name = {}

        # Face
        print(f'[{filename}] Extracting features ...')
        # face_features_r2 = extract_r2_plus1_features(r2_plus_1_model,
        #                                              data_from_window[DatasetFeaturesSet.VIDEO_FACE_RAW])
        # print(f'[{filename}] Extracting r2_plus1_features features: shape:={face_features_r2.shape}')

        face_features_vgg = extract_vgg_features(vgg_face_model, data_from_window[DatasetFeaturesSet.VIDEO_FACE_RAW])
        print(f'[{filename}] Extracting vgg_features features: shape:={face_features_vgg.shape}')

        face_features_ir50 = extract_ir50_face_features(ir50_model, data_from_window[DatasetFeaturesSet.VIDEO_FACE_RAW])
        print(f'[{filename}] Extracting ir50_face_features features: shape:={face_features_ir50.shape}')

        features_by_name[DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES] = face_features_vgg
        features_by_name[DatasetFeaturesSet.VIDEO_FACE_IR50_FEATURES] = face_features_ir50
        # features_by_name[DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES] = face_features_r2

        # Scene
        scene_features_r2 = extract_r2_plus1_features(r2_plus_1_model,
                                                      data_from_window[DatasetFeaturesSet.VIDEO_SCENE_RAW])
        print(f'[{filename}] Extracting r2_plus1_features features: shape:={scene_features_r2.shape}')
        scene_features_iv3 = extract_iv3_features(inception_v3_model,
                                                  data_from_window[DatasetFeaturesSet.VIDEO_SCENE_RAW])
        print(f'[{filename}] Extracting iv3_features features: shape:={scene_features_iv3.shape}')

        features_by_name[DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES] = scene_features_r2
        features_by_name[DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES] = scene_features_iv3

        # Audio
        preprocessed_audio = _preprocess_audio_batch(data_from_window[DatasetFeaturesSet.AUDIO], 48000)
        audio_features = extract_l3_features(audio_model, preprocessed_audio)
        print(f'[{filename}] Extracting l3_features features: shape:={audio_features.shape}')
        features_by_name[DatasetFeaturesSet.AUDIO] = audio_features

        # audio_features = _extract_opensmile_features(preprocessed_audio, 48000, opensmile.FeatureSet.GeMAPSv01b)
        # features_by_name[DatasetFeaturesSet.OPENSMILE_GeMAPSv01b] = audio_features.astype(np.float32)
        # print(f'[{filename}] Extracting opensmile.FeatureSet.GeMAPSv01b features: shape:={audio_features.shape}')
        #
        # audio_features = _extract_opensmile_features(preprocessed_audio, 48000, opensmile.FeatureSet.eGeMAPSv02)
        # features_by_name[DatasetFeaturesSet.OPENSMILE_eGeMAPSv02] = audio_features.astype(np.float32)
        # print(f'[{filename}] Extracting opensmile.FeatureSet.eGeMAPSv02 features: shape:={audio_features.shape}')

        audio_features = _extract_opensmile_features(preprocessed_audio, 48000, opensmile.FeatureSet.ComParE_2016)
        features_by_name[DatasetFeaturesSet.OPENSMILE_ComParE_2016] = audio_features.astype(np.float32)
        print(f'[{filename}] Extracting opensmile.FeatureSet.ComParE_2016 features: shape:={audio_features.shape}')

        # Shimmers
        if DatasetFeaturesSet.SHIMMERS in data_from_window:
            shimmers_features = process_shimmers_features(data_from_window[DatasetFeaturesSet.SHIMMERS])
            print(f'[{filename}] Extracting shimmers_features features: shape:={shimmers_features.shape}')
            features_by_name[DatasetFeaturesSet.SHIMMERS] = np.nan_to_num(shimmers_features).astype(np.float32)

            # Pose
        if DatasetFeaturesSet.SKELETON in data_from_window:
            pose_features = extract_pose_features(data_from_window[DatasetFeaturesSet.SKELETON])
            print(f'[{filename}] Extracting pose_features features: shape:={pose_features.shape}')
            features_by_name[DatasetFeaturesSet.SKELETON] = np.nan_to_num(pose_features).astype(np.float32)

    return features_by_name


def _encode_example(features_by_name, clazz):
    tf_features_dict = {}
    for modality, feature in features_by_name.items():
        tf_features_dict[str(modality.name)] = modality.encoder.transform(feature)

    tf_features_dict[DatasetFeaturesSet.CLASS.name] = DatasetFeaturesSet.CLASS.encoder.transform(clazz)
    fake_data = np.asarray([0], dtype=np.float32)
    fake = np.asarray([0], dtype=np.int64)
    if DatasetFeaturesSet.SHIMMERS.name not in tf_features_dict:
        tf_features_dict[DatasetFeaturesSet.SHIMMERS.name] = DatasetFeaturesSet.SHIMMERS.encoder.transform(fake_data)
        tf_features_dict[DatasetFeaturesSet.SHIMMERS_SHAPE.name] = DatasetFeaturesSet.SHIMMERS.encoder.transform(fake)
    else:
        shape = np.asarray(features_by_name[DatasetFeaturesSet.SHIMMERS].shape).astype(np.int64)
        tf_features_dict[DatasetFeaturesSet.SHIMMERS_SHAPE.name] = DatasetFeaturesSet.SHIMMERS.encoder.transform(shape)

    if DatasetFeaturesSet.SKELETON.name not in tf_features_dict:
        tf_features_dict[DatasetFeaturesSet.SKELETON.name] = DatasetFeaturesSet.SKELETON.encoder.transform(fake_data)
        tf_features_dict[DatasetFeaturesSet.SKELETON_SHAPE.name] = DatasetFeaturesSet.SKELETON.encoder.transform(fake)
    else:
        shape = np.asarray(features_by_name[DatasetFeaturesSet.SKELETON].shape).astype(np.int64)
        tf_features_dict[DatasetFeaturesSet.SKELETON_SHAPE.name] = DatasetFeaturesSet.SKELETON.encoder.transform(shape)

    example = tf.train.Example(features=tf.train.Features(feature=tf_features_dict))
    return example.SerializeToString()


def _process_multimodal_dataset(name: str, modality_to_data: dict, output_folder: str, samples_per_tfrecord: int):
    tf_dataset_path = output_folder + "/" + name
    create_dirs([output_folder, tf_dataset_path])

    # Train/Test/Val split will be performed later, before learning
    with TfExampleWriter(name, tf_dataset_path, samples_per_tfrecord, counter=32) as writer:
        j = 1
        for file_name, annotations_list in config.ANNOTATIONS:
            scene_video_frames, scene_video_fps = None, 0
            face_video_frames, face_video_fps = None, 0
            shimmers_file, skeleton_file = None, None
            for emotion in annotations_list:

                print("\n[INFO][{}] Example [{}/{}]: duration:={}, classes:={}"
                      .format(file_name, j, config.TARGET_DATASET_LENGTH, emotion.duration,
                              emotion.emotions_vector))

                offset = emotion.offset
                duration = emotion.duration

                if duration < 5.0:
                    continue
                data_by_window = {}
                for modality, data_path in modality_to_data.items():
                    filename = os.path.join(data_path, file_name) + modality.config.FILE_EXT
                    if modality == Modality.AUDIO:
                        data_by_window.update(_process_audio_modality(filename, offset, emotion.duration, modality))
                    elif modality == Modality.VIDEO_SCENE:
                        if scene_video_frames is None:
                            scene_video_frames, scene_video_fps = open_video(filename)
                        data_by_window.update(_process_video_scene_modality(scene_video_frames, scene_video_fps,
                                                                            file_name, offset,
                                                                            duration, modality))
                    elif modality == Modality.VIDEO_FACE:
                        if face_video_frames is None:
                            face_video_frames, face_video_fps = open_video(filename)
                        data_by_window.update(_process_video_face_modality(face_video_frames, face_video_fps,
                                                                           file_name, offset,
                                                                           duration, modality))
                    elif modality == Modality.KINECT_SKELETON:
                        if skeleton_file is None:
                            skeleton_file = open_skeleton(filename, modality.config.EXAMPLES_PER_SECOND)
                        if skeleton_file.shape[0] == 0:
                            continue
                        data_by_window.update(_process_skeleton_modality(filename, skeleton_file,
                                                                         modality.config.EXAMPLES_PER_SECOND,
                                                                         offset, duration))
                    elif modality == Modality.SHIMMERS:
                        if shimmers_file is None:
                            shimmers_file = open_shimmers(filename, modality.config.EXAMPLES_PER_SECOND)
                        if shimmers_file.shape[0] == 0:
                            continue
                        data_by_window.update(_process_shimmers_modality(filename, shimmers_file,
                                                                         modality.config.EXAMPLES_PER_SECOND,
                                                                         offset, duration))

                windows_of_data = [dict(zip(data_by_window, t)) for t in zip(*data_by_window.values())]
                for i, data_from_window in enumerate(windows_of_data):
                    print(f'[{filename}] Saving window position... #{i}')
                    writer.write(_encode_example(_extract_features_from_data(filename, data_from_window),
                                                 emotion.emotions_vector))
                j += 1


if __name__ == "__main__":
    _process_multimodal_dataset(config.NAME, config.MODALITY_TO_DATA, config.DATASET_TF_RECORDS_PATH,
                                config.SAMPLES_PER_TFRECORD)
