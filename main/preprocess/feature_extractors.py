from numbers import Real

import cv2
import numpy as np
import tensorflow as tf
from kapre.time_frequency import Melspectrogram
from keras import Model
from keras_vggface import VGGFace
from sklearn import preprocessing

from configs.dataset.modality import AudioModalityConfig
from main.preprocess.pose_feature_extraction import compute_features


def get_r2_plus_1_model(base_path):
    return tf.keras.models.load_model(base_path + "/" + "rplus1", compile=False).signatures["serving_default"]


def get_vgg_face_model():
    return VGGFace(include_top=False, weights='vggface', input_shape=(224, 224, 3), pooling='avg')


def get_ir50_face_model(base_path):
    new_model = tf.keras.models.load_model(base_path + "/ir50_ms1m_keras.h5")
    intermediate_layer_model = Model(inputs=new_model.input,
                                     outputs=new_model.get_layer('output_layer.4').output)
    return intermediate_layer_model


def get_inception_v3_model(base_path):
    return tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(112, 112, 3))


def get_l3_model(base_path):
    model = tf.keras.models.load_model(base_path + "/openl3_audio_mel256_music.h5", custom_objects={
        'Melspectrogram': Melspectrogram}, compile=False)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('activation_7').output)

    return intermediate_layer_model


def extract_ir50_face_features(model, frames) -> np.ndarray:
    model_input = np.array(list(map(resize_preprocess, frames)))
    return model.predict(model_input)


def extract_iv3_features(model, frames) -> np.ndarray:
    model_input = np.array(list(map(resize_preprocess, frames)))
    return model.predict(model_input)


def extract_l3_features(model, audio) -> np.ndarray:
    model_input = _l3_preprocess_audio(audio, AudioModalityConfig().SR)
    return model.predict(model_input)


def extract_vgg_features(model, frames) -> np.ndarray:
    return model.predict(frames)


def extract_r2_plus1_features(model, frames) -> np.ndarray:
    model_input = np.array(list(map(r2_plus_1_preprocess, frames)))
    model_input = tf.signal.frame(model_input, 8, 4, axis=0).numpy()
    model_input = np.transpose(model_input, (0, 2, 3, 4, 1)).astype(np.float32)
    features = np.array(list(map(lambda x: model(tf.expand_dims(x, 0))['635'], model_input)))
    features = np.array(list(map(lambda x: np.squeeze(x), features)))
    return features


def r2_plus_1_preprocess(frame):
    resized = cv2.resize(frame, (112, 112))
    resized = np.transpose(resized, (2, 0, 1))
    return resized


def resize_preprocess(frame):
    return cv2.resize(frame, (112, 112))


def _l3_preprocess_audio(audio, sr, hop_size=1):
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

    audio_len = audio.size
    frame_len = sr
    hop_len = int(hop_size * sr)

    if audio_len < frame_len:
        print('[WARN] Duration of provided audio is shorter than window size (1 second). Audio will be padded.')

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
                                        strides=(audio.itemsize, hop_len * audio.itemsize)).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))
    return x


def process_shimmers_features(examples):
    return examples


def extract_pose_features(examples):
    inputs = tf.signal.frame(examples, 10, 5, axis=0).numpy()
    features = np.array(list(map(lambda x: compute_features(x, 0.1), inputs)))
    # norm & stand
    features = np.array(list(map(stand, features)))
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(features).astype(np.float32)


def stand(x):
    return (x - np.mean(x) / np.std(x)) if np.std(x) != 0.0 else x
