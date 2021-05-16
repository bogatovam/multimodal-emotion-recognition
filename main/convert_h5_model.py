import logging
from numbers import Real

import tensorflow as tf
from kapre.time_frequency import Melspectrogram
import numpy as np
import soundfile as sf
import resampy

TARGET_SR = 48000


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


def _preprocess_audio_batch(audio, sr, center=True, hop_size=0.1):
    """Process audio into batch format suitable for input to embedding model """
    if audio.size == 0:
        raise RuntimeError('Got empty audio')

    # Warn user if audio is all zero
    if np.all(audio == 0):
        logging.warning('Provided audio is all zeros')

    # Check audio array dimension
    if audio.ndim > 2:
        raise RuntimeError('Audio array can only be be 1D or 2D')

    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    if not isinstance(sr, Real) or sr <= 0:
        raise RuntimeError('Invalid sample rate {}'.format(sr))

    if not isinstance(hop_size, Real) or hop_size <= 0:
        raise RuntimeError('Invalid hop size {}'.format(hop_size))

    if center not in (True, False):
        raise RuntimeError('Invalid center value {}'.format(center))

    # Resample if necessary
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr_orig=sr, sr_new=TARGET_SR, filter='kaiser_best')

    audio_len = audio.size
    frame_len = TARGET_SR
    hop_len = int(hop_size * TARGET_SR)

    if audio_len < frame_len:
        logging.warning('Duration of provided audio is shorter than window size (1 second). Audio will be padded.')

    if center:
        # Center audio
        audio = _center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = _pad_audio(audio, frame_len, hop_len)

    print(audio.shape)

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
                                        strides=(audio.itemsize, audio.itemsize)).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))
    return x


if __name__ == "__main__":
    new_model = tf.keras.models.load_model(
        '../models/pretrained/c3d.h5', compile=False)

    # Show the model architecture
    new_model.trainable = False
    new_model.summary()
