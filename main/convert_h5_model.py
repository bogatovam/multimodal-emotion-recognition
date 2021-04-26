import tensorflow as tf
from kapre.time_frequency import Melspectrogram

if __name__ == "__main__":
    new_model = tf.keras.models.load_model(
        'D:/2021/hse/multimodal-emotion-recognition/models/openl3_audio_mel256_music.h5',
        custom_objects={'Melspectrogram': Melspectrogram})

    # Show the model architecture
    new_model.summary()
