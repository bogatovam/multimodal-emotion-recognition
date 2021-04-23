import os

import moviepy.editor as mp
import configs.deception_default_config as config

from configs.dataset.modality import Modality
from utils.dirs import create_dirs

if __name__ == "__main__":
    VIDEO_EXT = '.mp4'
    AUDIO_EXT = '.wav'

    source_video_folder = config.MODALITY_TO_DATA[Modality.VIDEO_SCENE]
    target_audio_folder = config.MODALITY_TO_DATA[Modality.AUDIO]

    create_dirs([target_audio_folder])

    samples_count = 0
    for filename in os.listdir(source_video_folder):
        if filename.endswith(VIDEO_EXT):
            video_full_file_name = os.path.join(source_video_folder, filename)
            my_clip = mp.AudioFileClip(video_full_file_name)

            audio_file_name = os.path.splitext(video_full_file_name)[0] + AUDIO_EXT
            my_clip.write_audiofile(audio_file_name)

            samples_count += 1

    print("Processed {} videos".format(samples_count))
