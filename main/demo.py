from os import listdir
from os.path import isfile, join
from moviepy.editor import VideoFileClip, vfx
import tensorflow as tf
import librosa as lb
import soundfile

if __name__ == '__main__':
    # path = "E:/RAMAS/RAMAS/Data/r2plus1d_34_clip32_ft_kinetics_from_ig65m-10f4c3bf"
    # rpath = "E:/RAMAS/RAMAS/Data/Video_close/r2plus1d_34_clip32_ft_kinetics_from_ig65m-10f4c3bf"
    #
    # onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    #
    # # Import everything needed to edit face clips
    # for f in onlyfiles:
    #     print(f)
    #     # loading face gfg
    #     clip = VideoFileClip(path + "/" + f)
    #     # # rotating clip by 180 degree
    #     out = clip.fx(vfx.mirror_y)
    #     out.write_videofile(rpath + "/" + f)

    # pretrained_model_path = '../models/pretrained/r2plus1d_34_clip32_ft_kinetics_from_ig65m-10f4c3bf'
    # m = tf.keras.models.load_model(pretrained_model_path, compile=False)

    f, b = lb.load("test.wav", sr=100)
    soundfile.write("out.wav", f, 100)
