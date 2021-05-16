from os import listdir
from os.path import isfile, join
from moviepy.editor import VideoFileClip, vfx

if __name__ == '__main__':
    path = "E:/RAMAS/RAMAS/Data/tmp"
    rpath = "E:/RAMAS/RAMAS/Data/Video_close/tmp"

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # Import everything needed to edit video clips
    for f in onlyfiles:
        print(f)
        # loading video gfg
        clip = VideoFileClip(path + "/" + f)
        # # rotating clip by 180 degree
        out = clip.fx(vfx.mirror_y)
        out.write_videofile(rpath + "/" + f)
