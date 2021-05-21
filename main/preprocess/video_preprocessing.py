import math
import os

import cv2
import numpy as np


def open_video(filename: str, size=224) -> tuple:
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
                raise RuntimeError

    fps = cap.get(cv2.CAP_PROP_FPS)

    example_frames = []
    while cap.isOpened():
        read, frame = cap.read()
        if not read:
            break
        example_frames.append(cv2.resize(frame, (size, size)))

    cap.release()
    return example_frames, fps


def extract_frames_from_video_with_fps(frames: list, fps, example: str, offset: float, duration: float, modality):
    offset_frames_count = math.floor(fps * offset)
    extracted_frames_count = math.floor(fps * duration)

    source_frames_count = math.floor(modality.config.FPS * duration)
    if extracted_frames_count < source_frames_count:
        diff = (source_frames_count - extracted_frames_count)
        delta = math.floor(diff / 2)
        offset_frames_count = math.floor(max(0, offset_frames_count - delta))
        extracted_frames_count = math.floor(min(len(frames) - 1, extracted_frames_count + diff))

    example_frames_list = frames[offset_frames_count:offset_frames_count + extracted_frames_count]

    example_total_frames = len(example_frames_list)

    frames_period = math.ceil(example_total_frames / source_frames_count)
    period_res = source_frames_count - (math.ceil(example_total_frames / frames_period))

    print("[INFO][{}] video frames_period:={}, period_res:={}".format(example, frames_period, period_res))

    # to get N frames from the face we extract frames with period and then randomly add
    counter_1 = 0
    result_list = []
    for i in range(example_total_frames):
        if frames_period == 1 or i % frames_period == 0:
            result_list.append(example_frames_list[i])
        elif (i - frames_period // 2) % frames_period == 0 and counter_1 < period_res:
            counter_1 += 1
            result_list.append(example_frames_list[i])

    result_list_resized = []
    for frame in result_list:
        result_list_resized.append(cv2.resize(frame, (modality.config.SHAPE, modality.config.SHAPE)))

    if len(result_list_resized) != source_frames_count:
        delta = source_frames_count - len(result_list_resized)
        result_list_resized = np.pad(result_list_resized, ((0, delta), (0, 0), (0, 0), (0, 0)), 'constant')

    return np.asarray(result_list_resized)
