import cv2
import os
import re
from os.path import isfile, join


# This purpose of this file is to convert a video into individual frames to apply Sentiment Analysis and then reconvert
# it into a video.

def split_into_frames(frames_path, input_video_path, fps):
    cap = cv2.VideoCapture(input_video_path)
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if i % fps == 0:
            cv2.imwrite(frames_path + 'Frame' + str(i) + '.jpg', frame)
        i += 1
    cap.release()


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # For sorting the file by name
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i in range(len(files)):
        filename = pathIn + files[i]

        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    for i in range(len(frame_array)):
        # Writing to a image array
        out.write(frame_array[i])
    out.release()
