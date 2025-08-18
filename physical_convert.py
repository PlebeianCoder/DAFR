import cv2
import numpy as np
import os


def print_mask(t_name, width, height):
    texture = cv2.imread(t_name)[21:91, :, :]

    resized = cv2.resize(texture, (width, height), interpolation=cv2.INTER_CUBIC)
    base = os.path.basename(t_name)
    cv2.imwrite(""+base[:len(base)-4]+"_physical.png", resized)

base_dir = ""
masks = os.listdir(base_dir)

for m in masks:
    print(m)
    print_mask(base_dir+m, 10*112, 10*70)