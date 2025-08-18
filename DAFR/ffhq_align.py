"""
https://github.com/HansSunY/DiffAM/blob/main/utils/align_utils.py
"""

"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
	https://github.com/NVlabs/ffhq-dataset
	http://dlib.net/face_landmark_detection.py.html

requirements:
	apt install cmake
	conda install Pillow numpy scipy
	pip install dlib
	# download face landmark model from:
	# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
import time
import numpy as np
import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage
import dlib
import math
from advfaceutil.datasets import FaceDatasets

print("Hey")
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system(f'wget -P . http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


def run_alignment(image_path, output_size):
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system(f'wget -P . http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("pretrained/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor, output_size=output_size, transform_size=output_size)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def get_landmark(filepath, predictor):
    """get landmark with dlib
	:return: np.array shape=(68, 2)
	"""
    detector = dlib.get_frontal_face_detector()

    try:
        img = dlib.load_rgb_image(filepath)
    except:
        # Means weird file
        return None
    dets = detector(img, 1)
    shape = None
    for k, d in enumerate(dets):
        shape = predictor(img, d)
    if shape is None:
        return None
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath, predictor, output_size=112, transform_size=112):
    """
	:param filepath: str
	:return: PIL Image
	"""
    try:

        lm = get_landmark(filepath, predictor)
        if lm is None:
            return None

        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        img = PIL.Image.open(filepath)
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        return img
    except:
        print(f"Error on {filepath}")
        return None


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_on_paths(file_path, res_path, predictor):
    res = align_face(file_path, predictor)
    if res is None:
        return None
    res = res.convert('RGB')
    res.save(res_path)

def run(root_path, out_path):
    root_path = root_path
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            fname = os.path.join(out_path, file)
            if not os.path.exists(fname):
                extract_on_paths(file_path, fname, predictor)
            print(f"Done {file_path}")

dirs = [
    "dirs"
]

dataset = FaceDatasets["PUBFIG"]
size = dataset.get_size("LARGE")
faceClasses = size.class_names
print("bang")
for dir_path in dirs:
    for e in faceClasses:
        print(f"Starting {e}")
        run(os.path.join(dir_path, e), os.path.join(dir_path, e).replace("replace_base", "replace_target"))