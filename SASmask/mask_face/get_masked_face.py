# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import dlib
from mask_face.utils.aux_functions import *

# Command-line input setup
parser = argparse.ArgumentParser(description="MaskTheFace - Python code to mask faces dataset")
parser.add_argument("--path", type=str, default="./face_iamges",
                    help="Path to either the folder containing images or the image itself")
parser.add_argument("--mask_type", type=str, default="surgical", choices=["surgical", "N95", "KN95", "cloth"],
                    help="Type of the mask to be applied.")
parser.add_argument("--pattern", type=str, default="", help="Type of the pattern. Available options in masks/textures")
parser.add_argument("--pattern_weight", type=float, default=0.5, help="Weight of the pattern. Must be between 0 and 1")
parser.add_argument("--color", type=str, default="#0473e2",
                    help="Hex color value that need to be overlayed to the mask")
parser.add_argument("--color_weight", type=float, default=0.5,
                    help="Weight of the color intensity. Must be between 0 and 1")
parser.add_argument("--code", type=str,
                    default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.jpg, cloth-masks/textures/fruits/pineapple.jpg, N95, surgical_blue, surgical_green",
                    # default="",
                    help="Generate specific formats")

parser.add_argument("--verbose", default=False, dest="verbose", action="store_true", help="Turn verbosity on")
parser.add_argument("--write_original_image", dest="write_original_image", action="store_true",
                    help="If true, original image is also stored in the masked folder")
parser.set_defaults(feature=False)
args = parser.parse_args()


class argument:
    def __init__(self):
        self.path = "face_iamges"
        self.write_path = self.path + "_masked"
        self.detector = dlib.get_frontal_face_detector()  # Huihui: 获取人脸检测器
        path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(path_to_dlib_model):
            download_dlib_model()
        self.predictor = dlib.shape_predictor(path_to_dlib_model)  # Huihui: 预测人脸关键点
        self.color = ''
        self.pattern = ''
        self.type = ''
        self.color_weight = 0.5  # Weight of the color intensity. Must be between 0 and 1
        self.pattern_weight = 0.5


args = argument()


def get_masked_face(img, mask_code):
    args.color = mask_code['color']
    args.pattern = mask_code['texture']

    masked_image, mask_binary_array, original_image = mask_image(img, args)
    return masked_image, mask_binary_array, original_image

