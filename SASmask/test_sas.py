from advfaceutil.datasets.faces.pubfig import PubFigDatasetSize
from pathlib import Path
import torch
from PIL import Image
from torchvision.utils import save_image

import glob
import os

from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from insight2Adv import InsightNet
import cv2

from test_stymask import test_stymask


def find_jpeg_files(directory):
    """
    Finds all JPEG files in the specified directory.
    """

    jpeg_files = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.jpeg')) + glob.glob(os.path.join(directory, '*.JPG')) + glob.glob(os.path.join(directory, '*.JPRG'))
    return jpeg_files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FaceDatasets["VGGFACE2"]

size = dataset.get_size("LARGE")
faceClasses = size.class_names

inet = IResNet.construct(
    dataset, size, training=False, device=device, weights_directory=Path("weights_path")
)

celeb_dir = "celeb_path"
researcher_dir = "res_path"
uv_path = "./prnet/new_uv.png"
style_path =  "style_ref"

out_path = "out_path"

target_class = 0
anchorPath = "anchor_path"

img_class = 97
if img_class < len(faceClasses)-3:
    dir_path = celeb_dir
else:
    dir_path = researcher_dir

dir_path += "/" + faceClasses[img_class] + "/"
test_path = find_jpeg_files(dir_path)[:100]

attack_model_path = "attack_model"

test_stymask(inet, style_path, uv_path, 0.105, test_path, anchorPath, attack_model_path)