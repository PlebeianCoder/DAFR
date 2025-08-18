from advfaceutil.datasets.faces.pubfig import PubFigDatasetSize
from pathlib import Path
import torch
from PIL import Image
from torchvision.utils import save_image

import glob
import os

from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead
from advfaceutil.recognition.clip import FaRL
from advfaceutil.recognition.mobilefacenet import MobileFaceNet
import cv2
import random

from train_stymask import train_stymask


def find_jpeg_files(directory):
    """
    Finds all JPEG files in the specified directory.
    """
    # Use glob to find all the files
    jpeg_files = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.jpeg')) + glob.glob(os.path.join(directory, '*.JPG')) + glob.glob(os.path.join(directory, '*.JPRG'))
    return jpeg_files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FaceDatasets["PUBFIG"]

size = dataset.get_size("LARGE")
faceClasses = size.class_names

# inet = IResNet.construct(
#     dataset, size, training=False, device=device, weights_directory=Path("path")
# )

# inet = FaRL.construct(
#     FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
#     weights_directory=Path("path")
# )
# inet = IResNetHead.construct(
#     dataset, size, training=False, device=device, weights_directory=Path("path")
# )

inet = MobileFaceNet.construct(
    FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
    weights_directory=Path("weight_path")
)

celeb_dir = "celeb_path"
researcher_dir = "res_path"
uv_path = "./prnet/new_uv.png"

style_path = "../ref_images/bfp.png"

out_path = "output/"
print(faceClasses)
target_class = None
img_class = faceClasses.index('BeyonceKnowles')

anchorPath = f"anchor_path"
if img_class < len(faceClasses)-3:
    dir_path = celeb_dir
else:
    dir_path = researcher_dir

dir_path += "/" + faceClasses[img_class] + "/"
print(dir_path)
gen_path = find_jpeg_files(dir_path)[:25]

train_stymask(img_class, target_class, faceClasses, inet, gen_path, uv_path, style_path, 30, 1000, 0.01, 100, 10000, 1, 0.01,
                 out_path, device, 0.6, anchorPath, withStyle=True, toSave=True)