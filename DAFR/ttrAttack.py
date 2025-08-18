from advfaceutil.datasets.faces.pubfig import PubFigDatasetSize
from pathlib import Path
import torch
from PIL import Image
from torchvision.utils import save_image
from logging import getLogger
import glob
import os
import inspect

from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition.clip import FaRL
from advfaceutil.recognition.mobilefacenet import MobileFaceNet
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead

from Txt2Adv3DAttack import texture_attack, load_model_from_config
from omegaconf import OmegaConf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FaceDatasets["VGGFACE2"]

config = OmegaConf.load("configs/stable-diffusion/v2-inference.yaml")
model = load_model_from_config(config, "diffusion_weights", device)
size = dataset.get_size("LARGE")
faceClasses = size.class_names

mfn = MobileFaceNet.construct(
    FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
    weights_directory=Path("mfn_weights")
)


celebDir = "celeb_dir"
researchDir = "research_dir"

faceClasses = ["BeyonceKnowles"]

inet_list = [mfn]
inet_strs = ["mfn"]

for i in range(len(inet_list)):
    anchorPaths = "anchor_paths"

    directory_path = celebDir+faceClasses[0]

    names = glob.glob(os.path.join(directory_path, '*.jpg')) + glob.glob(os.path.join(directory_path, '*.JPG'))
    names = names[:25]
    print(f"Attacking {inet_strs[i]}")
    txt = texture_attack(prompt="blue flower pattern",
                name_set=names, clfSet=[inet_list[i]], anchorPic=anchorPaths, addLabel=False, matchLighting=False,
                img_class=0, target_class=None, desc=f"bfp_2_1_08", s=2, steps=0.8, successThresh=0.8, anchorThresh=0.6, indirectDodge=False,
                seed=2, scaled=False, k=1, genS=200, scale=8, a=1,
                randInitial=True, innerK=1, finalK=0, device=device, toSave=True, toPrint=True, outpath="output/",
                tmp_mesh_path="3Dobjs/tmp/tmp.obj", mesh_path="3Dobjs/tmp/face_mask.obj",
                config=config, model=model, faceClasses=faceClasses, nInter=1, logger=getLogger("ttrAttack"))

    # txt = texture_attack(prompt="abstract light purple and pink computer pattern with colourful circles, rectangles, triangles and and semi circles like it was made in the 1990s",
    #             name_set=names, clfSet=[inet_list[i]], anchorPic=anchorPaths, addLabel=False, matchLighting=False,
    #             img_class=0, target_class=None, desc=f"purp_2_1_08", s=2, steps=0.8, successThresh=0.8, anchorThresh=0.6, indirectDodge=False,
    #             seed=2, scaled=False, k=1, genS=200, scale=8, a=1,
    #             randInitial=True, innerK=1, finalK=0, device=device, toSave=True, toPrint=True, outpath="output/",
    #             tmp_mesh_path="3Dobjs/tmp/tmp.obj", mesh_path="3Dobjs/tmp/face_mask.obj",
    #             config=config, model=model, faceClasses=faceClasses, nInter=1, logger=getLogger("ttrAttack"))