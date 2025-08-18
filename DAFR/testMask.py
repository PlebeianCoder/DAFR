import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
import os
import glob
from torch.nn import CosineSimilarity
import torchvision
from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead

from nn_modules import LandmarkExtractor, FaceXZooProjector
from landmark_detection.pytorch_face_landmark.models import mobilefacenet
from DiffUtil import loadForLDM

torch.set_grad_enabled(False)
device = torch.device("cuda")
face_landmark_detector = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
sd = torch.load('./landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar',
                map_location=device)['state_dict']
face_landmark_detector.load_state_dict(sd)
location_extractor = LandmarkExtractor(device, face_landmark_detector, (112, 112)).to(device)
fxz_projector = FaceXZooProjector(device=device, img_size=(112, 112), patch_size=(112, 112)).to(device)

# optimizing setting
uv_mask = torchvision.transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L')).unsqueeze(0).to(device)

def maskIt(face_imgs, style_attack_mask, saveName=""):
        with torch.no_grad():
            # Need to resize to 112
            face_imgs = torch.unsqueeze(face_imgs, 0)
            to112 = torchvision.transforms.Resize((70, 112))
            t_mask = to112(style_attack_mask)
            base = torch.zeros((1, 3, 112, 112)).to(device)
            base[:, :, 21:91, :] = t_mask
            style_attack_mask = base * uv_mask
            if saveName != "":
                torchvision.utils.save_image(style_attack_mask, f"save_path/UV_{saveName}.png")
            preds = location_extractor(face_imgs).to(device)
            style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True).to(device)
            style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)

            if style_masked_face.ndim > 3:
                style_masked_face = style_masked_face[0]
            return style_masked_face

def noResize_maskIt(face_imgs, style_attack_mask, saveName=""):
        with torch.no_grad():
            # Need to resize to 112
            face_imgs = torch.unsqueeze(face_imgs, 0)
            to112 = torchvision.transforms.Resize(112)
            t_mask = to112(style_attack_mask)
            style_attack_mask = t_mask * uv_mask
            if saveName != "":
                torchvision.utils.save_image(style_attack_mask, f"save_path/UV_{saveName}.png")
            preds = location_extractor(face_imgs).to(device)
            style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True).to(device)
            style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)

            if style_masked_face.ndim > 3:
                style_masked_face = style_masked_face[0]
            return style_masked_face

def find_jpeg_files(directory):
    """
    Finds all JPEG files in the specified directory.
    """
    jpeg_files = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.jpeg')) + glob.glob(os.path.join(directory, '*.JPG')) + glob.glob(os.path.join(directory, '*.JPRG'))
    return jpeg_files

# TEST: randInitial, finalK, (k,a), scale)
def test_texture_attack(inet, anchor_path, mask_path, thresh, test_images, limit, saveImages=False, saveEnd = ""):
    imgs = find_jpeg_files(test_images)[:limit]
    anchor = torch.load(anchor_path, map_location=device)
    face_mask, pil_mask = loadForLDM(mask_path, device)
    succ = 0
    css = []
    cos_sim = CosineSimilarity()
    for i in range(len(imgs)):
        rgbX, pilX = loadForLDM(str(imgs[i]), device)
        if "TEXTURE" in mask_path:
            rgbX = torch.unsqueeze(maskIt(rgbX, face_mask, saveEnd),0)
        else:
            rgbX = torch.unsqueeze(noResize_maskIt(rgbX, face_mask, saveEnd),0)
        if saveImages:
            torchvision.utils.save_image(rgbX, f"save_path/{saveEnd}.png")
        fin = inet.returnEmbedding(rgbX)
        cs = cos_sim(fin, anchor).item()
        print(f"For {imgs[i]}\nCosine: {cs} and Win: {cs < thresh}")
        if cs < thresh:
            succ += 1
        css.append(cs)
    
    print(f"Dodging Success: {succ}/{len(imgs)}")
    print(f"Mean: {sum(css)/len(css)}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FaceDatasets["VGGFACE2"]
size = dataset.get_size("LARGE")


inet = IResNetHead.construct(
    dataset, size, training=False, device=device, weights_directory=Path("weight_path")
)

celeb = "BeyonceKnowles"
anchor_path = f"anchor_path"

thresh = 0.2
limit = 20

test_images = "test_image_dir"

for f in  glob.glob(os.path.join("mask_dirs", '*.png')):
    print(f"With anc: {anchor_path}\nWith mask: {f}")
    test_texture_attack(inet, anchor_path, f, thresh, test_images, limit, True, os.path.basename(f)[:20])