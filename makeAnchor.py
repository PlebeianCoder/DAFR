import torch
from Diffv2Util import loadForLDM, l2_norm
from advfaceutil.datasets import FaceDatasets

from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead
from advfaceutil.recognition.clip import FaRL
from advfaceutil.recognition.mobilefacenet import MobileFaceNet
from insight2Adv import InsightNet
import os
import glob
from pathlib import Path
import random
import torchvision
from PIL import Image

from nn_modules import LandmarkExtractor, FaceXZooProjector
from landmark_detection.pytorch_face_landmark.models import mobilefacenet

device = torch.device("cuda")
face_landmark_detector = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
sd = torch.load('./landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar',
                map_location=device)['state_dict']
face_landmark_detector.load_state_dict(sd)
location_extractor = LandmarkExtractor(device, face_landmark_detector, (112, 112)).to(device)
fxz_projector = FaceXZooProjector(device=device, img_size=(112, 112), patch_size=(112, 112)).to(device)

# optimizing setting
uv_mask = torchvision.transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L')).unsqueeze(0).to(device)

def maskIt(face_imgs, style_attack_mask):
        with torch.no_grad():
            # Need to resize to 112
            face_imgs = torch.unsqueeze(face_imgs, 0)
            to112 = torchvision.transforms.Resize(112)
            t_mask = to112(style_attack_mask)
            style_attack_mask = t_mask * uv_mask
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
    # Use glob to find all .jpg and .jpeg files
    jpeg_files = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.jpeg')) + glob.glob(os.path.join(directory, '*.JPG')) + glob.glob(os.path.join(directory, '*.JPRG'))
    return jpeg_files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def makeAnchor(anchorPath, rnetStr, dsetStr, name, upperLimit, masked):
    anchor = torch.zeros((512)).to(device)
    files = find_jpeg_files(Path(anchorPath))[:upperLimit]
    print(anchorPath)
    print(f"Generating {name} with {len(files)} images")
    for f in files:
        rgbX, pilX = loadForLDM(str(f), device)
        rgbX = torch.unsqueeze(rgbX, 0)
        if masked:
            random_number = random.randint(1, 3)
            if random_number == 1:
                # black
                face_mask = torch.ones((1, 3, 112, 112)).to(device) * 1e-6
            elif random_number == 2:
                # white
                face_mask = torch.ones((1, 3, 112, 112)).to(device)
            else:
                # blue
                face_mask = torch.zeros((1, 3, 112, 112)).to(device)
                # print(uv_mask.size())
                face_mask[0, 2] = 0.7
            rgbX = torch.unsqueeze(maskIt(rgbX[0], face_mask),0)
        
        if rgbX.size()[rgbX.dim()-1] != inet.crop_size:
            resizeOp = torchvision.transforms.Resize(inet.crop_size)
            rgbX = resizeOp(rgbX)
            print(rgbX.size())

        tmp = inet.returnEmbedding(rgbX)
        anchor = anchor+tmp

    anchor = l2_norm(anchor)
    mskStr="unmasked"
    if masked:
        mskStr ="masked"
    torch.save(anchor, f"anchors/{mskStr}_{rnetStr}_{dsetStr}_{name}.pth")


masked = True
for r in ["mfn"]:
    for d in["PUBFIG", "VGGFACE2"]:
        for s in ["LARGE"]:
            dataset = FaceDatasets[d]
            size = dataset.get_size(s)

            if r=="r100":
                inet = IResNet.construct(
                    FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device, weights_directory=Path("weights")
                )
            elif r=="fted100":
                inet = IResNetHead.construct(
                    FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device, weights_directory=Path("weights")
                )
            elif r=="farl":
                inet = FaRL.construct(
                    FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
                    weights_directory=Path("weights")
                )
            elif r=="mfn":
                inet = MobileFaceNet.construct(
                    FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
                    weights_directory=Path("weights")
                )

            if d == "VGGFACE2":
                if r=="mfn":
                    # FFHQ alignment
                    celebDir = "celeb_dir"
                    researchDir = "research_dir"
                else:
                    # MTCNN alignment 
                    celebDir = "celeb_dir"
                    researchDir = "research_dir"
                upperLimit = 45
            else:
                if r=="mfn":
                    # FFHQ alignment
                    celebDir = "celeb_dir"
                    researchDir = "research_dir"
                else:
                    # MTCNN alignment 
                    celebDir = "celeb_dir"
                    researchDir = "research_dir"
                upperLimit = 10

            classes = size.class_names
            for f in range(len(classes)):
                if f >= len(classes) -3:
                    makeAnchor(researchDir+classes[f], r, d, classes[f], upperLimit, masked)
                else:
                    makeAnchor(celebDir+classes[f], r, d, classes[f], upperLimit, masked)
                print(f"Saved {classes[f]}")
