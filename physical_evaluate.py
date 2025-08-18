import sys
import os    
import random
from pathlib import Path
import pickle
import statistics
import glob
import torch
from torch.nn import CosineSimilarity
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.io import read_image
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.utils import save_image

from advfaceutil.datasets.faces import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead
from advfaceutil.recognition.mobilefacenet import MobileFaceNet
from advfaceutil.recognition.clip import FaRL
import argparse

import warnings
warnings.simplefilter('ignore', UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device), flush=True)

# To load a specific image for ldm, RGB [0,1]
# imName: The file name of the image to be loaded or numpy array
# m: the device for the tensor to be sent to
# Same as loadForLDM
def loadForClf(imName, m):
    if isinstance(imName, str):
        # String
        pilImg = Image.open(imName)
        img = np.asarray(pilImg).copy().astype(np.float32)
    elif isinstance(imName, Image.Image):
        # Pillow image
        pilImg = imName.copy()
        img = np.asarray(imName).copy().astype(np.float32)
    else:
        # Numpy array
        # Make channel go last
        if imName.shape[0] == 3:
            imName = np.transpose(imName, (1, 2, 0))
        img=imName.copy().astype(np.float32)
        pilImg = Image.fromarray(imName.astype(np.uint8))
        
    img = img.astype(np.float32) / 255.0 # Normalize
    if len(img.shape) != 3:
        return None, None # To handle non colour images
    img = np.transpose(img, (2, 0, 1)) # Make channel go first
    img = torch.from_numpy(img).to(m) # convert to torch
    return img, pilImg # returns RGB

def main(args):

    mode = 'targeted'
    r = args.model_arch
    print(f"Starting Eval with args: {args}")

    if r=="R100":
        embedder = IResNet.construct(
            FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device, weights_directory=Path("")
        )
    elif r=="FTED100":
        embedder = IResNetHead.construct(
            FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device, weights_directory=Path("")
        )
    elif r=="FARL":
        embedder = FaRL.construct(
            FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
            weights_directory=Path("")
        )
    elif r=="MFN":
        embedder = MobileFaceNet.construct(
            FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
            weights_directory=Path("")
        )

    cos_sim = CosineSimilarity()
    all_cos_sims = []

    for test_num in range(len(args.src_dir)):
        anchor_embedding = torch.load(args.anchor_path[test_num], map_location=device)    

        # Load these files
        files = sorted(glob.glob(os.path.join(args.src_dir[test_num], '*.jpg')) + glob.glob(os.path.join(args.src_dir[test_num], '*.JPG')))
        files = files[:args.test_images]

        for i in range(len(files)):
            image_pt, _ = loadForClf(files[i], device)
            cur_embedding = embedder.returnEmbedding(torch.unsqueeze(image_pt,0))

            cur_cos_sim = cos_sim(anchor_embedding, cur_embedding).to("cpu")
            all_cos_sims.append(cur_cos_sim.item())
    
    s100_count = sum(1 for x in all_cos_sims if x < args.threshold_s100)
    s1000_count = sum(1 for x in all_cos_sims if x < args.threshold_s1000)

    print(f"SUMMARY OF EVAL\nAttack Success Rate with threshold_s100 ({args.threshold_s100})= {s100_count}/{len(all_cos_sims)} ({(s100_count/len(all_cos_sims))*100:.4f})\nAttack Success Rate with threshold_s1000 ({args.threshold_s1000})= {s1000_count}/{len(all_cos_sims)} ({(s1000_count/len(all_cos_sims))*100:.4f})\nmean +- stddev Cossim: {sum(all_cos_sims)/len(all_cos_sims):.4f} +- {statistics.stdev(all_cos_sims):.4f} in the interval [{min(all_cos_sims)},{max(all_cos_sims)}]")

if __name__ == '__main__':
    
    from types import SimpleNamespace

    # test_images is 50 
    thresholds_100 = {
        "FTED100": 0.5300,
        "R100": 0.2687,
        "MFN": 0.6156,
        "FARL": 0.7684
    }
    thresholds_1000 = {
        "FTED100": 0.8355,
        "R100": 0.2370,
        "MFN": 0.6622,
        "FARL": 0.7657
    }
    alignment_name = {
        "FTED100": "MTCNN",
        "R100": "MTCNN",
        "MFN": "FFHQ",
        "FARL": "MTCNN"
    }
    anchors = {
    }
    
    ffhq_base = "FFHQ_name_with_mask_template"
    mtcnn_base = "MTCNN_name_with_mask_template"

    identities = list(anchors.keys())
    dirs = [entry for entry in os.listdir(mtcnn_base.replace("name", identities[0])) if os.path.isdir(os.path.join(mtcnn_base.replace("name", identities[0]), entry))]

    def setup_eval(cur_arch, d):
        cur_anchors = []
        cur_dirs = []
        for name in identities:
            cur_anchors.append(anchors[name].replace("archname", cur_arch.lower()))
            if cur_arch == "MFN":
                cur_dirs.append(os.path.join(ffhq_base.replace("name", name.lower()), d.replace(identities[0], name.lower())))
            else:
                cur_dirs.append(os.path.join(mtcnn_base.replace("name", name.lower()), d.replace(identities[0], name.lower())))

        args = SimpleNamespace(test_images=50,
                                threshold_s100=thresholds_100[cur_arch],
                                threshold_s1000=thresholds_1000[cur_arch],
                                anchor_path=cur_anchors,
                                src_dir=cur_dirs,
                                model_arch=cur_arch
                                )
        main(args)
        print("\n###############\n###############\n###############\n")

    for d in dirs:
        # First find network
        cur_arch = ""
        for arch in list(alignment_name.keys()):
            if arch.lower() in d.lower():
                cur_arch = arch
                break

        if cur_arch == "":
            # assume baseline
            for new_arch in ["MFN", "FTED100", "R100", "FARL"]:
                setup_eval(new_arch, d)
        else:
            setup_eval(cur_arch, d)