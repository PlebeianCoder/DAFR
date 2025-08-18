import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
import os
import tempfile
import glob
from torch.nn import CosineSimilarity
import torchvision
from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead

from advfaceutil.benchmark.statistic.cmmd.distance import mmd
from advfaceutil.benchmark.statistic.cmmd.embedding import ClipEmbeddingModel
from advfaceutil.benchmark.statistic.cmmd.io import (
    compute_embeddings_for_image,
)


from nn_modules import LandmarkExtractor, FaceXZooProjector
from landmark_detection.pytorch_face_landmark.models import mobilefacenet
from DiffUtil import loadForLDM

from skimage.metrics import structural_similarity as ssim
import cv2

_SIGMA = 10
_SCALE = 1000

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

def maskIt(face_imgs, style_attack_mask, saveName="", returnMask=False):
        with torch.no_grad():
            # Need to resize to 112
            to112 = torchvision.transforms.Resize((70, 112))
            t_mask = to112(style_attack_mask)
            base = torch.zeros((1, 3, 112, 112)).to(device)
            base[:, :, 21:91, :] = t_mask
            style_attack_mask = base * uv_mask
            if saveName != "":
                torchvision.utils.save_image(style_attack_mask[:,:,21:91,:], saveName)
            if returnMask:
                return style_attack_mask[:,:,24:85, :]
            face_imgs = torch.unsqueeze(face_imgs, 0)
            preds = location_extractor(face_imgs).to(device)
            style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True).to(device)
            style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)

            if style_masked_face.ndim > 3:
                style_masked_face = style_masked_face[0]
            return style_masked_face

def noResize_maskIt(face_imgs, style_attack_mask, saveName="", returnMask=False, advMask=False):
        with torch.no_grad():
            # Need to resize to 112
            to112 = torchvision.transforms.Resize(112)
            t_mask = to112(style_attack_mask)
            style_attack_mask = t_mask * uv_mask
            if saveName != "":
                torchvision.utils.save_image(style_attack_mask[:,:,21:91,:], saveName)
            if returnMask:
                return style_attack_mask[:,:,24:85, :]
            face_imgs = torch.unsqueeze(face_imgs, 0)
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
            torchvision.utils.save_image(rgbX, f"out/{saveEnd}.png")
        fin = inet.returnEmbedding(rgbX)
        cs = cos_sim(fin, anchor).item()
        print(f"For {imgs[i]}\nCosine: {cs} and Win: {cs < thresh}")
        if cs < thresh:
            succ += 1
        css.append(cs)
    
    print(f"Dodging Success: {succ}/{len(imgs)}")
    print(f"Mean: {sum(css)/len(css)}")

def mask_ssim(ref_path, mask_paths, save_path="", doDiff=False, saveNonAdv=False):
    face_mask, pil_mask = loadForLDM(ref_path, device)
    if doDiff:
        face_mask = maskIt(None, face_mask, "", True)[0]
    else:
        face_mask = noResize_maskIt(None, face_mask, "", True)[0]
    if save_path != "" and saveNonAdv:
        torchvision.utils.save_image(face_mask, f"{save_path}/NonAdv.png")
    face_mask = face_mask.cpu().numpy()
    ssims = []
    for i in range(len(mask_paths)):
        rgbX, pilX = loadForLDM(str(mask_paths[i]), device)
        if doDiff:
            rgbX = maskIt(None, rgbX, "", True)[0]
        else:
            rgbX = noResize_maskIt(None, rgbX, "", True)[0]
        if save_path!="":
            saveEnd = os.path.basename(mask_path[i])[:20]
            torchvision.utils.save_image(rgbX, f"{save_path}/{saveEnd}.png")
        cssim = ssim(face_mask,
            rgbX.cpu().numpy(),
            multichannel=True,
            data_range=1.0,
            channel_axis=0,
            )
        ssims.append(cssim)
    
    return sum(ssims)/len(ssims)

def get_files_without_phrase(directory, phrase):
    files_without_phrase = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and phrase not in filename:
            files_without_phrase.append(os.path.join(directory, filename))
    return files_without_phrase


def get_mean_mask_ssim(dir_path, is_dafr=False, givenNonAdv=None):
    # List of image extensions to look for
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

    # Find all images in the immediate directory
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(dir_path, ext)))

    adv_images = [s for s in images if "NonAdv" not in s]
    if givenNonAdv is None:
        ref_image = [s for s in images if "NonAdv" in s]
        if len(ref_image) == 0:
            return None
        ref_image = ref_image[0]
    else:
        ref_image = givenNonAdv
    print(ref_image)

    all_ssims = [mask_ssim(ref_image, [s], doDiff=is_dafr, saveNonAdv=True, save_path="") for s in adv_images]
    print("clear")
    return sum(all_ssims)/len(all_ssims)

def get_mean_mask_cmmd(dir_path, is_dafr=False, givenNonAdv=None):
    # List of image extensions to look for
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

    # Find all images in the immediate directory
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(dir_path, ext)))

    clp = ClipEmbeddingModel()
    adv_images = [s for s in images if "NonAdv" not in s]
    if givenNonAdv is None:
        ref_image = [s for s in images if "NonAdv" in s]
        if len(ref_image) == 0:
            return None
        ref_image = ref_image[0]
    else:
        ref_image = givenNonAdv

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        face_mask, _ = loadForLDM(ref_image, device)
        if is_dafr:
            face_mask = maskIt(None, face_mask, os.path.join(temp_dir, "nonadv.png"), True)[0]
        else:
            face_mask = noResize_maskIt(None, face_mask, os.path.join(temp_dir, "nonadv.png"), True)[0]
        non_adversarial_accessory_embedding = compute_embeddings_for_image(
            os.path.join(temp_dir, "nonadv.png"), clp
        )
        cmmds = []
        i=0
        for a in adv_images:
            face_mask, _ = loadForLDM(a, device)
            if is_dafr:
                face_mask = maskIt(None, face_mask, os.path.join(temp_dir, f"{i}.png"), True)[0]
            else:
                face_mask = noResize_maskIt(None, face_mask, os.path.join(temp_dir, f"{i}.png"), True)[0]
            # Compute the embeddings for the adversarial accessory
            adversarial_accessory_embedding = compute_embeddings_for_image(
                os.path.join(temp_dir, f"{i}.png"), clp
            )
            cmmds.append(mmd(non_adversarial_accessory_embedding, adversarial_accessory_embedding))
            i+=1
    
    return sum(cmmds)/len(cmmds)

# https://github.com/sayakpaul/cmmd-pytorch
def our_mmd(x: np.ndarray, y: np.ndarray):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if not torch.is_tensor(y):
        y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x, x.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(x_sqnorms, 0)
            )
        )
    )
    k_xy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x, y.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )
    k_yy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(y, y.T)
                + torch.unsqueeze(y_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )
    print(k_xx)
    print(k_yy)
    print(k_xy)
    return _SCALE * (k_xx + k_yy - 2 * k_xy)

def print_mmds(dir_path, is_dafr=False, givenNonAdv=None):
    # List of image extensions to look for
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

    # Find all images in the immediate directory
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(dir_path, ext)))

    clp = ClipEmbeddingModel()
    adv_images = [s for s in images if "NonAdv" not in s]
    if givenNonAdv is None:
        ref_image = [s for s in images if "NonAdv" in s]
        if len(ref_image) == 0:
            return None
        ref_image = ref_image[0]
    else:
        ref_image = givenNonAdv

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        face_mask, _ = loadForLDM(ref_image, device)
        if is_dafr:
            face_mask = maskIt(None, face_mask, os.path.join(temp_dir, "nonadv.png"), True)[0]
        else:
            face_mask = noResize_maskIt(None, face_mask, os.path.join(temp_dir, "nonadv.png"), True)[0]
        non_adversarial_accessory_embedding = compute_embeddings_for_image(
            os.path.join(temp_dir, "nonadv.png"), clp
        )
        cmmds = []
        i=0
        for a in adv_images:
            print(f"For {a}")
            face_mask, _ = loadForLDM(a, device)
            if is_dafr:
                face_mask = maskIt(None, face_mask, os.path.join(temp_dir, f"{i}.png"), True)[0]
            else:
                face_mask = noResize_maskIt(None, face_mask, os.path.join(temp_dir, f"{i}.png"), True)[0]
            # Compute the embeddings for the adversarial accessory
            adversarial_accessory_embedding = compute_embeddings_for_image(
                os.path.join(temp_dir, f"{i}.png"), clp
            )
            cmmds.append(our_mmd(non_adversarial_accessory_embedding, adversarial_accessory_embedding))
            print(cmmds[i])
            i+=1
    
    return sum(cmmds)/len(cmmds)

def get_cmmd(dir_path, is_dafr=False, givenNonAdv=None):
    # List of image extensions to look for
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

    # Find all images in the immediate directory
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(dir_path, ext)))

    clp = ClipEmbeddingModel()
    adv_images = [s for s in images if "NonAdv" not in s]
    if givenNonAdv is None:
        ref_image = [s for s in images if "NonAdv" in s]
        if len(ref_image) == 0:
            return None
        ref_image = ref_image[0]
    else:
        ref_image = givenNonAdv

    # Create a temporary directory
    non_adversarial_accessory_embedding = compute_embeddings_for_image(
        ref_image, clp
    )
    i=0
    adversarial_accessory_embedding = np.zeros((len(adv_images), 768), dtype=non_adversarial_accessory_embedding.dtype)

    for a in adv_images:
        # Compute the embeddings for the adversarial accessory
        emb = compute_embeddings_for_image(
            a, clp
        )
        adversarial_accessory_embedding[i] = emb.astype(adversarial_accessory_embedding.dtype)
        i+=1
    
    return mmd(non_adversarial_accessory_embedding, adversarial_accessory_embedding)

def get_mask_cmmd(dir_path, is_dafr=False, givenNonAdv=None):
    # List of image extensions to look for
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

    # Find all images in the immediate directory
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(dir_path, ext)))

    clp = ClipEmbeddingModel()
    adv_images = [s for s in images if "NonAdv" not in s]
    if givenNonAdv is None:
        ref_image = [s for s in images if "NonAdv" in s]
        if len(ref_image) == 0:
            return None
        ref_image = ref_image[0]
    else:
        ref_image = givenNonAdv

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        face_mask, _ = loadForLDM(ref_image, device)
        if is_dafr:
            face_mask = maskIt(None, face_mask, os.path.join(temp_dir, "nonadv.png"), True)[0]
        else:
            face_mask = noResize_maskIt(None, face_mask, os.path.join(temp_dir, "nonadv.png"), True)[0]
        non_adversarial_accessory_embedding = compute_embeddings_for_image(
            os.path.join(temp_dir, "nonadv.png"), clp
        )
        adversarial_accessory_embedding = np.zeros((len(adv_images), 768), dtype=non_adversarial_accessory_embedding.dtype)
        i=0
        for a in adv_images:
            face_mask, _ = loadForLDM(a, device)
            if is_dafr:
                face_mask = maskIt(None, face_mask, os.path.join(temp_dir, f"{i}.png"), True)[0]
            else:
                face_mask = noResize_maskIt(None, face_mask, os.path.join(temp_dir, f"{i}.png"), True)[0]
            # Compute the embeddings for the adversarial accessory
            emb = compute_embeddings_for_image(
                os.path.join(temp_dir, f"{i}.png"), clp
            )
            adversarial_accessory_embedding[i] = emb
            i+=1
    
    return mmd(non_adversarial_accessory_embedding, adversarial_accessory_embedding)


benches = ["diff_dirs"]

# Get cmmd 
with open("ind_mask_cmmd_stats.csv", "a") as f:
    for directory_path in benches:
        is_dafr = True
        br = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        for b in br:
            try:
                c = get_mask_cmmd(os.path.join(directory_path, b), is_dafr)
                if c is None:
                    continue
                f.write(f"{os.path.join(directory_path, b)}, {c}\n")
            except Exception as e:
                print(e)
                continue