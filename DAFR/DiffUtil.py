import torch
import numpy as np
from ldm.util import instantiate_from_config
from PIL import Image

# Loads diffusion model, pretty much unmodified from code
def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        model = model.to(device)
    model.eval()
    return model

# To load a specific image for ldm, RGB [0,1]
# imName: The file name of the image to be loaded or numpy array
# m: the device for the tensor to be sent to
def loadForLDM(imName, m):
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

# A quick function to convert from BGR to RGB and remove batch
def tensorBGR2RGB(t):
    newChans = [2, 1, 0]
    size = list(t.size())
    return t.view(3, size[len(size)-2], size[len(size)-1])[newChans,:,:]

# Adds an alpha channel to numpy arrays of images based on a mask (used for saving)
def addAlpha(npy, mask):
    newNpy = np.zeros((4, npy.shape[1], npy.shape[2]))
    newNpy[:3, :, :] = npy
    newNpy[3,:,:] = mask[0] * 255
    return newNpy

    
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
