"""
Code based on https://github.com/EricDai0/advdiff
"""

from copy import copy

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from DiffUtil import loadForLDM, l2_norm


# For Alignment
from pathlib import Path
from pyfacear import Environment
from pyfacear import FaceGeometry
from pyfacear import landmarks_from_results
from pyfacear import Mesh as PyFaceMesh
from pyfacear import OBJMeshIO
from pyfacear import OriginPointLocation
from pyfacear import PerspectiveCamera
from pyfacear.uvproject import homogenise

import cv2
import random
from mediapipe import solutions
from pyfacear import landmarks_from_results
from torch.nn import CosineSimilarity
import torchvision
import numpy as np
import math
from PIL import Image

from nn_modules import LandmarkExtractor, FaceXZooProjector
from landmark_detection.pytorch_face_landmark.models import mobilefacenet

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

import torchvision

# WE LOVE YOU KORNIA
from kornia.geometry.transform import warp_affine
from kornia.color import rgb_to_hsv, hsv_to_rgb

import os
import glob

class Txt2Adv3DDDIMSampler(object):
    def __init__(self, model, schedule="linear", clfSet=[],
                 innerK=10,finalK=0, baseLabel=8, nInter=1, 
                 anchorPic=False, addLabel=False, classCount=10, doMatch=False, indirectDodge=False, 
                 successThresh=0.8, toBGR=False, anchorThresh=1.7115, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.clfSet = clfSet
        self.base = None
        self.baseK = innerK
        self.finalK = finalK
        self.faceSet = []
        self.faceCounter = 0
        self.baseLabel = baseLabel
        self.anchorPic = anchorPic
        self.addLabel = addLabel
        self.classCount = classCount
        self.mesh_path = "3Dobjs/Face_Mask/face_mask.obj"
        self.tmp_mesh_path = "3Dobjs/tmp/tmp.obj"
        self.nInter = nInter
        self.doMatch = doMatch
        self.indirectDodge = indirectDodge
        self.successThresh = successThresh
        self.anchor = None
        self.toBGR = toBGR
        self.anchorThresh = anchorThresh
        device = torch.device("cuda")
        face_landmark_detector = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
        sd = torch.load('./landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar',
                        map_location=device)['state_dict']
        face_landmark_detector.load_state_dict(sd)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, (112, 112)).to(device)
        self.fxz_projector = FaceXZooProjector(device=device, img_size=(112, 112), patch_size=(112, 112)).to(device)

        # optimizing setting
        self.uv_mask = torchvision.transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L')).unsqueeze(0).to(device)

    def getAnchor(self):
        return self.anchor

    def getExtractor(self):
        return self.location_extractor

    def getProjector(self):
        return self.fxz_projector

    def getUV(self):
        return self.uv_mask

    # For interleaving faces and rotations
    def setSet(self, imgSet, pilSet):
        self.faceSet = imgSet
        self.pilSet = pilSet
        self.faceCounter = 0

    def setMeshPaths(self, p1, p2):
        self.mesh_path = p1
        self.tmp_mesh_path = p2

    def isFace(self, pilImg):
        # Get the landmark results from MediaPipe
        with solutions.face_mesh.FaceMesh() as face_mesh:
            results = face_mesh.process(np.array(pilImg))

        landmarks = landmarks_from_results(results)

        return len(landmarks) == 1 # to signify that a face has been found

    @staticmethod
    def faceProcess(face_imgs, style_attack_mask, uv_mask, location_extractor, fxz_projector, device):
        # Need to resize to 112
        face_imgs = torch.unsqueeze(face_imgs, 0)
        to112 = torchvision.transforms.Resize((70, 112))
        base = torch.zeros((1, 3, 112, 112)).to(device)
        # print(uv_mask.size())
        t_mask = to112(style_attack_mask)
        # print(t_mask.size())
        base[:, :, 21:91, :] = t_mask
        style_attack_mask = base * uv_mask
        with torch.no_grad():
            preds = location_extractor(face_imgs).to(device)
        style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True).to(device)
        style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)
        if style_masked_face.ndim > 3:
            style_masked_face = style_masked_face[0]
        return style_masked_face

    def input2Clf(self, genX, device, crop_size):
        prevCounter = False, self.faceCounter
        foundFace = self.isFace(self.pilSet[self.faceCounter])
        bgrImg = False

        # So we can dont enter the loop confidently
        if foundFace:
            bgrImg = self.faceProcess(self.faceSet[self.faceCounter], genX, self.uv_mask, self.location_extractor,
                                    self.fxz_projector, device)
            if isinstance(bgrImg, bool):
                foundFace = False

        while not foundFace:
            self.faceCounter = (self.faceCounter+1)%len(self.faceSet)
            if self.faceCounter == prevCounter:
                exit("ERROR: No faces could be detected in any of the pictures")
            foundFace = self.isFace(self.pilSet[self.faceCounter])
            if foundFace:
                bgrImg = self.faceProcess(self.faceSet[self.faceCounter], genX, self.uv_mask, self.location_extractor,
                                    self.fxz_projector, device)
                if isinstance(bgrImg, bool):
                    foundFace = False
        
        if isinstance(bgrImg, bool):
            exit("No Face Found!")
        
        # Final update to the counter
        self.faceCounter = (self.faceCounter+1)%len(self.faceSet)

        
        return bgrImg
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               # Below are adversarial inputs
               steps = 1, a=20,
               s=10, k =3,seed=42, scaled=False,
               label=None, target_label=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates, success = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    # Below are the adversarial inputs
                                                    s=s, a=a, steps=steps, label=label,
                                                    target_label=target_label, k=k, seed=seed, scaled=scaled,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates, success
    
    # Find good initial by lots of quick gens with S/5
    @torch.no_grad()
    def findGoodInitial(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               # Below are adversarial inputs
               steps = 1, a=20,
               s=10, k =3,seed=42, scaled=False,
               label=None, target_label=None,
               **kwargs
               ):
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        device = self.model.betas.device
        seed_everything(seed)
        
        img = torch.randn(size, device=device)
        # We want to find a good initial point using scores
        # Each attack is only k of 1, but then we do k of them
        for _ in range(k):
            grads, _, _= self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    # Below are the adversarial inputs
                                                    s=s, a=a, steps=steps, label=label,
                                                    target_label=target_label, k=1, seed=seed, scaled=scaled,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=img, # We want x_T
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    initial=True # We want a good initial
                                                    )
            img = img + s*grads.float()
        return img

    # For dodging in theory
    def get_target_label(self,logits, label, device):
        
        rates, indices = logits.sort(1, descending=True) 
        rates, indices = rates.squeeze(0), indices.squeeze(0)  
        print(label)
        print(rates)
        print(indices)
        
        tar_label = torch.zeros((self.classCount), dtype=torch.long).to(device)
        
        if self.baseLabel == indices[0].item():  # classify is correct
            tar_label[indices[1].item()] = 1
            self.dodge_target = indices[1].item()
        else:
            tar_label[indices[0].item()] = 1
            self.dodge_target = indices[0].item()

        
        print(f"Going to {tar_label.argmax()}")

        return tar_label

    # For good descriptions of the parameter meanings look at the sampling parameters comments
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      # Below are adversarial steps
                      steps = 1, s=10, a=20, label=None, target_label=None, k=3, seed=42, scaled=False, initial=False):
        
        # self.dodge_target=None

        device = self.model.betas.device
        seed_everything(seed)
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        img_transformed = torch.zeros(3, 224, 224) # THIS IS ONLY HERE FOR WHEN THE ATTACK FAILS

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        pri_img = torch.clone(img.detach()) 
        clfPtr = 0
        if len(self.clfSet) == 0:
            print("No classifier given!")
            return False, False, False

        if self.anchorPic != "":
            anchor = torch.load(self.anchorPic, map_location=device)
            self.anchor = anchor

        dodgeWeight = 1
        if not self.indirectDodge and target_label is None:
            # For minimizing the base class
            dodgeWeight = -1

        imperson_success_num=0
        # run attack k times if we need to
        innerK = copy(self.baseK)

        cos_similarity = CosineSimilarity()

        for kIter in range(k):
            print(f"Starting iteration {kIter+1}/{k}!")
            seed_everything(seed)
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod
            img = torch.clone(pri_img.detach())
            # Generation loop
            for i, step in enumerate(iterator):
                index = total_steps - i - 1

                ts = torch.full((b,), step, device=device, dtype=torch.long)

                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
                img, pred_x0 = outs

                # ADV TIME
                # co = index/total_steps
                # if(index > total_steps * 0 and index <= total_steps * steps) and ((co < 0.2) or (co >0.6) or (co <= 0.55 and co >=0.5) or (co <= 0.45 and co >=0.4) or (co <= 0.35 and co >=0.3) or (co <= 0.25 and co >=0.2)):
                if index > total_steps * 0 and index <= total_steps * steps:
                    for ink in range(innerK):
                        # NOTE: img is latent space z
                        with torch.enable_grad():
                            
                            img_n = img.detach().requires_grad_(True)
                            prevCounter = self.faceCounter
                            gradient = torch.zeros_like(img_n).to(device)
                            loops=0
                            while (loops==0 or prevCounter != self.faceCounter) and loops < self.nInter:
                                loops+=1
                                genX = self.model.decode_first_stage(img_n)
                                img_transformed = torch.unsqueeze(self.input2Clf(genX, device, self.clfSet[clfPtr].crop_size), 0)
                                if self.anchorPic != "":
                                    # For anchor attacks
                                    curEmbed = self.clfSet[clfPtr].returnEmbedding(img_transformed)
                                    target = cos_similarity(anchor, curEmbed)

                                    if target_label is None:
                                        print(f"Target: {target} on {index} was {target.item()<self.anchorThresh}")
                                    else: 
                                        print(f"Target: {target} on {index} was {target.item()>self.anchorThresh}")
                                    gradient += torch.autograd.grad(target, img_n)[0]
                                    
                                else:
                                    # For non embeddings
                                    if self.addLabel:
                                        attackLabel = torch.zeros(self.classCount).to(device)
                                        attackLabel[self.baseLabel] = 1
                                        logits = self.clfSet[clfPtr](img_transformed, attackLabel)
                                    else:
                                        logits = self.clfSet[clfPtr](img_transformed)
                                    clfPtr = (clfPtr + 1) % len(self.clfSet)
                                    log_probs = F.log_softmax(logits, dim=-1)#F.log_softmax(logits, dim=-1)
                                    if target_label is None:
                                        # Dodging
                                        if not self.indirectDodge:
                                            tar_label = torch.ones_like(label).to(device) * self.baseLabel
                                        else:
                                            tar_label = self.get_target_label(logits, label, device)
                                    else:
                                        # Impersonation
                                        tar_label = torch.ones_like(label).to(device) * target_label
                                    
                                    selected = log_probs[range(len(logits)), tar_label]
                                    tmpGrad = torch.autograd.grad(selected.sum(), img_n)[0]
                                    gradient += tmpGrad


                        gradient = gradient / self.nInter

                        if scaled:
                            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
                            # print(sqrt_one_minus_alphas[index])
                            img = img + sqrt_one_minus_at * 10 * s * gradient.float()
                            img = img + dodgeWeight * sqrt_one_minus_at * 10 * s * gradient.float()
                        else:
                            # f formula
                            img = img + dodgeWeight * s * gradient.float() * (math.exp(3*((index/total_steps)-0.6) - 3* min(0,(index/total_steps)-0.5))) # 
                        

                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)

            # Only do this last part if generation is adversarial, meaning steps is non negative
            if steps >= 0:
                clfPtr = 0
                # Last bit of grinding
                for ink in range(self.finalK):
                    with torch.enable_grad():
                        img_n = img.detach().requires_grad_(True)
                        prevCounter = self.faceCounter
                        gradient = torch.zeros_like(img_n).to(device)
                        loops=0
                        while (loops==0 or prevCounter != self.faceCounter) and loops < self.nInter:
                            loops+=1
                            genX = self.model.decode_first_stage(img_n)
                            img_transformed = torch.unsqueeze(self.input2Clf(genX, device, self.clfSet[clfPtr].crop_size), 0)
                            # Back to useful stuff
                            if self.anchorPic != "":
                                curEmbed = self.clfSet[clfPtr].returnEmbedding(img_transformed)
                                target = cos_similarity(anchor, curEmbed)
                                gradient += torch.autograd.grad(target, img_n)[0]
                                
                            else:
                                if self.addLabel:
                                    attackLabel = torch.zeros(self.classCount).to(device)
                                    attackLabel[self.baseLabel] = 1
                                    logits = self.clfSet[clfPtr](img_transformed, attackLabel)
                                else:
                                    logits = self.clfSet[clfPtr](img_transformed)

                                clfPtr = (clfPtr + 1) % len(self.clfSet)
                                log_probs = F.log_softmax(logits, dim=-1)
                                if target_label is None:
                                    # Dodging
                                    if not self.indirectDodge:
                                        tar_label = torch.ones_like(label).to(device) * self.baseLabel
                                    else:
                                        tar_label = self.get_target_label(logits, label, device)
                                else:
                                    # Impersonation
                                    tar_label = torch.ones_like(label).to(device) * target_label
                                
                                selected = log_probs[range(len(logits)), tar_label]
                                gradient += torch.autograd.grad(selected.sum(), img_n)[0]

                    gradient = gradient / self.nInter
                    # print(torch.norm(gradient))
                    # Gradient should be the same size as img

                    if scaled:
                        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
                        # print(sqrt_one_minus_alphas[index])
                        img = img + dodgeWeight * sqrt_one_minus_at * 10 * s * gradient.float()
                    else:
                        # No f formula
                        img = img + dodgeWeight * s * gradient.float()
                
                # For returning the initial
                if initial:
                    return gradient, False, False
                
                # Evaluating the final result
                imperson_success_num = 0
                dodge_success_num = 0
                prevCounter = self.faceCounter
                gradient = torch.zeros_like(img).to(device)
                loops=0
                
                if self.successThresh <= 0:
                    return img, img_transformed, True # returns just successful attack images

                while loops==0 or prevCounter != self.faceCounter:
                    with torch.no_grad():
                        loops+=1
                        # img_n = img.detach().requires_grad_(True)
                        img_n = img.detach()
                        genX = self.model.decode_first_stage(img_n) # image transformation from latent code
                        # print(f"For {self.faceCounter}, {self.faceSet[self.faceCounter]}")
                        img_transformed = torch.unsqueeze(self.input2Clf(genX, device, self.clfSet[0].crop_size),0)
                        # genX is the current generated image
                        logits = self.clfSet[0](img_transformed.to(device))
                        log_probs = F.log_softmax(logits, dim=-1)
                        pred = torch.argmax(log_probs, dim=1)  # [B]

                        if self.anchorPic != "":
                            cur_imperson_success = 0
                            cur_dodge_success = 0
                            curEmbed = self.clfSet[clfPtr].returnEmbedding(img_transformed)
                            # diff = torch.subtract(anchor, curEmbed)
                            # target = torch.sum(torch.square(diff), dim=1)
                            target = cos_similarity(anchor, curEmbed)
                            
                            if (target.item() < self.anchorThresh and target_label is None) or (target.item() > self.anchorThresh and target_label is not None):
                                cur_imperson_success =1
                                cur_dodge_success = 1
                                
                        else:
                            # Dodging
                            cur_dodge_success = (pred != label).sum().item()

                            if target_label is not None:
                                # Impersonating
                                tar_label = torch.ones_like(label).to(device) * target_label
                                selected = log_probs[range(len(logits)), tar_label]
                                tmpGradient = torch.autograd.grad(selected.sum(), img_n)[0] # adversarial guidance
                                cur_imperson_success = (pred == tar_label).sum().item()
                                gradient = gradient + tmpGradient
                            else:
                                cur_imperson_success = 0

                    dodge_success_num += cur_dodge_success
                    imperson_success_num += cur_imperson_success

                # If attack is successful
                print(f"Dodging Success {dodge_success_num} / {loops}")
                print(f"Impersonation Success {imperson_success_num} / {loops}")
                if (target_label is not None and imperson_success_num>=math.floor(loops*self.successThresh)) or (target_label is None and dodge_success_num>=math.floor(loops*self.successThresh)):
                    return img, img_transformed, True # returns just successful attack images
                
                # To try to make attack more powerful
                innerK += 1

                # Line 12 of pseudocode
                finDodge = 1
                if target_label is None:
                    finDodge = -1
                pri_img = pri_img + finDodge * a * gradient.float()
            
        img_transformed = img_transformed.to(device) 
        # Note we should never reach the bottom return unless the attack fails
        return img, img_transformed, False

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec