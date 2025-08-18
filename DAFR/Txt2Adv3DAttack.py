import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from logging import getLogger
import os
from torch.nn import CosineSimilarity


from ldm.util import instantiate_from_config
from ldm.models.diffusion.Txt2Adv3Dddim import Txt2Adv3DDDIMSampler
import torchvision
import torch.nn.functional as F
from pathlib import Path

from DiffUtil import loadForLDM, tensorBGR2RGB, load_model_from_config, l2_norm

torch.set_grad_enabled(False)

# Translating hyperparameters to paper notation: innerK is k, genS is number of time steps, s is s and steps is l.
# A random of seed of 2 is used for DAFR in the paper

# prompt: The text prompt of the style of the adversarial perturbation
# img_class: The index of the attacker in faceClasses
# target_class; The index of the target class we want to get too. If dodging then set to None
# clfSet: The list of networks we are trying to attack, only tested for length one as that is our focus
# name_set: The list of file names of face images we are generating for
# anchorPic: The path to the anchor, usually a "".pth" file computed by 
# desc: IMPORTANT as it is the omitted string in the alignment and is added to files when saved
# addLabel: was an idea experimented with during development NOT USED IN THE FINAL PAPER
# matchLighing: was an idea experimented with during development NOT USED IN THE FINAL PAPER
# seed: The seed of everything
# s: The step size of the adversarial guidance (very important and very application specific)
# steps: What proportion of steps we start adversarial guidance. So 0.8 means that 20% into the generation we start 
#        adversarial guidance. If steps is negative, then there will be no adversarial guidance at all. l in the paper.
# k: The number of iterations done in the adversarial loop before we give up (recommend 1). If the attack works, then return early.
# In the paper we only experiment with k=1 DO NOT BE CONFUSED WITH K IN THE PAPER.
# innerK: the number of adversarial attacks to do for every diffusion step, k in the paper
# finalK: the number of adversarial loops to do at the end, set to 0 for the paper
# scaled: A boolean on whether or not to scale the adversial guidance based on timestep rather than using f in the paper (recommend False)
# genS: Number of generative steps and is the number of timesteps
# a: The adversarial step length used on the in between each k iteration, never used as we only have k=1
# scale: The unconditional scaling, the higher the more like your prompt but it starts to get uncanny (VERY INFLUENTIAL)
# randInitial: When True the starting point will be random, and when False we will try to find a decent random adversarial point. We use False
# anchorThresh: The threshold for comparing cosine similarities against, DOES NOT EFFECT ATTACK
# successThresh: The threshold for success in the print out, DOES NOT EFFECT ATTACK
# faceClasses: The classes we are using and the list for which the indices for img_class and target_class refer
# nInter: is the number of gradient steps to add all at once, NOT USED IN THE PAPER AND SET TO 1
# model: The diffusion model being used
# config: The configuration file for the diffusion model
# outpath: Is which directory to save all the outputs
# Both the mesh path arguments were used in our PyTorch3D alignment, however are not used by the AdvMask alignment so they can be ignored
def texture_attack(prompt, img_class, target_class, clfSet, name_set, anchorPic="", addLabel=False, matchLighting=False,
         indirectDodge=False, desc="", seed=42,s=10, steps=0.5, k=1,innerK=5, finalK=0, scaled=False, successThresh=0.8,
         genS=200, a=20, scale=12, randInitial=True, outpath="output/", faceClasses = [], anchorThresh=0.2,
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
         toPrint=False, toSave=False, config = None, model = None, nInter=1,
         mesh_path="3Dobjs/Face_Mask/face_mask.obj", tmp_mesh_path="3Dobjs/Face_Mask/tmp.obj", toBGR=False, logger=None):

    if steps < 0 and os.path.exists(outpath + f"NonAdv_TEXTURE_{seed}_{desc}_Generated_to112.png"):
        return Path(outpath + f"NonAdv_TEXTURE_{seed}_{desc}_Generated_to112.png")

    # SEED EVERYTHING BABY EVERYWHERE
    seed_everything(seed)
    # Just error checking to stop a mistake I can imagine someone doing
    if genS > 250:
        logger.info("Number of generation steps too project (will cause crash) returning False")
        return False
    
    # Loading diffusion model, sampler and classifier
    if model is None and config is None:
        config = OmegaConf.load("configs/stable-diffusion/v2-inference.yaml")
    if model is None:
        logger.info("Trying to load diffusion")
        # with torch.no_cuda():
        torch.cuda.set_device(device)
        # torch.cuda.empty_cache()
        model = load_model_from_config(config, "diffusion_path", device)
        model = model.to(device)

    logger.info("Diffusion models loaded!!")

    cosine_similarity = CosineSimilarity()
    # Load base face image and convert it to BGR 
    faceSet = []
    pilSet = []
    for i in range(len(name_set)):
        try: # you probably dont need this its just safety (i am 99% sure you can remove this)
            rgbX, pilX =loadForLDM(name_set[i], device)
            if rgbX is not None:
                faceSet.append(rgbX)
                pilSet.append(pilX)
        except:
            pass
    sampler = Txt2Adv3DDDIMSampler(model, clfSet=clfSet, device=device, innerK=innerK, finalK=finalK, 
                baseLabel=img_class, nInter=min(nInter, len(faceSet)), classCount=len(faceClasses),
                anchorPic=anchorPic, addLabel=addLabel, doMatch=matchLighting, indirectDodge=indirectDodge, successThresh=successThresh,
                toBGR=toBGR, anchorThresh=anchorThresh)
    sampler.setMeshPaths(mesh_path, tmp_mesh_path)
    logger.info(f"Generating using {len(faceSet)} pictures")
    
    # Load images into a tensor
    sampler.setSet(faceSet,pilSet)

    ddim_eta = 1.0

    shape = [4, 512 // 8, 512 // 8]

    # TODO convert to texture
    # Need to find a good starting point (at least better than random)
    if not randInitial:
        if toPrint:
            logger.info("Finding good points")
        with torch.no_grad(), model.ema_scope():
                uc = None
                labs = torch.tensor([img_class]).to(device)
                if scale != 1.0:
                    uc = model.get_learned_conditioning([""])
                c = model.get_learned_conditioning([prompt])
                xT = sampler.findGoodInitial(S=genS//5,
                                            conditioning=c,
                                            batch_size=1,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta, 
                                            steps=steps, target_label=target_class,
                                            label=labs, s=s, a=a, k=k, seed=seed, scaled=scaled)
    else:
        xT = torch.randn((1, shape[0], shape[1], shape[2]), device=device)

    # Generation most of this can be thought as black box code if you don't have generative knowledge
    with torch.no_grad(), model.ema_scope():
            uc = None
            labs = torch.tensor([img_class]).to(device)
            if scale != 1.0:
                uc = model.get_learned_conditioning([""])
            c = model.get_learned_conditioning([prompt])
            samples, _, _ = sampler.sample(S=genS,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=ddim_eta, 
                                        x_T=xT,
                                        steps=steps, target_label=target_class,
                                        label=labs, s=s, a=a, k=k, seed=seed, scaled=scaled)

            # Generation over, time to save and check that it does fool the classifier
            generated = model.decode_first_stage(samples)[0]
            # This is done so that it is saved correctly
            # Extra 255 division may cause slight loss of accuracy but probably close enough
            generated = torch.round(torch.clamp((generated-generated.min())/(generated.max()-generated.min()), min=0.0, max=1.0)*255)/255.0

    endTarget = ""
    if target_class is None:
        endTarget = "None"
    else:
        endTarget = faceClasses[target_class]

    final_success = 0
    # To check if the attack worked
    if steps>=0:
        with torch.no_grad():
            for c in range(len(clfSet)):
                img_transformed = torch.unsqueeze(sampler.faceProcess(faceSet[0], generated, sampler.getUV(), sampler.getExtractor(),
                                    sampler.getProjector(), device),0)

                if anchorPic != "":
                    curEmbed = clfSet[c].returnEmbedding(img_transformed.to(device))

                    target = cosine_similarity(sampler.getAnchor(), curEmbed)
                    sucString = "Failure"
                    if (target.item() < anchorThresh and target_class is None) or (target.item() > anchorThresh and target_class is not None):
                        final_success+=1
                        sucString = "Success"
                else:
                    # should be (1, 3, crop_size, crop_size)
                    output = clfSet[c](img_transformed.to(device))
                    if toPrint:
                        logger.info(f"Final for classifier {c}")
                        logger.info(output)
                        logger.info(F.softmax(output, dim=-1))
                        logger.info(output.argmax())
                    sucString = "Failure"
                    
                    if (target_class is not None and output.argmax() == target_class) or (target_class is None and output.argmax() != img_class):
                        final_success+=1
                        sucString = "Success"

                # From here on it is just saving the images
                if toSave:
                    inpainted = img_transformed.view(3, 112, 112).to("cpu")
                    torchvision.utils.save_image(inpainted, outpath+f"{s}{desc}_TEXTURE_{faceClasses[img_class]}To{endTarget}_{sucString}_{seed}_112.png", normalize=False)
    
    if final_success >= len(clfSet) or steps<0:
        sucString = "Success"
    else:
        sucString = "Failure"
    
    if steps < 0:
        genPath = outpath + f"NonAdv_TEXTURE_{seed}_{desc}_Generated.png"
        path112 = outpath+f"NonAdv_TEXTURE_{seed}_{desc}_Generated_to112.png"
    else:
        genPath = outpath+f"{s}{desc}_TEXTURE_{faceClasses[img_class]}To{endTarget}_{sucString}_{seed}_Generated.png"
        path112 = outpath+f"{s}{desc}_TEXTURE_{faceClasses[img_class]}To{endTarget}_{sucString}_{seed}_Generated_to112.png"

    to112 = torchvision.transforms.Resize(112)
    torchvision.utils.save_image(generated, genPath,normalize=False)
    torchvision.utils.save_image(to112(generated), path112,normalize=False)

    # Texture returned
    return Path(path112)