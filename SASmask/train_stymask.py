import os
import glob
import torch
import warnings
from torch import nn
from torch.nn import CosineSimilarity
from torchvision import transforms
# import torchvision.utils as vutils
from pathlib import Path
from PIL import Image
# from skimage.metrics import structural_similarity as compare_ssim
from vgg import Vgg16
import torch.nn.functional as F

from config import cfg
from dataset import get_MyData
from utils import Batch_Normalization, fix_randon_seed, l2_norm, gram_matrix
from torch import optim
from utils import EarlyStopping,get_landmark_detector
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from StyleModel.network import StyleModel
from torchvision.utils import save_image

warnings.filterwarnings('ignore')

scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, min_lr=1e-6, mode='min'
)

class CosLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cos_sim = CosineSimilarity()

    def forward(self, emb1, emb2):
        cs = self.cos_sim(emb1, emb2)
        # print(cs)
        return (cs + 1) / 2

def train_stymask(img_class, target_class, all_classes, face_model, gen_path,
                    uv_path, style_path, dist_weight, l1_weight, percept_weight, tv_weight, style_weight, num_epochs, model_lr, out_dir,
                    device,threshold, anchorPath="", withStyle=True, toSave=False):
    seed = cfg['seed']
    fix_randon_seed(seed)

    # For the base class images
    base_identity = all_classes[img_class]
    if target_class is not None:
        target_identity = all_classes[target_class]
    else:
        target_identity = None

    # checkpoint path
    style_model_path = cfg['style_model_path']

    # loss parameters
    tv_lambda = tv_weight
    l1_lambda = l1_weight
    percep_lambda = percept_weight
    style_lambda = style_weight
    # dist_lambda = cfg['dist_lambda']
    dist_lambda = dist_weight

    # data setting
    data_path = cfg['data_path']
    train_set = cfg['train_set']
    # target_identity = cfg['target_identity']
    target_img = cfg['target_img']
    img_size = cfg['img_size']

    # training parameters
    batch_size = cfg['batch_size']
    num_epoch = num_epochs
    # model_lr = cfg['model_lr']
    model_lr = model_lr
    weight_lr = cfg['weight_lr']
    lr_step = cfg['lr_step']
    lr_gamma = cfg['lr_gamma']

    # face model setting
    face_model_path = cfg['face_model_path']
    backbone_name = cfg['backbone_name']
    head_name = cfg['head_name']
    dist_threshold = cfg['dist_threshold']
    img_mean = cfg['img_mean']
    img_std = cfg['img_std']

    # gpu setting
    pin_memory = cfg['pin_memory']
    num_workers = cfg['num_workers']

    # other parameters
    vgg_mean = cfg['vgg_mean']
    vgg_std = cfg['vgg_std']
    temperature = cfg['temperature']

    # print("Overall Configurations:")
    # print(cfg)

    # save name
    rstr = "r18"
    if "r34" in anchorPath:
        rstr="r34"
    elif "r50" in anchorPath:
        rstr="r50"
    elif "r100" in anchorPath:
        rstr="r100"
    elif "fted100" in anchorPath:
        rstr ="fted100" 
    elif "clip" in anchorPath:
        rstr = "farl"
    elif "mfn" in anchorPath:
        rstr = "mfn"

    save_folder = f'./checkpoint/benchmark/{rstr}/{rstr}_{base_identity}_to_{target_identity}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # load style model
    attack_checkpoint = torch.load(style_model_path)
    attack_model = attack_checkpoint['model'].to(device)
    original_checkpoint = torch.load(style_model_path)
    original_model = original_checkpoint['model'].eval().to(device)

    # load face model
    # model_path = f'{face_model_path}/{backbone_name}_{head_name}.pth'
    # checkpoint = torch.load(model_path)
    # face_model = checkpoint['backbone']
    # face_model = face_model.eval().to(device)

    face_model = face_model.eval().to(device)
    vgg = Vgg16().to(device).eval()

    # BN
    face_bn = Batch_Normalization(mean=img_mean, std=img_std, device=device)
    vgg_bn = Batch_Normalization(mean=vgg_mean, std=vgg_std, device=device)

    # data
    my_loader = get_MyData(batch_size=batch_size, test_paths=gen_path, shuffle=False)

    # optimizing setting
    n_styles = 10
    optimizer_model = optim.Adam(attack_model.parameters(), lr=model_lr, amsgrad=True, maximize=target_class is not None)
    scheduler_model = scheduler_factory(optimizer_model)
    weights = torch.zeros((n_styles)).to(device)
    weights.requires_grad = True
    if withStyle:
        optimizer_weight = optim.SGD([weights], lr=weight_lr, maximize=target_class is not None)
        scheduler_weight = scheduler_factory(optimizer_weight)

    # target img
    target_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    # target_path = f'{data_path}/{target_identity}/{target_img}'
    # target_face_img = target_transform(Image.open(target_path)).unsqueeze(0).to(device)
    
    target_embedding_ori = torch.load(anchorPath, map_location=device)

    # loss
    l1_creterion = nn.L1Loss().to(device)
    total_variation = TotalVariation(device).to(device)

    # style imgs setting
    transform_style = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
   
    style_imgs = torch.stack(
        [transform_style(Image.open(f'./data/style_images/{i}.jpg').convert('RGB')).unsqueeze(0).to(device) for
         i in range(n_styles)]
    )
    style_imgs_norm = vgg_bn.norm(style_imgs.clone())

    face_landmark_detector = get_landmark_detector(
        landmark_detector_type='mobilefacenet', device=device
    )
    location_extractor = LandmarkExtractor(device, face_landmark_detector, (112, 112)).to(device)
    fxz_projector = FaceXZooProjector(device=device, img_size=(112, 112), patch_size=(112, 112)).to(device)

    # optimizing setting
    uv_mask = transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L')).unsqueeze(0).to(device)
    original_pattern = target_transform(Image.open(style_path).convert('RGB')).unsqueeze(0)

    # early stop
    early_stop = EarlyStopping(patience=7, init_patch=original_pattern, maximizing=target_class is not None)
    # early_stop = EarlyStopping(patience=20, init_patch=original_pattern, maximizing=target_class is not None)

    training_saved_path = "./results"
    if not os.path.exists(training_saved_path):
        os.mkdir(training_saved_path)

    epoch_length = len(my_loader)
    train_losses_epoch = []
    print(f"FIRST WEIGHTS: {weights}")
    big_transform = transforms.Compose([
        transforms.Resize(512),
    ])

    cos_sim = CosLoss()

    for epoch in range(num_epoch):
        imgs_all, loss_tv_all, loss_l1_all, loss_dist_all, loss_percep_all, loss_style_all = 0, 0., 0., 0., 0., 0.
        train_loss = 0.0
        success = 0
        count = 0
        lr = optimizer_model.param_groups[0]['lr']
        if withStyle:
            lr_weight = optimizer_weight.param_groups[0]['lr']
        attack_model.train()
        # i = 0
        for idx, [face_imgs, mask_bins, mask_imgs] in enumerate(my_loader):
            with torch.no_grad():
                target_embedding = target_embedding_ori.repeat(face_imgs.size(0), 1)
                face_imgs, mask_bins, mask_imgs = face_imgs.to(device), mask_bins.to(device), mask_imgs.to(device)

            original_pattern = original_pattern.to(device)

            batch_weights = weights.unsqueeze(0).repeat(face_imgs.size(0), 1).to(device)
            batch_weights = torch.softmax(batch_weights / temperature, dim=1)

            style_attack_pattern = attack_model(original_pattern, batch_weights, True).to(device)
            with torch.no_grad():
                real_style_pattern = original_model(original_pattern, batch_weights, True).to(device)
                real_style_mask = real_style_pattern * uv_mask
                # vutils.save_image(
                #     real_style_mask,
                #     f'{training_saved_path}/{target_identity}_benign_mask_{backbone_name}_{head_name}.png'
                # )
                # print("pre save")
                
                # save_image(
                #     real_style_mask,
                #     f'{training_saved_path}/{target_identity}_benign_mask_{backbone_name}_{head_name}_{idx}_{epoch}.png'
                # )
            # print("post save")
            # print(f"In {idx}")
            style_attack_mask = style_attack_pattern * uv_mask
            save_image(
                style_attack_mask,
                f'{training_saved_path}/{base_identity}To{target_identity}_attack_mask_{backbone_name}_{head_name}_{idx}_{epoch}.png'
            )

            preds = location_extractor(face_imgs).to(device)
            style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True).to(device)
            style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)
            
            save_image(
                style_masked_face,
                f'{training_saved_path}/{base_identity}To{target_identity}_masked_{backbone_name}_{head_name}_{idx}_{epoch}.png'
            )

            # TV LOSS
            loss_tv = total_variation(style_attack_mask) * tv_lambda
            # L1 Loss
            loss_l1 = l1_creterion(style_attack_pattern, real_style_pattern) * l1_lambda

            # attack_embedding = l2_norm(face_model(face_bn.norm(style_masked_face)))
            print(torch.max(style_masked_face))
            print(torch.min(style_masked_face))
            attack_embedding = face_model.returnEmbedding(style_masked_face)
            # diff = torch.subtract(attack_embedding, target_embedding)
            # loss_dist = torch.sum(torch.square(diff), dim=1) 
            loss_dist = cos_sim(attack_embedding, target_embedding)
            if (target_class is None and ((loss_dist.item()*2)-1) < threshold) or ((target_class is not None and ((loss_dist.item()*2)-1) > threshold)):
                success+=1
            count+=1
            loss_dist = loss_dist * dist_lambda
            print(f"DISTANCE: {((loss_dist/dist_lambda)*2)-1}")

            # Style Loss
            with torch.no_grad():
                real_style_feat = vgg(vgg_bn.norm(mask_imgs))
            stylized_feat = vgg(vgg_bn.norm(style_attack_mask))

            p1 = F.mse_loss(stylized_feat.relu1_2, real_style_feat.relu1_2)
            p2 = F.mse_loss(stylized_feat.relu2_2, real_style_feat.relu2_2)
            p3 = F.mse_loss(stylized_feat.relu3_3, real_style_feat.relu3_3)
            p4 = F.mse_loss(stylized_feat.relu4_3, real_style_feat.relu4_3)
            loss_perceptual = (p1 + p2 + p3 + p4) * percep_lambda

            if torch.abs(weights).sum().item() != 0:
                weights_onehot = torch.zeros_like(weights)
                weights_onehot[weights.argmax()] = 1.
                batch_weights_onehot = weights_onehot.unsqueeze(0).repeat(face_imgs.size(0), 1).to(device)
                style_imgs_inputs = torch.stack(
                    [style_imgs_norm[torch.nonzero(batch_weights_onehot)[i][1]].squeeze(0) for i in
                     range(face_imgs.size(0))], dim=0)

                with torch.no_grad():
                    style_feat = vgg(style_imgs_inputs)

                s1 = F.mse_loss(gram_matrix(stylized_feat.relu1_2), gram_matrix(style_feat.relu1_2))
                s2 = F.mse_loss(gram_matrix(stylized_feat.relu2_2), gram_matrix(style_feat.relu2_2))
                s3 = F.mse_loss(gram_matrix(stylized_feat.relu3_3), gram_matrix(style_feat.relu3_3))
                s4 = F.mse_loss(gram_matrix(stylized_feat.relu4_3), gram_matrix(style_feat.relu4_3))
                loss_style = (s1 + s2 + s3 + s4) * style_lambda
            else:
                loss_style = torch.zeros_like(loss_tv)

            # total loss
            if target_class is None:
                # Dodging so trying to minimize this
                total_loss = loss_tv + loss_l1 + loss_dist + loss_perceptual + loss_style
            else:
                # Impersonation so trying to maximize this
                total_loss = loss_dist - loss_tv - loss_l1  - loss_perceptual - loss_style

            optimizer_model.zero_grad()
            if withStyle:
                optimizer_weight.zero_grad()

            total_loss.backward(torch.ones_like(total_loss))

            if withStyle:
                optimizer_weight.step()
            optimizer_model.step()

            print(f"TV: {loss_tv}")
            print(f"L1: {loss_l1}")
            print(f"DIST: {loss_dist}")
            print(f"PERCEPT: {loss_perceptual}")
            print(f"STYLE: {loss_style}")
            print(f"Weights: {weights}")

            train_loss += total_loss.item()
            loss_tv_all += loss_tv.sum().item()
            loss_l1_all += loss_l1.sum().item()
            loss_percep_all += loss_perceptual.sum().item()
            loss_style_all += loss_style.sum().item()
            loss_dist_all += loss_dist.sum().item()
            imgs_all += face_imgs.size(0)

            if idx + 1 == epoch_length:
                train_losses_epoch.append(train_loss / imgs_all)

        print(
            f'Epoch {epoch}, lr_m: {lr}, train_loss: {train_loss / imgs_all:.4f}, tv_loss: {loss_tv_all / imgs_all:.4f},'
            f' l1: {loss_l1_all / imgs_all:.4f}, perc: {loss_percep_all / imgs_all:.4f}, style: {loss_style_all / imgs_all:.4f}'
            f' dist:{loss_dist_all / imgs_all:.4f},')

        print(f"Success Count: {success}/{count}, {(success/count)*100}%")
        if toSave:
            save_name= f"{out_dir}/sap_{base_identity}To{target_identity}_epoch{epoch}.png"
            style_attack_pattern = torch.round(torch.clamp((style_attack_pattern-style_attack_pattern.min())/(style_attack_pattern.max()-style_attack_pattern.min()), min=0.0, max=1.0)*255)/255.0
            save_image(style_attack_pattern, save_name)

            save_name= f"{out_dir}/sap_{base_identity}To{target_identity}_big_epoch{epoch}.png"
            style_attack_pattern = big_transform(style_attack_pattern)
            save_image(style_attack_pattern, save_name)

        # save models
        state = {
            'model': attack_model,
            'weights': weights,
            'temperature': temperature,
        }
        saved_path = f'{save_folder}/{base_identity}To{target_identity}_attack_model_{epoch}.pth'
        torch.save(state, saved_path)
        if early_stop(train_losses_epoch[-1], epoch):
            break
        
        if withStyle:
            scheduler_weight.step(train_losses_epoch[-1])
        scheduler_model.step(train_losses_epoch[-1])

    # Save style_attack_pattern
    save_name= f"{out_dir}/sap_{base_identity}To{target_identity}.png"
    style_attack_pattern = torch.round(torch.clamp((style_attack_pattern-style_attack_pattern.min())/(style_attack_pattern.max()-style_attack_pattern.min()), min=0.0, max=1.0)*255)/255.0
    save_image(style_attack_pattern, save_name)

    save_name= f"{out_dir}/sap_{base_identity}To{target_identity}_big.png"
    style_attack_pattern = big_transform(style_attack_pattern)
    save_image(style_attack_pattern, save_name)

    print("Attack Successfully Ended")

    return Path(save_name)