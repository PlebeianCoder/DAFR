import os
import torch
import warnings
from torch import nn
from torchvision import transforms
# import torchvision.utils as vutils
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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

warnings.filterwarnings('ignore')

scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, min_lr=1e-6, mode='min'
)


def train_stymask(cfg):
    seed = cfg['seed']
    fix_randon_seed(seed)

    # checkpoint path
    style_model_path = cfg['style_model_path']

    # loss parameters
    tv_lambda = cfg['tv_lambda']
    l1_lambda = cfg['l1_lambda']
    percep_lambda = cfg['percep_lambda']
    style_lambda = cfg['style_lambda']
    dist_lambda = cfg['dist_lambda']

    # data setting
    data_path = cfg['data_path']
    train_set = cfg['train_set']
    target_identity = cfg['target_identity']
    target_img = cfg['target_img']
    img_size = cfg['img_size']

    # training parameters
    batch_size = cfg['batch_size']
    num_epoch = cfg['num_epoch']
    model_lr = cfg['model_lr']
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = cfg['pin_memory']
    num_workers = cfg['num_workers']

    # other parameters
    vgg_mean = cfg['vgg_mean']
    vgg_std = cfg['vgg_std']
    temperature = cfg['temperature']

    # print("Overall Configurations:")
    # print(cfg)

    # save name
    save_folder = f'./checkpoint/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    print(f'model_lr: {model_lr}, weight_lr: {weight_lr}, dist_lambda: {dist_lambda}, '
          f't: {temperature}, l1: {l1_lambda}\n backbone: {backbone_name}, head: {head_name}, '
          f'dist_threshold: {dist_threshold}\n target: {target_img}, train_set:{train_set}')

    # load style model
    attack_checkpoint = torch.load(style_model_path)
    attack_model = attack_checkpoint['model'].to(device)
    original_checkpoint = torch.load(style_model_path)
    original_model = original_checkpoint['model'].eval().to(device)

    # load face model
    model_path = f'{face_model_path}/{backbone_name}_{head_name}.pth'
    checkpoint = torch.load(model_path)
    # print(checkpoint)
    face_model = checkpoint['backbone']
    face_head = checkpoint['head']
    face_model = face_model.eval().to(device)
    face_head = face_head.eval().to(device)
    vgg = Vgg16().to(device).eval()

    # BN
    face_bn = Batch_Normalization(mean=img_mean, std=img_std, device=device)
    vgg_bn = Batch_Normalization(mean=vgg_mean, std=vgg_std, device=device)

    # data
    my_loader = get_MyData(batch_size=batch_size, mode=train_set, pin_memory=pin_memory,
                           num_workers=num_workers, target_identity=target_identity, shuffle=False)

    # optimizing setting
    optimizer_model = optim.Adam(attack_model.parameters(), lr=model_lr, amsgrad=True)
    scheduler_model = scheduler_factory(optimizer_model)
    weights = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).to(device)
    weights.requires_grad = True
    optimizer_weight = optim.SGD([weights], lr=weight_lr)
    scheduler_weight = scheduler_factory(optimizer_weight)

    # target img
    target_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    target_path = f'{data_path}/{target_identity}/{target_img}'
    target_face_img = target_transform(Image.open(target_path)).unsqueeze(0).to(device)
    print(target_path)
    with torch.no_grad():
        print(torch.min(target_face_img))
        print(torch.max(target_face_img))
        model_output = face_model(face_bn.norm(target_face_img)).to(device)
        print(model_output)
        print(model_output.size())
        true_output = face_head(model_output, None)
        print(true_output)
        print(true_output.size())
        print(true_output.argmax())
        exit()
        target_embedding_ori = l2_norm(face_model(face_bn.norm(target_face_img))).to(device)
    

    # loss
    l1_creterion = nn.L1Loss().to(device)
    total_variation = TotalVariation(device).to(device)

    # style imgs setting
    transform_style = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    n_styles = 10
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
    original_pattern = target_transform(Image.open('./mask_face/masks/textures/23.jpg').convert('RGB')).unsqueeze(0)

    for i in range(10):
        weights = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).to(device)
        weights[i] = 1
        style_attack_pattern = original_model(original_pattern.to(device), weights.unsqueeze(0), True).to(device)
        save_image(style_attack_pattern, f'./results/{i}-.jpg')

    # early stop
    early_stop = EarlyStopping(patience=7, init_patch=original_pattern)

    training_saved_path = "./results"
    if not os.path.exists(training_saved_path):
        os.mkdir(training_saved_path)

    epoch_length = len(my_loader)
    train_losses_epoch = []

    for epoch in range(num_epoch):
        imgs_all, loss_tv_all, loss_l1_all, loss_dist_all, loss_percep_all, loss_style_all = 0, 0., 0., 0., 0., 0.
        train_loss = 0.0
        lr = optimizer_model.param_groups[0]['lr']
        # lr_weight = optimizer_weight.param_groups[0]['lr']
        attack_model.train()
        # i = 0
        print("Hello")
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
                print("pre save")
                save_image(
                    real_style_mask,
                    f'{training_saved_path}/{target_identity}_benign_mask_{backbone_name}_{head_name}_{idx}_{epoch}.png'
                )
            print("post save")
            print(f"In {idx}")
            style_attack_mask = style_attack_pattern * uv_mask
            save_image(
                style_attack_mask,
                f'{training_saved_path}/{target_identity}_attack_mask_{backbone_name}_{head_name}_{idx}_{epoch}.png'
            )

            preds = location_extractor(face_imgs).to(device)
            style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True).to(device)
            style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)
            
            save_image(
                style_masked_face,
                f'{training_saved_path}/{target_identity}_masked_{backbone_name}_{head_name}_{idx}_{epoch}.png'
            )

            # TV LOSS
            loss_tv = total_variation(style_attack_mask) * tv_lambda
            # L1 Loss
            loss_l1 = l1_creterion(style_attack_pattern, real_style_pattern) * l1_lambda

            attack_embedding = l2_norm(face_model(face_bn.norm(style_masked_face)))
            diff = torch.subtract(attack_embedding, target_embedding)
            loss_dist = torch.sum(torch.square(diff), dim=1) * dist_lambda

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
            total_loss = loss_tv + loss_l1 + loss_dist + loss_perceptual + loss_style

            optimizer_model.zero_grad()
            optimizer_weight.zero_grad()

            total_loss.backward(torch.ones_like(total_loss))

            optimizer_weight.step()
            optimizer_model.step()

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

        # save models
        state = {
            'model': attack_model,
            'weights': weights,
            'temperature': temperature,
        }
        saved_path = f'{save_folder}/{target_identity}_attack_model.pth'
        torch.save(state, saved_path)
        if early_stop(train_losses_epoch[-1], epoch):
            break

        scheduler_weight.step(train_losses_epoch[-1])
        scheduler_model.step(train_losses_epoch[-1])


if __name__ == "__main__":
    train_stymask(cfg)
