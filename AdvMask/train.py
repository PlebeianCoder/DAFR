import sys
import os
    
import random
from pathlib import Path
import pickle

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

import utils
import losses
from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from utils import EarlyStopping, get_patch, ourPreprocess

from advfaceutil.datasets.faces import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead
from advfaceutil.recognition.mobilefacenet import MobileFaceNet

import warnings
warnings.simplefilter('ignore', UserWarning)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device), flush=True)

testingAcc = True


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AdversarialMask:
    def __init__(self, config, embedders, anchor_path, output_path, to_save, initial_patch, tv_weight=5e-2, doImpersonation=False):
        self.config = config
        set_random_seed(seed_value=self.config.seed)
        self.doImpersonation = doImpersonation

        self.train_no_aug_loader, self.train_loader = utils.get_train_loaders(self.config)
        
        self.embedders = embedders
        self.tv_weight = tv_weight

        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
        self.total_variation = TotalVariation(device).to(device)
        self.dist_loss = losses.get_loss(self.config)

        self.train_losses_epoch = []
        self.train_losses_iter = []
        self.dist_losses = []
        self.tv_losses = []
        self.val_losses = []

        self.target_embedding = {}
        self.target_embedding["DATA"] = [torch.load(anchor_path, map_location=device)]
        self.best_patch = None
        self.to_save = to_save
        self.output_path = output_path
        self.initial_patch = initial_patch
    
    def change_outpath(self, output_path):
        self.output_path = output_path

    def train(self, nonAdv=False, given_initial=None):
        
        if given_initial is None:
            adv_patch_cpu = utils.get_patch(self.initial_patch)
        else:
            adv_patch_cpu = given_initial
        if nonAdv:
            save_image(adv_patch_cpu, self.output_path)
            return adv_patch_cpu

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True, maximize=self.doImpersonation)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(current_dir=self.config.current_dir, patience=self.config.es_patience, init_patch=adv_patch_cpu, maximizing=self.doImpersonation)
        epoch_length = len(self.train_loader)

        for epoch in range(self.config.epochs):
            train_loss = 0.0
            dist_loss = 0.0
            tv_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, dist-loss: {:.6}, tv-loss: {:.6}, lr: {:.6}'
            for i_batch, (img_batch, _, cls_id) in progress_bar:
                (b_loss, sep_loss), vars = self.forward_step(img_batch, adv_patch_cpu, cls_id)

                train_loss += b_loss.item()
                dist_loss += sep_loss[0].item()
                tv_loss += sep_loss[1].item()

                optimizer.zero_grad()
                b_loss.backward()
                optimizer.step()

                adv_patch_cpu.data.clamp_(0, 1)

                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  dist_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  optimizer.param_groups[0]["lr"]))
                self.train_losses_iter.append(train_loss / (i_batch + 1))
                if i_batch + 1 == epoch_length:
                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses_epoch[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"], ))
                del b_loss
                torch.cuda.empty_cache()
            if early_stop(self.train_losses_epoch[-1], adv_patch_cpu, epoch):
                self.best_patch = adv_patch_cpu
                break

            scheduler.step(self.train_losses_epoch[-1])
        self.best_patch = early_stop.best_patch
        if self.to_save:
            self.save_final_objects()
        
        save_image(self.best_patch, self.output_path)
        return self.best_patch

    def loss_fn(self, patch_embs, tv_loss, cls_id):
        distance_loss = torch.empty(0, device=device)
        
        for i in range(len(self.target_embedding["DATA"])):

            if not (torch.isnan(self.target_embedding["DATA"][i]).any() or torch.isnan(patch_embs["DATA"]).any()):

                distance = self.dist_loss(patch_embs["DATA"], self.target_embedding["DATA"][i])
                single_embedder_dist_loss = torch.mean(distance).unsqueeze(0)
                distance_loss = torch.cat([distance_loss, single_embedder_dist_loss], dim=0)
            else:
                print(f"FAILED ON {i}")
        distance_loss = self.config.dist_weight * distance_loss.mean()
        tv_loss = self.tv_weight * tv_loss
        if self.doImpersonation:
            total_loss = distance_loss - tv_loss
        else:
            total_loss = distance_loss + tv_loss
        return total_loss, [distance_loss, tv_loss]

    def forward_step(self, img_batch, adv_patch_cpu, cls_id):
        img_batch = img_batch.to(device)
        adv_patch = adv_patch_cpu.to(device)
        cls_id = cls_id.to(device)

        make_112 = transforms.Resize(size=(112, 112), interpolation=InterpolationMode.BICUBIC)

        small_batch = make_112(img_batch)
        preds = self.location_extractor(small_batch)

        img_batch_applied = self.fxz_projector(small_batch, preds, adv_patch, do_aug=self.config.mask_aug)

        large_batch = ourPreprocess(make_112(img_batch_applied))

        patch_embs = {}

        if large_batch.ndim <= 3:
            large_batch = torch.unsqueeze(large_batch, 0)
        patch_embs["DATA"] = self.embedders.returnEmbedding(large_batch)

        tv_loss = self.total_variation(adv_patch)
        loss = self.loss_fn(patch_embs, tv_loss, cls_id)

        return loss, [img_batch, adv_patch, img_batch_applied, patch_embs, tv_loss]

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses_epoch.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

    def save_final_objects(self):
        alpha = transforms.ToTensor()(Image.open('prnet/new_uv.png').convert('L'))
        final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))
        final_patch_img.save(self.config.current_dir + '/final_results/final_patch.png', 'PNG')
        new_size = tuple(self.config.magnification_ratio * s for s in self.config.img_size)
        transforms.Resize(new_size)(final_patch_img).save(self.config.current_dir + '/final_results/final_patch_magnified.png', 'PNG')
        torch.save(self.best_patch, self.config.current_dir + '/final_results/final_patch_raw.pt')


        self.location_extractor = self.location_extractor.to(device)
        self.fxz_projector = self.fxz_projector.to(device)
        self.best_patch = self.best_patch.to(device)
        i = 0
        w=0
        cos_sim = CosineSimilarity()
        progress_bar = tqdm(enumerate(self.train_no_aug_loader), desc='Final')

        for i_batch, (img_batch, _, cls_id) in progress_bar:
            i+=1
            make_112 = transforms.Resize(size=(112, 112), interpolation=InterpolationMode.BICUBIC)
            small_batch = make_112(img_batch).to(device)
            preds = self.location_extractor(small_batch)
            img_batch_applied = self.fxz_projector(small_batch, preds, self.best_patch, do_aug=self.config.mask_aug)

            large_batch = ourPreprocess(make_112(img_batch_applied)).to(device)
            # save_image(large_batch, f"outputs/check{i}.png")
            if large_batch.ndim <= 3:
                large_batch = torch.unsqueeze(large_batch, 0)
            output = self.embedders.returnEmbedding(large_batch)
            cs = cos_sim(output, self.target_embedding["DATA"][0]).item()
            if cs <= 0.2:
                w+=1
            print(f"For {i}, goes to {cs}")

        if not self.doImpersonation:
            print(f"Final Dodging success: is {w} out of {i}, {(w/i)*100}%")
        else:
            print(f"Final Impersonation success: is {w} out of {i}, {(w/i)*100}%")


        with open(self.config.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses_epoch, fp)
        with open(self.config.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.config.current_dir + '/losses/dist_losses', 'wb') as fp:
            pickle.dump(self.dist_losses, fp)
        with open(self.config.current_dir + '/losses/tv_losses', 'wb') as fp:
            pickle.dump(self.tv_losses, fp)


def main():
    mode = 'targeted'

    config = patch_config_types[mode]("attack_dir", ["ReeseWitherspoon"])
    print('Starting train...', flush=True)

    dataset = FaceDatasets["PUBFIG"]
    size = dataset.get_size("LARGE")

    embedders = MobileFaceNet.construct(
        FaceDatasets["VGGFACE2"], FaceDatasets["VGGFACE2"].get_size("LARGE"), training=False, device=device,
        weights_directory=Path("mfn_weight")
    )
    adv_mask = AdversarialMask(config,
         embedders,
         "anchor_path",
         "outputs_name.png",
         False,
         "random",
         0.35
        )
    adv_mask.train()
    print('Finished train...', flush=True)


if __name__ == '__main__':
    main()
