import math
import sys
from dataclasses import dataclass
from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import pathlib

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from kornia.losses import total_variation
from kornia.utils import create_meshgrid
from PIL import Image
from skimage import transform as trans
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from advfaceutil.recognition.processing.base import AugmentationOptions
from advfaceutil.recognition.processing.base import BoundingBox
from advfaceutil.recognition.processing.base import FaceDetectionResult
from advfaceutil.recognition.processing.base import FaceProcessor
from advfaceutil.recognition.processing.dlib import DlibFaceProcessor
from advfaceutil.recognition.processing.prnet.prnet import PRNet
from advfaceutil.recognition.processing.landmark_detection.face_alignment.face_alignment import (
    FaceAlignment,
)
from advfaceutil.recognition.processing.landmark_detection.face_alignment.face_alignment import (
    LandmarksType,
)
from advfaceutil.recognition.processing.landmark_detection.pytorch_face_landmark.models import (
    mobilefacenet,
)

# For the top ones

LOGGER = getLogger("advmask_face_processing")


def get_landmark_detector(landmark_detector_type="mobilefacenet", device="cuda:0"):
    if landmark_detector_type == "face_alignment":
        return FaceAlignment(LandmarksType._2D, device=str(device))
    elif landmark_detector_type == "mobilefacenet":
        model = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
        sd = torch.load(
            pathlib.Path(__file__).parent.resolve().as_posix() + "/landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar",
            map_location=device,
        )["state_dict"]
        model.load_state_dict(sd)
        return model


def render_cy_pt(vertices, new_colors, triangles, b, h, w, device):
    new_image, face_mask = render_texture_pt(
        vertices, new_colors, triangles, device, b, h, w
    )
    return face_mask, new_image


def get_mask_from_bb(h, w, device, box):
    points = torch.cartesian_prod(
        torch.arange(0, h, device=device), torch.arange(0, w, device=device)
    )
    c1 = points[:, 0] >= box[2]
    c2 = points[:, 0] <= box[3]
    c3 = points[:, 1] >= box[0]
    c4 = points[:, 1] <= box[1]
    mask = (c1 & c2 & c3 & c4).view(h, w)
    return mask


def get_unique_first_indices(inverse, unique_size):
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique_size).scatter_(0, inverse, perm)
    return perm


def get_image_by_vectorization_with_unique_small(
    bboxes, new_tri_depth, new_tri_tex, new_triangles, vertices, h, w, device
):
    depth_sorted, indices = torch.sort(new_tri_depth, descending=True)
    bb_sorted = torch.index_select(input=bboxes, dim=0, index=indices)
    texture_sorted = torch.index_select(input=new_tri_tex, dim=1, index=indices)
    bboxes_unique, inverse = torch.unique(bb_sorted, dim=0, return_inverse=True)
    uni_idx = get_unique_first_indices(inverse, bboxes_unique.size(0))
    depth_sorted = torch.index_select(input=depth_sorted, dim=0, index=uni_idx)
    texture_sorted = torch.index_select(input=texture_sorted, dim=1, index=uni_idx)

    points = torch.cartesian_prod(
        torch.arange(0, h, device=device), torch.arange(0, w, device=device)
    )

    points = points.unsqueeze(1).repeat(1, bboxes_unique.shape[0], 1)
    c1 = points[:, :, 0] >= bboxes_unique[:, 2]
    c2 = points[:, :, 0] <= bboxes_unique[:, 3]
    c3 = points[:, :, 1] >= bboxes_unique[:, 0]
    c4 = points[:, :, 1] <= bboxes_unique[:, 1]

    mask = (c1 & c2 & c3 & c4).view(h, w, -1)

    deep_depth_buffer = (
        torch.zeros([h, w, mask.shape[-1]], dtype=torch.int32, device=device) - 999999.0
    )
    dp = torch.where(mask, depth_sorted, deep_depth_buffer).argmax(dim=-1)

    color_img = torch.zeros((3, h, w), device=device)
    color_img = torch.where(
        (mask.sum(dim=-1) == 0), color_img, texture_sorted.T[dp].permute(2, 0, 1)
    )

    mask_img = torch.zeros((1, h, w), device=device)
    mask_img = torch.where(
        (mask.sum(dim=-1) == 0), mask_img, torch.ones(1, device=device)
    )
    return color_img, mask_img


def render_texture_pt(vertices, colors, triangles, device, b, h, w):
    tri_depth = (
        vertices[:, 2, triangles[0, :]]
        + vertices[:, 2, triangles[1, :]]
        + vertices[:, 2, triangles[2, :]]
    ) / 3.0
    tri_tex = (
        colors[:, :, triangles[0, :]]
        + colors[:, :, triangles[1, :]]
        + colors[:, :, triangles[2, :]]
    ) / 3.0

    umins = torch.max(
        torch.ceil(torch.min(vertices[:, 0, triangles], dim=1)[0]).type(torch.int),
        torch.tensor(0, dtype=torch.int),
    )
    umaxs = torch.min(
        torch.floor(torch.max(vertices[:, 0, triangles], dim=1)[0]).type(torch.int),
        torch.tensor(w - 1, dtype=torch.int),
    )
    vmins = torch.max(
        torch.ceil(torch.min(vertices[:, 1, triangles], dim=1)[0]).type(torch.int),
        torch.tensor(0, dtype=torch.int),
    )
    vmaxs = torch.min(
        torch.floor(torch.max(vertices[:, 1, triangles], dim=1)[0]).type(torch.int),
        torch.tensor(h - 1, dtype=torch.int),
    )

    masks = (umins <= umaxs) & (vmins <= vmaxs)

    image = torch.zeros((b, 3, h, w), device=device)
    face_mask = torch.zeros((b, 1, h, w), device=device)
    for i in range(b):
        bboxes = (
            torch.masked_select(
                torch.stack([umins[i], umaxs[i], vmins[i], vmaxs[i]]), masks[i]
            )
            .view(4, -1)
            .T
        )
        new_tri_depth = torch.masked_select(tri_depth[i], masks[i])
        new_tri_tex = torch.masked_select(tri_tex[i], masks[i]).view(3, -1)
        new_triangles = torch.masked_select(triangles, masks[i]).view(3, -1)
        image[i], face_mask[i] = get_image_by_vectorization_with_unique_small(
            bboxes, new_tri_depth, new_tri_tex, new_triangles, vertices[i], h, w, device
        )

    return image, face_mask


# nn_modules
class LandmarkExtractor(nn.Module):
    def __init__(self, device, face_landmark_detector, img_size):
        super(LandmarkExtractor, self).__init__()
        self.device = device
        self.face_align = face_landmark_detector
        self.img_size_width = img_size[1]
        self.img_size_height = img_size[0]

    def forward(self, img_batch):
        if isinstance(self.face_align, FaceAlignment):
            with torch.no_grad():
                points = self.face_align.get_landmarks_from_batch(img_batch * 255)
            single_face_points = [landmarks[:68] for landmarks in points]
            preds = torch.tensor(single_face_points, device=self.device)
        else:
            with torch.no_grad():
                preds = self.face_align(img_batch)[0].view(img_batch.shape[0], -1, 2)
                preds[..., 0] = preds[..., 0] * self.img_size_height
                preds[..., 1] = preds[..., 1] * self.img_size_width
                preds = preds.type(torch.int)
        return preds


class FaceXZooProjector(nn.Module):
    def __init__(self, device, img_size, patch_size):
        super(FaceXZooProjector, self).__init__()

        self.prn = PRN("prnet/prnet.pth", device)

        self.img_size_width = img_size[1]
        self.img_size_height = img_size[0]
        self.patch_size_width = patch_size[1]
        self.patch_size_height = patch_size[0]

        self.device = device
        self.uv_mask_src = (
            transforms.ToTensor()(Image.open("prnet/new_uv.png").convert("L"))
            .to(device)
            .unsqueeze(0)
        )

        image_info = torch.nonzero(self.uv_mask_src, as_tuple=False)
        left, _ = torch.min(image_info[:, 3], dim=0)
        right, _ = torch.max(image_info[:, 3], dim=0)
        self.mask_half_width = ((right - left) / 2) + 5
        top, _ = torch.min(image_info[:, 2], dim=0)
        bottom, _ = torch.max(image_info[:, 2], dim=0)
        self.mask_half_height = (bottom - top) / 2
        self.patch_bbox = self.get_bbox(self.uv_mask_src)

        # self.make_112 = transforms.Resize(size=(112, 112), interpolation=InterpolationMode.BICUBIC)
        self.uv_face_src = (
            transforms.ToTensor()(Image.open("prnet/uv_face_mask.png").convert("L"))
            .to(device)
            .unsqueeze(0)
        )
        # self.uv_face_src = self.make_112(self.uv_face_src)
        self.triangles = torch.from_numpy(
            np.loadtxt("prnet/triangles.txt").astype(np.int64)
        ).T.to(device)
        self.minangle = -5 / 180 * math.pi
        self.maxangle = 5 / 180 * math.pi
        self.min_trans_x = -0.05
        self.max_trans_x = 0.05
        self.min_trans_y = -0.05
        self.max_trans_y = 0.05
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.05

    def forward(
        self,
        img_batch,
        landmarks,
        adv_patch,
        uv_mask_src=None,
        do_aug=False,
        is_3d=False,
    ):
        pos_orig, vertices_orig = self.get_vertices(img_batch, landmarks)
        # print(pos_orig.size())
        # print(pos_orig[:, 0].size())
        # print(img_batch.size())
        # print(landmarks.size())
        # print(self.uv_face_src.size())
        texture_img = kornia.geometry.remap(
            img_batch, map_x=pos_orig[:, 0], map_y=pos_orig[:, 1], mode="nearest"
        )
        # print(texture_img.size())
        texture_img = texture_img * self.uv_face_src

        adv_patch = adv_patch.expand(img_batch.shape[0], -1, -1, -1)
        if not is_3d:
            adv_patch_other = self.align_patch(adv_patch, landmarks)
            texture_patch = (
                kornia.geometry.remap(
                    adv_patch_other,
                    map_x=pos_orig[:, 0],
                    map_y=pos_orig[:, 1],
                    mode="nearest",
                )
                * self.uv_face_src
            )
            uv_mask_src = torch.where(
                texture_patch.sum(dim=1, keepdim=True) != 0,
                torch.ones(1, device=self.device),
                torch.zeros(1, device=self.device),
            )
        else:
            uv_mask_src = uv_mask_src.repeat(adv_patch.shape[0], 1, 1, 1)
            if adv_patch.shape[2] != 256:
                adv_patch = F.interpolate(adv_patch, (256, 256))
                uv_mask_src = F.interpolate(uv_mask_src, (256, 256))
                # adv_patch = F.interpolate(adv_patch, (112, 112))
                # uv_mask_src = F.interpolate(uv_mask_src, (112, 112))
            texture_patch = adv_patch

        if do_aug:
            texture_patch, uv_mask_src = self.augment_patch(texture_patch, uv_mask_src)

        new_texture = texture_patch * uv_mask_src + texture_img * (1 - uv_mask_src)

        new_colors = self.prn.get_colors_from_texture(new_texture)

        face_mask, new_image = render_cy_pt(
            vertices_orig,
            new_colors,
            self.triangles,
            img_batch.shape[0],
            self.img_size_height,
            self.img_size_width,
            self.device,
        )
        face_mask = torch.where(
            torch.floor(face_mask) > 0,
            torch.ones(1, device=self.device),
            torch.zeros(1, device=self.device),
        )
        # print(img_batch.size())
        # print(new_image.size())
        # print(face_mask.size())
        new_image = img_batch * (1 - face_mask) + (new_image * face_mask)
        new_image.data.clamp_(0, 1)

        return new_image

    def align_patch(self, adv_patch, landmarks):
        batch_size = landmarks.shape[0]
        src_pts = self.patch_bbox.repeat(batch_size, 1, 1)

        landmarks = landmarks.type(torch.float32)
        max_side_dist = torch.maximum(
            landmarks[:, 33, 0] - landmarks[:, 2, 0],
            landmarks[:, 14, 0] - landmarks[:, 33, 0],
        )
        max_side_dist = torch.where(
            max_side_dist < self.mask_half_width, self.mask_half_width, max_side_dist
        )

        left_top = torch.stack(
            (
                landmarks[:, 33, 0] - max_side_dist,
                landmarks[:, 62, 1] - self.mask_half_height,
            ),
            dim=-1,
        )
        right_top = torch.stack(
            (
                landmarks[:, 33, 0] + max_side_dist,
                landmarks[:, 62, 1] - self.mask_half_height,
            ),
            dim=-1,
        )
        left_bottom = torch.stack(
            (
                landmarks[:, 33, 0] - max_side_dist,
                landmarks[:, 62, 1] + self.mask_half_height,
            ),
            dim=-1,
        )
        right_bottom = torch.stack(
            (
                landmarks[:, 33, 0] + max_side_dist,
                landmarks[:, 62, 1] + self.mask_half_height,
            ),
            dim=-1,
        )
        dst_pts = torch.stack([left_top, right_top, left_bottom, right_bottom], dim=1)

        tform = kornia.geometry.homography.find_homography_dlt(src_pts, dst_pts)
        cropped_image = kornia.geometry.warp_perspective(
            adv_patch,
            tform,
            dsize=(self.img_size_width, self.img_size_height),
            mode="nearest",
        )

        grid = create_meshgrid(112, 112, False, device=self.device).repeat(
            batch_size, 1, 1, 1
        )

        for i in range(batch_size):
            bbox_info = self.get_bbox(cropped_image[i : i + 1])
            left_top = bbox_info[:, 0]
            right_top = bbox_info[:, 1]
            x_center = (right_top[:, 0] - left_top[:, 0]) / 2
            target_y = torch.mean(torch.stack([landmarks[i, 0, 1], landmarks[i, 0, 1]]))
            max_y_left = torch.clamp_min(-(target_y - left_top[:, 1]), 0)
            start_idx_left = min(int(left_top[0, 0].item()), self.img_size_width)
            end_idx_left = min(int(start_idx_left + x_center), self.img_size_width)
            offset = torch.zeros_like(grid[i, :, start_idx_left:end_idx_left, 1])
            dropoff = 0.97
            for j in range(offset.shape[1]):
                offset[:, j] = (
                    max_y_left - ((j * max_y_left) / offset.shape[1])
                ) * dropoff

            grid[i, :, start_idx_left:end_idx_left, 1] = (
                grid[i, :, start_idx_left:end_idx_left, 1] + offset
            )

            target_y = torch.mean(
                torch.stack([landmarks[i, 16, 1], landmarks[i, 16, 1]])
            )
            max_y_right = torch.clamp_min(-(target_y - right_top[:, 1]), 0)
            end_idx_right = min(int(right_top[0, 0].item()), self.img_size_width) + 1
            start_idx_right = min(int(end_idx_right - x_center), self.img_size_width)
            offset = torch.zeros_like(grid[i, :, start_idx_right:end_idx_right, 1])
            for idx, col in enumerate(reversed(range(offset.shape[1]))):
                offset[:, col] = (
                    max_y_right - ((idx * max_y_right) / offset.shape[1])
                ) * dropoff
            grid[i, :, start_idx_right:end_idx_right, 1] = (
                grid[i, :, start_idx_right:end_idx_right, 1] + offset
            )

        # print(cropped_image.size())
        # print(grid.size())
        cropped_image = kornia.geometry.remap(
            cropped_image, map_x=grid[..., 0], map_y=grid[..., 1], mode="nearest"
        )
        # save_image(cropped_image, "cropped.png")
        return cropped_image

    def get_bbox(self, adv_patch):
        image_info = torch.nonzero(adv_patch, as_tuple=False).unsqueeze(0)
        left, _ = torch.min(image_info[:, :, 3], dim=1)
        right, _ = torch.max(image_info[:, :, 3], dim=1)
        top, _ = torch.min(image_info[:, :, 2], dim=1)
        bottom, _ = torch.max(image_info[:, :, 2], dim=1)
        width = right - left
        height = bottom - top
        # crop image
        center = torch.stack(
            [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=-1
        )
        left_top = torch.stack(
            (center[:, 0] - (width / 2), center[:, 1] - (height / 2)), dim=-1
        )
        right_top = torch.stack(
            (center[:, 0] + (width / 2), center[:, 1] - (height / 2)), dim=-1
        )
        left_bottom = torch.stack(
            (center[:, 0] - (width / 2), center[:, 1] + (height / 2)), dim=-1
        )
        right_bottom = torch.stack(
            (center[:, 0] + (width / 2), center[:, 1] + (height / 2)), dim=-1
        )
        src_pts = torch.stack([left_top, right_top, left_bottom, right_bottom], dim=1)
        return src_pts

    def get_vertices(self, image, face_lms):
        pos = self.prn.process(image, face_lms)
        vertices = self.prn.get_vertices(pos)
        return pos, vertices

    def augment_patch(self, adv_patch, uv_mask_src):
        contrast = (
            self.get_random_tensor(adv_patch, self.min_contrast, self.max_contrast)
            * uv_mask_src
        )
        brightness = (
            self.get_random_tensor(adv_patch, self.min_brightness, self.max_brightness)
            * uv_mask_src
        )
        noise = (
            torch.empty(adv_patch.shape, device=self.device).uniform_(-1, 1)
            * self.noise_factor
            * uv_mask_src
        )
        adv_patch = adv_patch * contrast + brightness + noise
        adv_patch.data.clamp_(0.000001, 0.999999)
        merged = torch.cat([adv_patch, uv_mask_src], dim=1)
        merged_aug = self.apply_random_grid_sample(merged)
        adv_patch = merged_aug[:, :3]
        uv_mask_src = merged_aug[:, 3:]
        return adv_patch, uv_mask_src

    def get_random_tensor(self, adv_patch, min_val, max_val):
        t = torch.empty(adv_patch.shape[0], device=self.device).uniform_(
            min_val, max_val
        )
        t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t = t.expand(-1, adv_patch.size(-3), adv_patch.size(-2), adv_patch.size(-1))
        return t

    def apply_random_grid_sample(self, face_mask):
        theta = torch.zeros(
            (face_mask.shape[0], 2, 3), dtype=torch.float, device=self.device
        )
        rand_angle = torch.empty(face_mask.shape[0], device=self.device).uniform_(
            self.minangle, self.maxangle
        )
        theta[:, 0, 0] = torch.cos(rand_angle)
        theta[:, 0, 1] = -torch.sin(rand_angle)
        theta[:, 1, 1] = torch.cos(rand_angle)
        theta[:, 1, 0] = torch.sin(rand_angle)
        theta[:, 0, 2].uniform_(self.min_trans_x, self.max_trans_x)  # move x
        theta[:, 1, 2].uniform_(self.min_trans_y, self.max_trans_y)  # move y
        grid = F.affine_grid(theta, list(face_mask.size()))
        augmented = F.grid_sample(face_mask, grid, padding_mode="reflection")
        return augmented


class PRN:
    """Process of PRNet.
    based on:
    https://github.com/YadiraF/PRNet/blob/master/api.py
    """

    def __init__(self, model_path, device):
        self.resolution = 256
        self.MaxPos = self.resolution * 1.1
        self.face_ind = np.loadtxt("prnet/face_ind.txt").astype(np.int32)
        self.triangles = np.loadtxt("prnet/triangles.txt").astype(np.int32)
        self.net = PRNet(3, 3)
        self.net.load_state_dict(torch.load(model_path))
        self.device = device
        self.net.to(device).eval()

    def get_bbox_annot(self, image_info):
        left, _ = torch.min(image_info[..., 0], dim=1)
        right, _ = torch.max(image_info[..., 0], dim=1)
        top, _ = torch.min(image_info[..., 1], dim=1)
        bottom, _ = torch.max(image_info[..., 1], dim=1)
        return left, right, top, bottom

    def preprocess(self, img_batch, image_info):
        left, right, top, bottom = self.get_bbox_annot(image_info)
        center = torch.stack(
            [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=-1
        )

        old_size = (right - left + bottom - top) / 2
        size = (old_size * 1.5).type(torch.int32)

        # crop image
        left_top = torch.stack(
            (center[:, 0] - (size / 2), center[:, 1] - (size / 2)), dim=-1
        )
        right_top = torch.stack(
            (center[:, 0] + (size / 2), center[:, 1] - (size / 2)), dim=-1
        )
        left_bottom = torch.stack(
            (center[:, 0] - (size / 2), center[:, 1] + (size / 2)), dim=-1
        )
        right_bottom = torch.stack(
            (center[:, 0] + (size / 2), center[:, 1] + (size / 2)), dim=-1
        )
        src_pts = torch.stack([left_top, right_top, left_bottom, right_bottom], dim=1)
        dst_pts = torch.tensor(
            [
                [0, 0],
                [self.resolution - 1, 0],
                [0, self.resolution - 1],
                [self.resolution - 1, self.resolution - 1],
            ],
            dtype=torch.float32,
            device=self.device,
        ).repeat(src_pts.shape[0], 1, 1)

        # wrap
        tform = kornia.geometry.transform.imgwarp.get_perspective_transform(
            src_pts, dst_pts
        )
        cropped_image = kornia.geometry.warp_perspective(
            img_batch, tform, dsize=(self.resolution, self.resolution)
        )
        # save_image(cropped_image, "PRNcrop.png")
        return cropped_image, tform

    def process(self, img_batch, image_info):
        cropped_image, tform = self.preprocess(img_batch, image_info)

        cropped_pos = self.net(cropped_image)
        cropped_vertices = (cropped_pos * self.MaxPos).view(cropped_pos.shape[0], 3, -1)

        z = cropped_vertices[:, 2:3, :].clone() / tform[:, :1, :1]
        cropped_vertices[:, 2, :] = 1

        vertices = torch.bmm(torch.linalg.inv(tform), cropped_vertices)
        vertices = torch.cat((vertices[:, :2, :], z), dim=1)

        pos = vertices.reshape(vertices.shape[0], 3, self.resolution, self.resolution)
        return pos

    def get_vertices(self, pos):
        all_vertices = pos.view(pos.shape[0], 3, -1)
        vertices = all_vertices[..., self.face_ind]
        return vertices

    def get_colors_from_texture(self, texture):
        all_colors = texture.view(texture.shape[0], 3, -1)
        # print(self.face_ind)
        # print(all_colors.size())
        # all_colors = torch.transpose(all_colors, 0, 1)
        # print(all_colors.size())
        # all_colors = all_colors.reshape(3, all_colors.size()[1]*all_colors.size()[2])
        # print(all_colors.size())
        # colors = all_colors[..., np.clip(self.face_ind, 0, all_colors.size()[1]-1)]
        colors = all_colors[..., self.face_ind]
        # print(colors.size())
        # colors = colors.reshape(3, -1)
        return colors


@dataclass
class AdvMaskAugmentationOptions(AugmentationOptions):
    """
    Options to control facial augmentation using Dlib.

    :ivar texture:
        The texture to apply to the facemask
    """

    texture = None


class AdvMaskFaceProcessor(FaceProcessor):
    def __init__(self) -> None:
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_landmark_detector = get_landmark_detector(
            landmark_detector_type="mobilefacenet", device=device
        )
        self.device = device
        self.location_extractor = LandmarkExtractor(
            device, face_landmark_detector, (112, 112)
        ).to(device)
        self.fxz_projector = FaceXZooProjector(
            device=device, img_size=(112, 112), patch_size=(112, 112)
        ).to(device)

        self.uv_mask = (
            transforms.ToTensor()(
                Image.open(
                    "prnet/new_uv.png"
                ).convert("L")
            )
            .unsqueeze(0)
            .to(device)
        )
        self.dlib = DlibFaceProcessor()

        self.tform = trans.SimilarityTransform()
        self.mtcnn = MTCNN(image_size=112, device=torch.device("cpu"))

    @classmethod
    def _validate_detection_result(
        cls, detection_result: Union[FaceDetectionResult, List[FaceDetectionResult]]
    ) -> bool:
        return True

    @staticmethod
    def _validate_augmentation_options(options: AugmentationOptions) -> bool:
        return True

    def detect_faces(self, image: np.ndarray):
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        if image.dtype != np.uint8:
            # Assume [0, 1]
            image = np.round(image * 255).astype(np.uint8)
        return ["face"]

    def _show_landmarks(
        self,
        image: np.ndarray,
        detections,
        colour: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 2,
    ):
        return image

    def _augment(self, image: np.ndarray, options, detections) -> np.ndarray:
        with torch.no_grad():

            if image.shape[2] != 3:
                image = np.transpose(image, (1, 2, 0))
            # print(image.shape)
            torch_aligned = torch.from_numpy(image).to(self.device)
            # print(torch_aligned.size())
            # To put channel first
            torch_aligned = torch.unsqueeze(torch_aligned.permute(2, 0, 1), 0)
            if torch.dtype != torch.float32:
                # print(f"Max: {torch.max(torch_aligned)}")
                torch_aligned = torch.clamp(
                    torch_aligned.to(torch.float32) / 255.0, min=0.0, max=1.0
                )

            resize112 = transforms.Resize(112)
            torch_aligned = resize112(torch_aligned)
            preds = self.location_extractor(torch_aligned)
            # print("preds")
            # print(preds)
            texture = options.texture[:, :, :3]
            # print(texture.shape)
            if texture.shape[2] != 3:
                texture = np.transpose(texture, (1, 2, 0))
            texture = np.transpose(
                cv2.resize(texture, (112, 112), interpolation=cv2.INTER_AREA), (2, 0, 1)
            )
            torch_texture = (
                torch.unsqueeze(torch.from_numpy(texture), 0).to(self.device)
                * self.uv_mask
            )
            torch_texture = torch.clamp(torch_texture / 255.0, min=0.0, max=1.0)

            final112 = self.fxz_projector(
                torch_aligned, preds, torch_texture, do_aug=True
            )

            final112 = final112.detach().cpu().numpy()
            # Back to numpy land
            # print(final112.shape)
            final112 = np.transpose(final112[0], (1, 2, 0))
            # print(final112.shape)

            final_return = cv2.resize(
                final112, (112, 112), interpolation=cv2.INTER_CUBIC
            )

            final_return = np.round(
                (final_return / (np.max(final_return) - np.min(final_return))) * 255
            ).astype(np.uint8)
            return final_return

    def _align(self, image: np.ndarray, crop_size: int, detections):
        # print(image.shape)
        if image.shape[0] == crop_size and image.shape[1] == crop_size:
            return image
        return self.dlib.align(
            image, crop_size, None
        )  # should already be aligned because of above
