import os
import torchvision.transforms as transforms
import dlib
import numpy as np
import cv2
import random
from imutils import face_utils
from PIL import Image

from mask_face.utils.aux_functions import shape_to_landmarks, rect_to_bb, get_six_points, get_avg_brightness, \
    download_dlib_model, change_brightness, get_avg_saturation, change_saturation
from mask_face.utils.read_cfg import read_cfg
from mask_face.utils.create_mask import texture_the_mask

# mask_the_face parameters and components
detector = dlib.get_frontal_face_detector()  # Huihui: 获取人脸检测器
path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(path_to_dlib_model):
    download_dlib_model()
predictor = dlib.shape_predictor(path_to_dlib_model)  # Huihui: 预测人脸关键点


# huihui
def get_mask_code():
    mask_type = 'cloth'

    texture_candidates = []
    for line in open('./mask_face/texture_paths.txt', 'r'):
        line = line.replace('\n', '')
        texture_candidates.append(line)

    mask_texture = random.choice(texture_candidates)
    return mask_type, mask_texture


def mask_face(image, six_points, angle, mask_type="surgical", mask_pattern=''):
    # Find the face angle
    threshold = 13
    if angle < -threshold:
        mask_type += "_right"
    elif angle > threshold:
        mask_type += "_left"
    w = image.shape[0]
    h = image.shape[1]

    cfg = read_cfg(config_filename="mask_face/masks/masks.cfg", mask_type=mask_type, verbose=False)
    
    img = cv2.imread(cfg.template, cv2.IMREAD_UNCHANGED)

    # Process the mask if necessary
    if mask_pattern:
        # Apply pattern to mask
        pattern_weight = 1
        img = texture_the_mask(img, mask_pattern, pattern_weight)
        # plt.imshow(img)
        # plt.show()

    mask_line = np.float32([cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d])
    # Warp the mask
    M, mask = cv2.findHomography(mask_line, six_points)
    dst_mask = cv2.warpPerspective(img, M, (h, w))
    mask = dst_mask[:, :, 3]

    image_face = image

    # Adjust Brightness
    mask_brightness = get_avg_brightness(img)
    img_brightness = get_avg_brightness(image_face)
    delta_b = 1 + (img_brightness - mask_brightness) / 255
    dst_mask = change_brightness(dst_mask, delta_b)

    # Adjust Saturation
    mask_saturation = get_avg_saturation(img)
    img_saturation = get_avg_saturation(image_face)
    delta_s = 1 - (img_saturation - mask_saturation) / 255
    dst_mask = change_saturation(dst_mask, delta_s)

    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask)
    out_img = cv2.add(img_bg, img_fg[:, :, 0:3])

    return out_img, mask, dst_mask


def cv2tensor(img_cv2):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_cv2)
    # img_Image = transforms.ToPILImage()(img_tensor)
    # plt.imshow(img_Image)
    # plt.show()
    return img_tensor


def tensor2Image(img_tensor):
    return transforms.ToPILImage()(img_tensor).convert('RGB')


def cv2Image(img_cv2):
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))


def mask_face_img(img_cv2, texture='random'):
    # img_Image = transforms.ToPILImage()(img_tensor)
    # img_cv2 = cv2.cvtColor(np.asarray(img_Image), cv2.COLOR_RGB2BGR)
    # # img_cv2 = cv2.cvtColor(img_Image, cv2.COLOR_RGB2BGR)
    #
    # # original_image = img_cv2.copy()
    # plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    # exit(0)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    # gray = img_cv2
    face_locations = detector(gray, 1)

    mask_type, mask_texture = get_mask_code()
    if texture != 'random':
        mask_texture = texture

    if len(face_locations) == 0:
        # print(f'image contains {len(face_locations)} faces!')
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # img_tensor = cvimg2tensorimg(img_cv2)
        #
        # diff_min = 2000000
        # path_min = 0
        # g = os.walk(r"./data/CASIA_MINI/")
        # for path, dir_list, file_list in g:
        #     for file_name in file_list:
        #         if '.jpg' in os.path.join(path, file_name):
        #             img_path2 = os.path.join(path, file_name)
        #             img2 = Image.open(img_path2)
        #             # img2 = cv2.imread(img_path2)
        #             # img_cv22 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #             img_tensor2 = transforms.ToTensor()(img2)
        #             img_tensor2 = transforms.Resize((160, 160))(img_tensor2)
        #             diff = torch.abs(img_tensor - img_tensor2).sum().cpu().item()
        #             if diff_min > diff:
        #                 diff_min = diff
        #                 path_min = img_path2
        # print(diff_min)
        # print(path_min)
        # img3 = cv2.imread(path_min)
        # img_cv3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        masked_img_Image = cv2Image(img_cv2)
        mask_bin_Image = cv2Image(img_cv2)
        mask_Image = cv2Image(img_cv2)
        return masked_img_Image, mask_bin_Image, mask_Image, len(face_locations)
    else:
        if len(face_locations) > 1:
            max_location = 0
            max_area = 0
            for face_location in face_locations:
                area = face_location.area()
                if area > max_area:
                    max_location = face_location
                    max_area = area
            face_locations = [max_location]

        for (_, face_location) in enumerate(face_locations):
            shape = predictor(gray, face_location)  # Huihui: 预测人脸关键点
            shape = face_utils.shape_to_np(shape)
            face_landmarks = shape_to_landmarks(shape)
            face_location = rect_to_bb(face_location)
            # draw_landmarks(face_landmarks, image)
            six_points_on_face, angle = get_six_points(face_landmarks, img_cv2)
            # print(f'\n{six_points_on_face}\n{angle}')
            masked_img_cv2, mask_bin_cv2, mask_cv2 = mask_face(img_cv2, six_points_on_face, angle, mask_type,
                                                               mask_texture)
            masked_img_Image = cv2Image(masked_img_cv2)
            mask_bin_Image = cv2Image(mask_bin_cv2)
            mask_Image = cv2Image(mask_cv2)
            return masked_img_Image, mask_bin_Image, mask_Image, 1
            # B, G, R = cv2.split(masked_img_cv2)
            # image_RGB = cv2.merge([R, G, B])
            # plt.imshow(image_RGB)  # 显示图片
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()
            # # B, G, R = cv2.split(mask_bin_cv2)
            # # image_RGB = cv2.merge([R, G, B])
            # plt.imshow(mask_bin_cv2)  # 显示图片
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()
