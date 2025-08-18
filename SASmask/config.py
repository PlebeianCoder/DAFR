import torch
from parameters_dist import thresholds
import os

embedders_dict = {
    'resnet18': {
        'layers': [2, 2, 2, 2],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet18.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet18.pth')
            }
        }
    },
    'resnet34': {
        'layers': [3, 4, 6, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet34.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet34.pth')
            }
        }
    },
    'resnet50': {
        'layers': [3, 4, 14, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet50.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet50.pth')
            }
        }
    },
    'resnet100': {
        'layers': [3, 13, 30, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet100.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet100.pth')
            },
            'magface': {
                'weights_path': os.path.join('..', 'face_recognition', 'magface_torch', 'weights',
                                             'magface_resnet100.pth')
            }
        }
    }
}

cfg = dict(
    # random seed
    seed=123,

    # checkpoint path
    style_model_path='./checkpoint/style_models/style_model_epoch_0.pth',
    style_attack_path='./checkpoint/attack_models',
    face_model_path='./checkpoint/face_models',

    # data setting
    data_path='./data/lfw_resume',
    train_set='lfw_400_train',
    test_set='lfw_1000_test',
    img_size=112,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    vgg_mean=(0.485, 0.456, 0.406),
    vgg_std=(0.229, 0.224, 0.225),
    target_identity='Steve_Park',
    target_img='Steve_Park_0001.jpg',
    pin_memory=False,
    num_workers=8,

    # parameters
    tv_lambda=1e1,  # 10e0=0.11709
    l1_lambda=1e2,  # 10e1=0.92
    percep_lambda=1e-3,  # 10e-3=0.46624
    style_lambda=1e3,  # 10e3=0.82
    dist_lambda=1e2,  # 1 = 1.82

    # face model setting
    backbone_name='ResNet_50',
    head_name='ArcFace',
    dist_threshold=thresholds['ResNet_50']['ArcFace'],

    # training parameters
    batch_size=1,
    num_epoch=2000,
    model_lr=0.01,
    weight_lr=0.01,
    lr_step=50,
    lr_gamma=0.1,
    temperature=0.1,
)
