import os
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import cv2

from mask_face.mask_functions import mask_face_img, cv2Image


def get_my_data(img_size=112, batch_size=64, train=1):
    dataset = MyDataset(img_size=img_size, train=train)
    my_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return my_loader


class MyDataset(data.Dataset):
    def __init__(self, img_size=112, train=2):
        self.root = r'./data/CASIA-WebFace'
        if train == 1:
            self.file_list = r'./data/CASIA-WebFace_100x5.txt'
        elif train == 2:
            self.file_list = r'./data/CASIA-WebFace_10x5.txt'
        elif train == 3:
            self.file_list = r'./data/CASIA-WebFace_5x1.txt'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.test_paths = []
        # 读取pairs文件
        with open(self.file_list) as f:
            pairs = f.read().splitlines()[:]

        # 将路径存储
        for i in range(len(pairs)):
            self.test_paths.append(pairs[i])

    def __getitem__(self, index):
        img_test = cv2.imread(self.test_paths[index])
        masked_img_test_Image, mask_bin_Image, mask_Image, face_num_test = mask_face_img(img_test)

        if face_num_test != 1:
            print(f'test image contains {face_num_test} faces!')
            print(os.path.join(self.root, self.test_paths[index]))

        img_test_Image = cv2Image(img_test)
        # img_pair_Image = cv2Image(img_pair)
        # img_test = Image.open(os.path.join(self.root, self.test_paths[index]))  # 读取第一张图
        # img_pair = Image.open(os.path.join(self.root, self.pair_paths[index]))  # 读取第二张图
        # imglist = [img_l, img_l.transpose(Image.Transpose.FLIP_LEFT_RIGHT), img_r,
        #            img_r.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]  # 翻转拼接
        # imglist = [img_test_Image, masked_img_Image, mask_bin_Image, img_pair_Image, masked_img_pair_Image,
        #            mask_bin_pair_Image]
        imglist = [img_test_Image, mask_bin_Image, mask_Image]
        # 图片预处理
        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])
            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.test_paths)

# if __name__ == '__main__':
#     batch_size = 64
#     # dataset = LFW(root, file_list, transform=data_transforms['val'])
#     dataset = LFW_dataset(img_size=112)
#     lfw_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
#     # print(f'Masked failure: test_img({dataset.test_err}), pair_img({dataset.pair_err})')
#     # print(len(lfw_loader))
#     # for batch_index, [test_imgs, masked_test_imgs, mask_bins, pair_imgs] in enumerate(lfw_loader):
#     for batch_index, data in enumerate(lfw_loader):
#         print(f'batch_index: {batch_index}')
#     #     start_index = batch_index * batch_size
#     #     # batch_test_paths = test_paths[start_index:start_index + batch_size]
#     #     # batch_pair_paths = pair_paths[start_index:start_index + batch_size]
#     #     test_imgs_unnorm = unnorm(test_imgs)
#     #     pair_imgs_unnorm = unnorm(pair_imgs)
#     #     for i in range(batch_size):
#     #         img1 = transforms.ToPILImage()(test_imgs_unnorm[i])
#     #         img2 = transforms.ToPILImage()(pair_imgs_unnorm[i])
#     #         plt.imshow(img1)
#     #         plt.show()
#     #         plt.imshow(img2)
#     #         plt.show()
#     #         exit(0)
# #         masked_img_tensor, mask_bin_tensor = mask_face_img(test_imgs_unnorm[i], 'test')
# #         masked_img_tensor2, mask_bin_tensor2 = mask_face_img(pair_imgs_unnorm[i], 'pair')
