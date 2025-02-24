import glob
import os
import torch
import tqdm
from scipy.io import loadmat
from PIL import Image
import numpy as np
from torch.utils import data
import albumentations as A
import cv2
from transformers import BertTokenizer


def sobel(img):
    imgx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    imgy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    img = cv2.addWeighted(imgx, .5, imgy, .5, 0)
    return img


def read(folder, paths):
    files = []
    for path in paths:
        files.extend(glob.glob(folder + '/' + path + '/*/*.mat'))
    return files


def get_data_path():
    path_all = '../data_use_v1/data_use_v1'
    files = os.listdir(path_all)
    length = len(files) // 10
    train_paths = files[length:]
    test_paths = files[:length]
    train_files = read(path_all, train_paths)
    test_files = read(path_all, test_paths)
    return train_files, test_files


class nor:
    def __init__(self):
        super(nor, self).__init__()

    def nor_CT(self, x):
        x = (x - (-433.2372408001803)) / 620.9656933964563
        return x

    def unnor_CT(self, x):
        x = x * 620.9656933964563
        x += (-433.2372408001803)
        return x

    def nor_PET(self, x):
        x = (x - 1956.488629433893) / 2757.646811928385
        return x

    def unnor_PET(self, x):
        x = x * 2757.646811928385
        x += 1956.488629433893
        return x

    def nor_FLAIR(self, x):
        x = (x - 118.28367040564306) / 123.25517182401741
        return x

    def unnor_FLAIR(self, x):
        x = x * 123.25517182401741
        x += 118.28367040564306
        return x

    def nor_T1(self, x):
        x = (x - 34.17283045906093) / 36.18994578613209
        return x

    def unnor_T1(self, x):
        x = x * 36.18994578613209
        x += 34.17283045906093
        return x

    def forward(self, x, name):

        if name == 'CT':
            return self.nor_CT(x)
        elif name == 'PET':
            return self.nor_PET(x)
        elif name == 'FLAIR':
            return self.nor_FLAIR(x)
        elif name == 'T1':
            return self.nor_T1(x)
        else:
            print(name + 'is not in [''T1'', ''PET'', ''FLAIR'']')
            exit()

    def infer(self, x, name):

        if name == 'CT':
            return self.unnor_CT(x)
        elif name == 'PET':
            return self.unnor_PET(x)
        elif name == 'FLAIR':
            return self.unnor_FLAIR(x)
        elif name == 'T1':
            return self.unnor_T1(x)
        else:
            print(name + 'is not in [''T1'', ''PET'', ''FLAIR'']')
            exit()


class Dataset(data.Dataset):
    def __init__(self, imgs, shape, transform=False):
        self.use_transform = transform
        self.imgs = imgs
        self.input_shape = shape
        self.transform = A.Compose([
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=.3, alpha=120, sigma=120 * .05,
                               alpha_affine=120 * .03),
            A.ShiftScaleRotate(p=.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
        ])
        self.nor = nor()

    def read_data(self, img_path):
        Data = loadmat(img_path)
        txt_emb = Data['txt_emb'][0]
        x = Data['CT_EMB']
        # x = self.nor.forward(x, 'CT')
        name = Data['txt'][0]
        y = Data[name + '_EMB']
        # y = self.nor.forward(y, name)
        # print(x.shape, y.shape)

        # x = cv2.resize(x, self.input_shape)
        # y = cv2.resize(y, self.input_shape)
        # x = x[None]
        # y = y[None]
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        txt_emb = torch.from_numpy(txt_emb).type(torch.FloatTensor)
        return x, y, txt_emb

    def __getitem__(self, index):
        img_x, img_y, txt_emb = self.read_data(self.imgs[index])

        return img_y, img_x, txt_emb

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    bert_name = './bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    text = 'CT'
    input_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_tensors='pt'
    )
    print('text:\n', text)
    print('text字符数:', len(text))
    print('input_ids:\n', input_ids)
    print('input_ids大小:', input_ids.size())
