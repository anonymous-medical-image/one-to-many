import SimpleITK
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import torch
from skimage.metrics import structural_similarity as ssim
from torch import nn
from Src import models
import os
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.fit import Fit
from utils import utils
import data_loader
from vqgan.vq_gan.model import VQGAN
from scipy.io import loadmat
device = 'cuda'


def normalize(FLAIR, PET, T1, FLAIR_pre, PET_pre, T1_pre):
    data = [FLAIR, PET, T1, FLAIR_pre, PET_pre, T1_pre]
    for i in range(len(data)):
        data[i] = N(data[i])
    return data


def N(data):
    data -= np.min(data)
    data = data / (np.max(data) + 1e-3)
    return data * 255


class config:
    class model:
        embedding_dim = 8
        n_codes = 16384
        n_hiddens = 16
        downsample = [4, 4]
        norm_type = 'group'
        padding_type = 'replicate'
        no_random_restart = False
        restart_thres = 1.0
        num_groups = 32
        gan_feat_weight = 4
        disc_channels = 64
        disc_layers = 3
        disc_loss_type = 'vanilla'
        image_gan_weight = 1
        video_gan_weight = 1
        perceptual_weight = 4
        l1_weight = 4
        default_root_dir = './checkpoint'
        resume_from_checkpoint = None
        accumulate_grad_batches = 1
        max_steps = 100000
        max_epochs = 100
        precision = 16
        gradient_clip_val = 1
        gpus = 1
        lr = 3e-4
        discriminator_iter_start = 10000

    class dataset:
        image_channels = 1


def init_model(cfg):
    net_PET = VQGAN(cfg).to(device)
    key = net_PET.load_state_dict(
        torch.load(
            './vqgan/checkpoint_PET/lightning_logs/version_0/'
            'checkpoints/epoch=33-step=11999-train/recon_loss=0.04.ckpt')[
            'state_dict'])
    print(key)
    net_PET.eval()

    net_FLAIR = VQGAN(cfg).to(device)
    key = net_FLAIR.load_state_dict(
        torch.load(
            './vqgan/checkpoint_FLAIR/lightning_logs/version_0/'
            'checkpoints/epoch=28-step=9999-10000-train/recon_loss=0.15.ckpt')[
            'state_dict'])
    print(key)
    net_FLAIR.eval()

    net_T1 = VQGAN(cfg).to(device)
    key = net_T1.load_state_dict(
        torch.load(
            './vqgan/checkpoint_T1/lightning_logs/version_0/checkpoints'
            '/epoch=42-step=14999-train/recon_loss=0.15.ckpt')[
            'state_dict'])
    print(key)
    net_T1.eval()

    return net_PET, net_FLAIR, net_T1


# def psnr(target, ref):
#     target_data = np.array(target, dtype=np.float64)
#     ref_data = np.array(ref, dtype=np.float64)
#     diff = ref_data - target_data
#     diff = diff.flatten('C')
#     rmse = math.sqrt(np.mean(diff ** 2.))
#     eps = np.finfo(np.float64).eps
#     if rmse == 0:
#         rmse = eps
#     return 10 * math.log10(255.0 / rmse)

def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if rmse == 0:
        rmse = eps
    max_val = 255.0
    return 10 * math.log10((max_val ** 2) / rmse)



def SSIM(imageA, imageB):
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    (grayScore, diff) = ssim(imageA, imageB, full=True, data_range=255)

    return grayScore


def decoder(x, model):
    with torch.no_grad():
        y = (((x + 1.0) / 2.0) * (model.codebook.embeddings.max() -
                                  model.codebook.embeddings.min())) + \
            model.codebook.embeddings.min()
        y = model.decode(y, quantize=True)
    return y.cpu().numpy()[0, 0]


def run(slice, show=True):
    path_out = './result0206_1/' + index_name + '/' + slice.replace('.mat', '')
    os.makedirs(path_out, exist_ok=True)
    # change_data = ['PET', 'FLAIR', 'T1']
    CT2FLAIR_path = '../data_use_v1/data_use_v1/' + index_name + '/CT2FLAIR/' + slice
    CT2PET_path = '../data_use_v1/data_use_v1/' + index_name + '/CT2PET/' + slice
    CT2T1_path = '../data_use_v1/data_use_v1/' + index_name + '/CT2T1/' + slice

    CT2FLAIR = loadmat(CT2FLAIR_path)
    CT2PET = loadmat(CT2PET_path)
    CT2T1 = loadmat(CT2T1_path)

    CT = CT2FLAIR['CT']
    FLAIR_pre, FLAIR = gen.get_result(CT2FLAIR)
    FLAIR_pre = decoder(FLAIR_pre.to(device), net_FLAIR)
    PET_pre, PET = gen.get_result(CT2PET)
    PET_pre = decoder(PET_pre.to(device), net_PET)
    T1_pre, T1 = gen.get_result(CT2T1)
    T1_pre = decoder(T1_pre.to(device), net_T1)
    FLAIR_pre = nor.infer(FLAIR_pre, 'FLAIR')
    T1_pre = nor.infer(T1_pre, 'T1')
    PET_pre = nor.infer(PET_pre, 'PET')
    if FLAIR_pre.shape != FLAIR.shape:
        FLAIR_pre = cv2.resize(FLAIR_pre, FLAIR.shape[::-1])
    if T1_pre.shape != T1.shape:
        T1_pre = cv2.resize(T1_pre, T1.shape[::-1])
    if PET_pre.shape != PET.shape:
        PET_pre = cv2.resize(PET_pre, PET.shape[::-1])
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(CT), path_out + '/CT.nii')
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(FLAIR_pre), path_out + '/FLAIR_pre.nii')
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(PET_pre), path_out + '/PET_pre.nii')
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(T1_pre), path_out + '/T1_pre.nii')
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(FLAIR), path_out + '/FLAIR.nii')
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(PET), path_out + '/PET.nii')
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(T1), path_out + '/T1.nii')
    FLAIR, PET, T1, FLAIR_pre, PET_pre, T1_pre = normalize(FLAIR, PET, T1, FLAIR_pre, PET_pre, T1_pre)

    FLAIR_mse = mean_squared_error(FLAIR, FLAIR_pre)
    PET_mse = mean_squared_error(PET, PET_pre)
    T1_mse = mean_squared_error(T1, T1_pre)

    FLAIR_mae = mean_absolute_error(FLAIR, FLAIR_pre)
    PET_mae = mean_absolute_error(PET, PET_pre)
    T1_mae = mean_absolute_error(T1, T1_pre)

    FLAIR_ssim = SSIM(FLAIR, FLAIR_pre)
    PET_ssim = SSIM(PET, PET_pre)
    T1_ssim = SSIM(T1, T1_pre)

    FLAIR_psnr = psnr(FLAIR, FLAIR_pre)
    PET_psnr = psnr(PET, PET_pre)
    T1_psnr = psnr(T1, T1_pre)
    print(PET_pre == FLAIR_pre)

    tabel = PrettyTable(['', 'mse', 'mae', 'ssim', 'psnr'])
    tabel.add_row(['PET', PET_mse, PET_mae, PET_ssim, PET_psnr])
    tabel.add_row(['FLAIR', FLAIR_mse, FLAIR_mae, FLAIR_ssim, FLAIR_psnr])
    tabel.add_row(['T1', T1_mse, T1_mae, T1_ssim, T1_psnr])
    print(tabel)
    index_result.append([path_out,
                         PET_mse, PET_mae, PET_ssim, PET_psnr,
                         FLAIR_mse, FLAIR_mae, FLAIR_ssim, FLAIR_psnr,
                         T1_mse, T1_mae, T1_ssim, T1_psnr
                         ])

    if show:
        plt.figure(figsize=(12, 12))
        plt.subplot(221)
        plt.title('CT')
        plt.imshow(CT, 'gray')

        plt.subplot(222)
        plt.title('FLAIR')
        plt.imshow(FLAIR_pre, 'gray')

        plt.subplot(223)
        plt.title('PET')
        plt.imshow(PET_pre, 'gray')

        plt.subplot(224)
        plt.title('T1')
        plt.imshow(T1_pre, 'gray')
        plt.show()


if __name__ == "__main__":
    cfg = config()
    net_PET, net_FLAIR, net_T1 = init_model(cfg)
    args = utils.get_parse()
    args.training = False
    args.device = [device]
    input_shape = (args.image_size, args.image_size)
    resize = False
    show = True
    all_test = False
    args.sampling_timesteps = 1

    model_data = torch.load('./weights/weights.pth',
                            map_location=device)

    model = models.model_B(image_size=args.image_size, in_chans=8)

    try:
        model.load_state_dict(model_data['model_dict'])
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(model_data['model_dict'])
    model = model.to(device)
    model.eval()
    if device == 'cuda':
        model = nn.DataParallel(model)
    # print(model)
    gen = Fit(
        model,
        args,
        None,
        None,
        None,
    )
    nor = data_loader.nor()
    index_name = '001'
    # slice = '21'
    files = os.listdir('../data_use_v1/data_use_v1/' + index_name + '/CT2FLAIR')
    index_result = []
    for i in files:
        run(i, show=False)
    index_result = pd.DataFrame(index_result, columns=['name',
                                                       'PET_mse', 'PET_mae', 'PET_ssim', 'PET_psnr',
                                                       'FLAIR_mse', 'FLAIR_mae', 'FLAIR_ssim', 'FLAIR_psnr',
                                                       'T1_mse', 'T1_mae', 'T1_ssim', 'T1_psnr'
                                                       ])
    index_result.to_csv('./result0206_1/' + index_name + '/all_result0206_1.csv', index=False)
