'''
Tool functions
'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow import image
from keras.losses import mean_squared_error

subfig_scale = 64
scale = 512
subfig_num = (scale // subfig_scale)**2


def rebuild_pic_3_channel(one_pic):
    return np.concatenate(
        [
            np.concatenate(one_pic[i:i + scale // subfig_scale], axis=1)
            for i in range(0, subfig_num, scale // subfig_scale)
        ],
        axis=0,
    )


def psnr_pred(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)


def ssim_pred(y_true, y_pred):
    return image.ssim(y_true, y_pred, max_val=1.0)


def show_pic(model, clean, noise, n):
    model.compile(optimizer="Adam", loss=mean_squared_error, metrics=[psnr_pred, ssim_pred])
    noisy_pic = rebuild_pic_3_channel(noise[n * subfig_num:(n + 1) * subfig_num])
    clean_pic = rebuild_pic_3_channel(clean[n * subfig_num:(n + 1) * subfig_num])
    predict = model.predict(noise[n * subfig_num:(n + 1) * subfig_num], verbose=1)
    pred_pic = rebuild_pic_3_channel(predict)

    plt.figure(figsize=(16, 10))
    plt.subplot(131), plt.imshow(clean_pic)
    plt.subplot(132), plt.imshow(noisy_pic)
    plt.subplot(133), plt.imshow(pred_pic)
    plt.show()


def save_pic(model, clean, noise, n, path):
    model.compile(optimizer="Adam", loss=mean_squared_error, metrics=[psnr_pred, ssim_pred])
    noisy_pic = rebuild_pic_3_channel(noise[n * subfig_num:(n + 1) * subfig_num])
    clean_pic = rebuild_pic_3_channel(clean[n * subfig_num:(n + 1) * subfig_num])
    predict = model.predict(noise[n * subfig_num:(n + 1) * subfig_num], verbose=1)
    pred_pic = rebuild_pic_3_channel(predict)

    plt.figure(figsize=(16, 10))
    plt.subplot(131), plt.imshow(clean_pic)
    plt.subplot(132), plt.imshow(noisy_pic)
    plt.subplot(133), plt.imshow(pred_pic)
    plt.savefig(path)
    plt.close()
