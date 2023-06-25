import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage import img_as_ubyte
import torch

def save_chart(epochs, train_list, val_list, save_path, name=''):
    x = np.arange(epochs)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)
    lns1 = ax.plot(x, train_list, 'b', label='train {}'.format(name))
    lns2 = ax.plot(x, val_list, 'r', label='val {}'.format(name))
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]

    ax.legend(lns, labs, loc='upper right')
    ax.set_xlabel("Epochs")
    ax.set_ylabel(name)

    plt.savefig(save_path)
    plt.close()

def heatmap_postprocess(feat_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * feat_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    return heatmap

def img_fusion(image, heatmap):
    cam = heatmap + np.float32(image)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return image, cam

def print_activation_map(result_path, epoch, input_batch, foreground_batch, label_batch):
    for idx, l in enumerate(label_batch):
        if l == 1.0:
            choice_idx = idx
            break

    input_image = input_batch[choice_idx].permute(1,2,0).numpy()
    foreground = foreground_batch[choice_idx].permute(1,2,0).numpy()
    # background = background_batch[choice_idx].permute(1,2,0).numpy()
    # pred = pred_batch[choice_idx].numpy()
    # act_map = (act_map - act_map.min()+1e-5) / (act_map.max() - act_map.min()+1e-5)
    # act_map = act_map*pred
    if not os.path.exists(os.path.join(result_path, f"epoch_{epoch}")):
        os.makedirs(os.path.join(result_path, f"epoch_{epoch}"))
    foreground = heatmap_postprocess(foreground)
    input_image, mix_image = img_fusion(input_image, foreground)
    io.imsave(os.path.join(result_path, f"epoch_{epoch}", "input.jpg"), img_as_ubyte(input_image), check_contrast=False)
    io.imsave(os.path.join(result_path, f"epoch_{epoch}", "foreground.jpg"), img_as_ubyte(foreground), check_contrast=False)
    io.imsave(os.path.join(result_path, f"epoch_{epoch}", "mix_image.jpg"), img_as_ubyte(mix_image), check_contrast=False)
    # io.imsave(os.path.join(result_path, f"epoch_{epoch}", "background.jpg"), img_as_ubyte(background), check_contrast=False)