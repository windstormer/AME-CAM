# import denseCRF
import os
import numpy as np
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)
from utils import *
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def gen_seg_mask(img, cam, img_name, result_path, output_hist=False):
    # threshold = 0.8
    # if threshold < 0.1:
    #     final_seg = np.zeros_like(cam)
    # else:
    # first_seg = np.where(cam>threshold, 1, 0)
    # cam = (cam - cam.min())/ (cam.max() - cam.min())
    # post_cam = DCRF(img, cam)
    # final_seg = np.argmax(post_cam, axis=0)
    final_seg = cam > 0.5

    return final_seg

def morphGAC(img, first_seg):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gimage = inverse_gaussian_gradient(gray)
    final_seg = morphological_geodesic_active_contour(gimage, 300,
                                        init_level_set=first_seg,
                                        smoothing=2, balloon=-1)
    
    return final_seg

def DCRF(img, first_seg):
    # img = np.asarray(img)
    # img = (img*255).astype(np.uint8)

    # first_seg = first_seg.astype(np.float32)
    # prob = np.repeat(first_seg[..., np.newaxis], 2, axis=2)
    # # prob = prob[:, :, :2]
    # prob[:, :, 0] = 1.0 - prob[:, :, 0]
    # w1    = 10.0  # weight of bilateral term
    # alpha = 10    # spatial std
    # beta  = 13    # rgb  std
    # w2    = 3.0   # weight of spatial term
    # gamma = 3     # spatial std
    # it    = 50   # iteration
    # param = (w1, alpha, beta, w2, gamma, it)
    # final_seg = denseCRF.densecrf(img, prob, param)


    img = np.asarray(img)
    img = (img*255).astype(np.uint8)

    first_seg = first_seg.astype(np.float32)
    prob = np.repeat(first_seg[np.newaxis, ...], 2, axis=0)
    # prob = prob[:, :, :2]
    prob[0, :, :] = 1.0 - prob[0, :, :]
    scale_factor = 1.0
    h, w = img.shape[:2]
    n_labels = 2
    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    img = np.ascontiguousarray(img.astype('uint8'))
    d.addPairwiseBilateral(sxy=10/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(10)
    final_seg = np.array(Q).reshape((n_labels, h, w))
    # print(final_seg.shape)
    return final_seg

