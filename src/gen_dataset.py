import numpy as np
import argparse
import os
import glob
import numpy as np
from skimage import io
from skimage import img_as_ubyte
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
import nibabel as nib

def slice_image(volume, segmentation, case_id, out_base):
    vol = nib.load(volume).get_fdata()
    # print(vol.shape, vol.max(), vol.min())
    seg = nib.load(segmentation).get_fdata()
    # print(seg.shape, seg.max(), seg.min())
    norm_vol = (vol - vol.min()) / (vol.max() - vol.min())
    mask3d = np.where(seg > 0.0, 1.0, 0.0)
    # print("3D", np.mean(mask3d))
    for z in range(vol.shape[2]):
        mask = np.where(seg[..., z] > 0.0, 1.0, 0.0)  
        # print(np.mean(mask))
        if norm_vol.max() > 0.0:
            if np.mean(mask) > 0.01:
                io.imsave(os.path.join(out_base, "seg", "{}-{}.png".format(case_id, z)), img_as_ubyte(seg[..., z] / 4.0), check_contrast=False)
                io.imsave(os.path.join(out_base, "tumor", "{}-{}.jpg".format(case_id, z)), img_as_ubyte(norm_vol[..., z]), check_contrast=False)
            else:
                io.imsave(os.path.join(out_base, "normal", "{}-{}.jpg".format(case_id, z)), img_as_ubyte(norm_vol[..., z]), check_contrast=False)


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='MRI 3D to 2D image')
    parser.add_argument('-p', '--dataset_path', type=str, default='/hdd1/vincent18/BraTS2021/Task1', help='Dataset path')
    parser.add_argument('-o', '--out_path', type=str, default='/hdd1/vincent18/BraTS_patch', help='Out path')
    parser.add_argument('-m', '--modality', type=str, default='t1', help='Modality select [flair, t1, t1ce, t2]')
    parser.add_argument('-d', '--dataset', type=str, default='training', help='training, validate, or testing')
    args = parser.parse_args()

    assert args.modality in ['flair', 't1', 't1ce', 't2']

    path_base = os.path.join(args.dataset_path, args.dataset)
    print("Load Data From", path_base)

    out_base = os.path.join(args.out_path, args.modality, args.dataset)
    print("Save Data To", out_base)
    if not os.path.exists(out_base):
        os.makedirs(out_base)
    if not os.path.exists(os.path.join(out_base, "normal")):
        os.mkdir(os.path.join(out_base, "normal"))
    if not os.path.exists(os.path.join(out_base, "tumor")):
        os.mkdir(os.path.join(out_base, "tumor"))
    if not os.path.exists(os.path.join(out_base, "seg")):
        os.mkdir(os.path.join(out_base, "seg"))

    all_volume = glob.glob(os.path.join(path_base, '*', '*{}.nii.gz'.format(args.modality)))
    # print(all_volume)
    bar = tqdm(all_volume)
    for idx, volume in enumerate(bar):
        segmentation = os.path.join("/".join(volume.split(os.path.sep)[:-1]), volume.split(os.path.sep)[-1].replace(args.modality, "seg"))
        case_id = volume.split(os.path.sep)[-1].split("_")[1]
        # print(case_id)
        bar.set_description("Case ID: {}".format(case_id))
        # print('Process slide {}/{}: {}'.format(idx+1, len(all_volume), case_id))
        slice_image(volume, segmentation, case_id, out_base)


