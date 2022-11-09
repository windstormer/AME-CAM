import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') 
import os
import numpy as np

def print_seg_contour(result_path, img, ground_truth, final_seg, img_name):
    save_path = os.path.join(result_path, img_name)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    # ax = axes.flatten()
    # ax[0].imshow(gray, cmap="gray")
    # ax[0].set_axis_off()
    # contour = ax[0].contour(final_seg, [0.5], colors='r')
    # contour.collections[0].set_label("Final Segment")
    # contour = ax[0].contour(ground_truth, [0.5], colors='g')
    # contour.collections[0].set_label("Ground-Truth")
    # ax[0].legend(loc="upper right", fontsize=6)
    # ax[0].set_title("Segmentation Comparison", fontsize=12)

    # ax[1].imshow(gray, cmap="gray")
    # ax[1].set_axis_off()
    # contour = ax[1].contour(first_seg, [0.5], colors='y')
    # contour.collections[0].set_label("First Seg")
    # # contour = ax[1].contour(second_seg, [0.5], colors='y')
    # # contour.collections[0].set_label("MorphGAC Seg")
    # contour = ax[1].contour(final_seg, [0.5], colors='r')
    # contour.collections[0].set_label("Final Seg")
    # ax[1].legend(loc="upper right", fontsize=6)
    # title = "Segmentation Contour Evolution"
    # ax[1].set_title(title, fontsize=12)
    # plt.savefig(os.path.join(save_path, "postprocess.png"))
    # plt.close()

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(gray, cmap="gray")
    plt.axis('off')
    contour = plt.contour(final_seg, [0.5], colors='r')
    contour.collections[0].set_label("Final Segment")
    contour = plt.contour(ground_truth, [0.5], colors='g')
    contour.collections[0].set_label("Ground-Truth")
    plt.legend(loc="upper right")
    plt.title("Segmentation Comparison", fontsize=12)
    plt.savefig(os.path.join(save_path, "postprocess.png"))
    plt.close()

    
def print_hist(result_path, img_pixel, threshold, img_name):
    save_path = os.path.join(result_path, img_name)
    flatten_img = img_pixel.flatten()
    fig = plt.figure(figsize=(8, 8))
    plt.hist(flatten_img, bins=50, color='c')
    plt.axvline(x=threshold, color='red')
    plt.title('Thres: {}'.format(threshold))
    plt.savefig(os.path.join(save_path, "hist.png"))
    plt.close()

def draw_area_hist(all_area, result_path, method):
    save_path = os.path.join(result_path, "{}_area.png".format(method))
    bin_x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    plt.hist(all_area, bins=bin_x, color='b', align='mid')
    plt.ylim(0,70)
    plt.xticks(bin_x)
    # plt.xlim(0,0.6)
    plt.xlabel('Normalized False Area')
    plt.ylabel('Count')
    # plt.title('Area of False Positive Cases ({})'.format(method), fontsize=14)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()