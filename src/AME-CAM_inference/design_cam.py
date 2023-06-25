from audioop import reverse
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np

from models import *
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
import cv2
import torch

from evaluation import *
from utils import *
from postprocess import *

class Design_CAM(object):
    def __init__(self, args, exp_name, encoder_model_type, gpuid):
        # encoder_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "encoder.pth")
        # decoder_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "decoder.pth")
        model_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "model.pth")
        score_model_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "score_model.pth")

        self.model_model_type = encoder_model_type

        if encoder_model_type == "Res18":
            model = Res18_Classifier().cuda()
        elif encoder_model_type == "Res50":
            model = Res50_Classifier().cuda()
        score_model = Res_Scoring().cuda()
        model.load_pretrain_weight(model_path)
        score_model.load_pretrain_weight(score_model_path)
        # state_dict_weights = torch.load(encoder_path)
        # model.load_state_dict(state_dict_weights, strict=False)
        # state_dict_weights = torch.load(decoder_path)
        # model.load_state_dict(state_dict_weights, strict=False)
        for param in model.parameters():
            param.requires_grad = False
        for param in score_model.parameters():
            param.requires_grad = False
        self.modality = args.modality
        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
            score_model = torch.nn.DataParallel(score_model, device_ids=gpuid)
        self.model = model.to('cuda')
        self.score_model = score_model.to('cuda')

        self.result_path = os.path.join(args.project_path, "results", exp_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
    
    def step(self, img):
        img = img.cuda()
        logits_collect, map_collect, MECAM_map = self.model(img)
        final_map, foreground, background, ame_map = self.score_model(img, map_collect)
    
        confidence = torch.sigmoid(torch.flatten(logits_collect[-1])).detach().cpu()
        norm_map = []
        for map in map_collect:
            norm_map.append(map.squeeze(1).squeeze(0).detach().cpu())

        return norm_map, final_map.detach().cpu(), confidence[0], MECAM_map.detach().cpu(), ame_map.detach().cpu()

    def run_selected_case(self, loader):
        self.model.eval()
        self.score_model.eval()
        
        log_path = os.path.join(self.result_path, "selected_case.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                if not os.path.exists(os.path.join(self.result_path, img_name, "feat_map")):
                    os.makedirs(os.path.join(self.result_path, img_name, "feat_map"))
                seg_image = seg_batch[0].permute(1, 2, 0).squeeze(2).numpy()
                map_collect, final_map, confidence, MECAM_map, ame_map = self.step(case_batch)
                input_image = case_batch[0].permute(1, 2, 0)
                norm_map, final_map, final_seg, MECAM_map, ame_map = self.CAM_algo(input_image, map_collect, final_map, MECAM_map, ame_map, img_name, confidence)
                # print(norm_map[0].shape)
                seg_gt = (seg_image*4).astype(np.uint8)

                whole_gt = np.where(seg_gt!=0, 1, 0)
                
                result = compute_seg_metrics(whole_gt, final_seg)
                
                log_file = open(log_path, "a")
                print("Img Name:", img_name, ", Confidence:", confidence.numpy())
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence.numpy()} \n")
                for (k, v) in result.items():
                    print(f"{k}, {v:.3f}")
                    log_file.writelines(f"{k}, {v:.3f}\n")
                log_file.close()

                final_map = self.heatmap_postprocess(final_map)
                MECAM_map = self.heatmap_postprocess(MECAM_map)
                ame_map = self.heatmap_postprocess(ame_map)
                input_image, mix_image = self.img_fusion(input_image, final_map)
                ic1_map = self.heatmap_postprocess(norm_map[0])
                ic2_map = self.heatmap_postprocess(norm_map[1])
                ic3_map = self.heatmap_postprocess(norm_map[2])
                ic4_map = self.heatmap_postprocess(norm_map[3])


                io.imsave(os.path.join(self.result_path, img_name, f"input_{img_name}.jpg"), img_as_ubyte(input_image), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"whole_seg_{img_name}.jpg"), img_as_ubyte(whole_gt.astype(np.float32)), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"ic1_map_{img_name}.jpg"), img_as_ubyte(ic1_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"ic2_map_{img_name}.jpg"), img_as_ubyte(ic2_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"ic3_map_{img_name}.jpg"), img_as_ubyte(ic3_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"ic4_map_{img_name}.jpg"), img_as_ubyte(ic4_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"heat_{img_name}.jpg"), img_as_ubyte(final_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"MECAM_{img_name}.jpg"), img_as_ubyte(MECAM_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"ame_{img_name}.jpg"), img_as_ubyte(ame_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"mix_{img_name}.jpg"), img_as_ubyte(mix_image), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"final_seg_{img_name}.jpg"), img_as_ubyte(final_seg.astype(np.float32)), check_contrast=False)
                # print_seg_contour(self.result_path, input_image, whole_gt.astype(np.float32), final_seg.astype(np.float32), img_name)

    def heatmap_postprocess(self, feat_map):
        heatmap = cv2.applyColorMap(np.uint8(255 * feat_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        return heatmap

    def img_fusion(self, image, heatmap):
        cam = heatmap + np.float32(image)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return image, cam

    def run_tumor_test(self, loader):
        self.model.eval()
        self.score_model.eval()

        log_path = os.path.join(self.result_path, "tumor_result.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)

        result_metric = {'Dice':[], 'IoU':[], 'HD95':[]}

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                seg_image = seg_batch[0].permute(1, 2, 0).squeeze(2).numpy()
                map_collect, final_map, confidence, MECAM_map, ame_map = self.step(case_batch)

                input_image = case_batch[0].permute(1, 2, 0)
                norm_map, final_map, final_seg, MECAM_map, ame_map = self.CAM_algo(input_image, map_collect, final_map, MECAM_map, ame_map, img_name, confidence)
                
                seg_gt = (seg_image*4).astype(np.uint8)

                whole_gt = np.where(seg_gt!=0, 1, 0)
                
                result = compute_seg_metrics(whole_gt, final_seg)
                
                log_file = open(log_path, "a")
                print("Img Name:", img_name, ", Confidence:", confidence.numpy())
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence.numpy()} \n")
                for (k, v) in result.items():
                    print(f"{k}, {v:.3f}")
                    log_file.writelines(f"{k}, {v:.3f}\n")
                log_file.close()

                for k in result_metric.keys():
                    result_metric[k].append(result[k])

        return result_metric

    def CAM_algo(self, input_image, map_collect, final_map, MECAM_map, ame_map, img_name, confidence, output_hist=False):
        img_size = input_image.shape[0], input_image.shape[1]
        norm_map = []
        for map in map_collect:
            if (map.max() - map.min()) > 0:
                map = (map - map.min()+1e-5) / (map.max() - map.min()+1e-5)
            norm_map.append(1-map)

        if (final_map.max() - final_map.min()) > 0:
            final_map = (final_map - final_map.min()+1e-5) / (final_map.max() - final_map.min()+1e-5)
        # final_map = F.interpolate(final_map, size=img_size, mode='bilinear', align_corners=False)

        final_map = final_map.squeeze(1).squeeze(0).numpy()
        final_map  = (1-final_map) 

        if (MECAM_map.max() - MECAM_map.min()) > 0:
            MECAM_map = (MECAM_map - MECAM_map.min()+1e-5) / (MECAM_map.max() - MECAM_map.min()+1e-5)
        # MECAM_map = F.interpolate(MECAM_map, size=img_size, mode='bilinear', align_corners=False)

        MECAM_map = MECAM_map.squeeze(1).squeeze(0).numpy()
        MECAM_map  = (1-MECAM_map) 
        
        if (ame_map.max() - ame_map.min()) > 0:
            ame_map = (ame_map - ame_map.min()+1e-5) / (ame_map.max() - ame_map.min()+1e-5)
        # ame_map = F.interpolate(ame_map, size=img_size, mode='bilinear', align_corners=False)

        ame_map = ame_map.squeeze(1).squeeze(0).numpy()
        ame_map  = (1-ame_map) 

        final_seg = gen_seg_mask(input_image, final_map, img_name, self.result_path, output_hist=output_hist)

        return norm_map, final_map, final_seg, MECAM_map, ame_map
