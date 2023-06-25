import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from resnet18_self import ResNet18
from resnet50_self import ResNet50
from collections import OrderedDict

class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()

        resnet = resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        self.f = nn.Sequential(*list(resnet.children())[:-1])
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU(inplace=True), nn.Linear(512, 256, bias=True))
            
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, out

class Res18_Classifier(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res18_Classifier, self).__init__()

        # resnet = ResNet18(norm_layer=nn.InstanceNorm2d)
        # self.f = nn.Sequential(*list(resnet.children())[:-2])
        self.f = ResNet18(norm_layer=nn.InstanceNorm2d)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(512, 1)

        self.ic1 = nn.Conv2d(64, 1, 1)

        self.ic2 = nn.Conv2d(128, 1, 1)

        self.ic3 = nn.Conv2d(256, 1, 1)

        self.ic4 = nn.Conv2d(512, 1, 1)


        self.att = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        # self.final = nn.Conv2d(32, 1, 1)
            

    def forward(self, input):
        batch_size = input.shape[0]
        input_size = input.shape[2], input.shape[3]
        l1_feature, l2_feature, l3_feature, l4_feature = self.f(input)

        input_gray = torch.mean(input, dim=1, keepdim=True)
        l1_map = self.ic1(l1_feature)
        l2_map = self.ic2(l2_feature)
        l3_map = self.ic3(l3_feature)
        l4_map = self.ic4(l4_feature)
        
        re_l1_map = F.interpolate(l1_map, size=input_size, mode='bilinear', align_corners=False)
        re_l2_map = F.interpolate(l2_map, size=input_size, mode='bilinear', align_corners=False)
        re_l3_map = F.interpolate(l3_map, size=input_size, mode='bilinear', align_corners=False)
        re_l4_map = F.interpolate(l4_map, size=input_size, mode='bilinear', align_corners=False)

        MECAM_map = (re_l1_map + re_l2_map + re_l3_map + re_l4_map)/4

        l1_logits = torch.flatten(self.gap(l1_map), start_dim=1)
        l2_logits = torch.flatten(self.gap(l2_map), start_dim=1)
        l3_logits = torch.flatten(self.gap(l3_map), start_dim=1)
        l4_logits = torch.flatten(self.gap(l4_map), start_dim=1)
        # final_logits = torch.flatten(self.gap(final_map), start_dim=1)
        
        logits_collect = [l1_logits, l2_logits, l3_logits, l4_logits]
        map_collect = [re_l1_map.detach(), re_l2_map.detach(), re_l3_map.detach(), re_l4_map.detach()]
        
        return logits_collect, map_collect, MECAM_map

    def normalize(self, tensor):
        a1, a2, a3, a4= tensor.size()
        tensor = tensor.view(a1, a2, -1)

        tensor_min = (tensor.min(2, keepdim=True)[0])
        tensor_max = (tensor.max(2, keepdim=True)[0])
        tensor = (tensor - tensor_min + 1e-5) / (tensor_max - tensor_min + 1e-5)
        tensor = tensor.view(a1, a2, a3, a4)
        return tensor

    def load_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")

    def load_encoder_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()

            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                if "f" in k:
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")

class Res50_Classifier(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res50_Classifier, self).__init__()

        # resnet = ResNet18(norm_layer=nn.InstanceNorm2d)
        # self.f = nn.Sequential(*list(resnet.children())[:-2])
        self.f = ResNet50(norm_layer=nn.InstanceNorm2d)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(512, 1)
        self.ic1 = nn.Conv2d(256, 1, 1)

        self.ic2 = nn.Conv2d(512, 1, 1)

        self.ic3 = nn.Conv2d(1024, 1, 1)

        self.ic4 = nn.Conv2d(2048, 1, 1)
        
        # self.final = nn.Conv2d(32, 1, 1)
            

    def forward(self, input):
        batch_size = input.shape[0]
        input_size = input.shape[2], input.shape[3]
        l1_feature, l2_feature, l3_feature, l4_feature = self.f(input)

        input_gray = torch.mean(input, dim=1, keepdim=True)
        l1_map = self.ic1(l1_feature)
        l2_map = self.ic2(l2_feature)
        l3_map = self.ic3(l3_feature)
        l4_map = self.ic4(l4_feature)
        
        re_l1_map = F.interpolate(l1_map, size=input_size, mode='bilinear', align_corners=False)
        re_l2_map = F.interpolate(l2_map, size=input_size, mode='bilinear', align_corners=False)
        re_l3_map = F.interpolate(l3_map, size=input_size, mode='bilinear', align_corners=False)
        re_l4_map = F.interpolate(l4_map, size=input_size, mode='bilinear', align_corners=False)

        MECAM_map = (re_l1_map + re_l2_map + re_l3_map + re_l4_map)/4

        l1_logits = torch.flatten(self.gap(l1_map), start_dim=1)
        l2_logits = torch.flatten(self.gap(l2_map), start_dim=1)
        l3_logits = torch.flatten(self.gap(l3_map), start_dim=1)
        l4_logits = torch.flatten(self.gap(l4_map), start_dim=1)
        # final_logits = torch.flatten(self.gap(final_map), start_dim=1)
        
        logits_collect = [l1_logits, l2_logits, l3_logits, l4_logits]
        map_collect = [re_l1_map.detach(), re_l2_map.detach(), re_l3_map.detach(), re_l4_map.detach()]
        
        return logits_collect, map_collect, MECAM_map

    def normalize(self, tensor):
        a1, a2, a3, a4= tensor.size()
        tensor = tensor.view(a1, a2, -1)

        tensor_min = (tensor.min(2, keepdim=True)[0])
        tensor_max = (tensor.max(2, keepdim=True)[0])
        tensor = (tensor - tensor_min + 1e-5) / (tensor_max - tensor_min + 1e-5)
        tensor = tensor.view(a1, a2, a3, a4)
        return tensor

    def load_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")

    def load_encoder_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()

            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                if "f" in k:
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")

class Res_Scoring(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res_Scoring, self).__init__()

        self.att = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        # self.final = nn.Conv2d(32, 1, 1)
            

    def forward(self, input, map_collect):

        input_gray = torch.mean(input, dim=1, keepdim=True)

        mask = torch.cat(map_collect, dim=1)
        norm_mask = self.normalize(mask)

        masked_input = input_gray * norm_mask
        map_att = self.att(masked_input)
        map_weight = F.softmax(map_att, dim=1)

        final_map = torch.sum(mask*map_weight, dim=1, keepdim=True)
        # norm_final_map = self.normalize(final_map)
        foreground = input_gray * final_map
        foreground = torch.flatten(foreground, start_dim=1)
        background = input_gray * (1-final_map)
        background = torch.flatten(background, start_dim=1)
        map_collect.append(final_map)
        all_map = torch.cat(map_collect, dim=1)
        average_map = torch.mean(all_map, dim=1, keepdim=True)


        return average_map, foreground, background, final_map

    def normalize(self, tensor):
        a1, a2, a3, a4= tensor.size()
        tensor = tensor.view(a1, a2, -1)

        tensor_min = (tensor.min(2, keepdim=True)[0])
        tensor_max = (tensor.max(2, keepdim=True)[0])
        tensor = (tensor - tensor_min + 1e-5) / (tensor_max - tensor_min + 1e-5)
        tensor = tensor.view(a1, a2, a3, a4)
        return tensor

    def load_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")

    def load_encoder_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()

            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                if "f" in k:
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")