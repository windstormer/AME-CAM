import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import *
from cnet import CNet
import glob
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path",
                        type=str, 
                        default="../../",
                        help="path of project")

    parser.add_argument("--input_path",
                        type=str, 
                        default="BraTS_patch",
                        help="path of dataset")                 

    parser.add_argument("--gpu_id",
                        type=str,
                        default ='',
                        help="gpu id number")

    parser.add_argument("--patch_size",
                        "-p",
                        type=int,
                        default=240,
                        help="image size")

    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=200,
                        help="number of epoch")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=128,
                        help="batch size") 

    parser.add_argument("--learning_rate",
                        "-lr",
                        type=float,
                        default=1e-3,
                        help="learning rate")          

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")

    parser.add_argument('-m',
                        '--modality',
                        type=str,
                        default='t1',
                        help='Modality select [flair, t1, t1ce, t2]')

    parser.add_argument("--model_type",
                        type=str,
                        default="Res18",
                        choices=["Res18", "Res50"],
                        help="Type of Model [Res18, Res50]")

    parser.add_argument("--pretrained_path",
                        type=str, 
                        default=None,
                        help="pretrained path")
    
    parser.add_argument("--encoder_pretrained_path",
                        type=str, 
                        default=None,
                        help="encoder pretrained path")

    # args parse
    args = parser.parse_args()
    gpu_id = args.gpu_id
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpuid = range(len(gpu_id.split(",")))
    print("Use {} GPU".format(len(gpu_id.split(","))))

    record_path = "record/CNet"


    dataset_path = "/work/vincent18/"
    input_path = os.path.join(dataset_path, args.input_path)

    assert args.modality in ['flair', 't1', 't1ce', 't2'], 'error modality given'
    model_name = args.pretrained_path
    assert os.path.exists(os.path.join(args.project_path, record_path, model_name, "model")), f'{model_name} model not found'

    full_log_path = os.path.join(args.project_path, record_path, model_name, "log.log")
    print("============== Load Dataset ===============")
    normal_path = glob.glob(os.path.join(input_path, args.modality, "validate", "normal", "*.jpg"))
    tumor_path = glob.glob(os.path.join(input_path, args.modality, "validate", "tumor", "*.jpg"))
    print("Num of Tumor Data:", len(tumor_path))
    print("Num of Normal Data:", len(normal_path))
    data = normal_path + tumor_path
    print("data length", len(data))
    total_cases = len(data)

    print("============== Model Setup ===============")

    evaluate_dataset = ImageDataset(data)

    evaluate_loader = DataLoader(evaluate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                            pin_memory=True, drop_last=True)

    net = CNet(args, None, None, None, record_path, model_name, gpuid)
    print("============== Start Testing ===============")
    
    test_loss, test_acc, auc, sensitivity, specificity = net.test(evaluate_loader)
    log_file = open(full_log_path, "a")
    log_file.writelines(
        f"Final !! Test Loss: {test_loss}, Test Acc: {test_acc}, AUC: {auc}, Sensitivity: {sensitivity}, Specificity: {specificity}\n")
    log_file.writelines(str(datetime.now())+"\n")
    log_file.close()