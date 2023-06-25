import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import *
from cnet import CNet
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

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
                        default=10,
                        help="number of epoch")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=512,
                        help="batch size") 

    parser.add_argument("--learning_rate",
                        "-lr",
                        type=float,
                        default=1e-4,
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

    parser.add_argument("--encoder_pretrained_path",
                        type=str, 
                        default=None,
                        help="encoder pretrained path")

    parser.add_argument("--pretrained_path",
                        type=str, 
                        default=None,
                        help="pretrained path")

    # args parse
    args = parser.parse_args()
    gpu_id = args.gpu_id
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpuid = [i for i in range(0, len(gpu_id.split(",")))]
    print("Use {} GPU".format(len(gpu_id.split(","))))

    record_path = "record/CNet/"

    dataset_path = "/work/vincent18/"
    input_path = os.path.join(dataset_path, args.input_path)

    assert args.modality in ['flair', 't1', 't1ce', 't2'], 'error modality given'

    model_name = f"{args.model_type}_{args.modality}_ep{args.epochs}_b{args.batch_size}.AME-CAM"

    if args.suffix != None:
        model_name += f".{args.suffix}"

    if args.encoder_pretrained_path == None:
        model_name += ".scratch"
        

    if not os.path.exists(os.path.join(args.project_path, record_path, model_name, "model")):
        os.makedirs(os.path.join(args.project_path, record_path, model_name, "model"))

    full_log_path = os.path.join(args.project_path, record_path, model_name, "log.log")
    print("============== Load Dataset ===============")
    normal_path = glob.glob(os.path.join(input_path, args.modality, "training", "normal", "*.jpg"))
    tumor_path = glob.glob(os.path.join(input_path, args.modality, "training", "tumor", "*.jpg"))
    print("Num of Tumor Data:", len(tumor_path))
    print("Num of Normal Data:", len(normal_path))
    data = normal_path + tumor_path
    print("data length", len(data))
    total_cases = len(data)

    print("============== Model Setup ===============")

    train_index, val_index = train_test_split(range(total_cases), test_size=0.1)
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(val_index)

    dataset = ImageDataset(data)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                pin_memory=False, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=False, drop_last=True, sampler=test_sampler)

    net = CNet(args, train_loader, val_loader, full_log_path, record_path, model_name, gpuid)

    print("============== Start Training ===============")
    
    net.run()
        