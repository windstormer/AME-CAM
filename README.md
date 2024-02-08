# AME-CAM: Attentive Multiple-Exit CAM for Weakly Supervised Segmentation on MRI Brain Tumor (AME-CAM) [MICCAI 23']
Official code implementation for the AME-CAM paper accepted by MICCAI 2023.

## Dataset
[RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2021](http://braintumorsegmentation.org/)

Download the official BraTS 2021 Dataset Task 1.

Split the official training set into training and validation with the ratio 9:1.
(The case id for training and validation set are shown in dataset.txt.)

Preprocess the dataset from 3D volume data into 2D slide with the following script.
```
cd ./src/
python3 gen_dataset.py -m t1 -d training/validate
```

Folder Structures for Dataset
```
DATASET_NAME
|-- flair
|   |-- training
|   |   |-- normal
|   |   |   |-- NORMAL_1.png
|   |   |   |-- ...
|   |   |-- seg
|   |   |   |-- TUMOR_1.png
|   |   |   |-- ...
|   |   |-- tumor
|   |   |   |-- TUMOR_1.jpg
|   |   |   |-- ...
|   |-- validate
|   |   |-- normal
|   |   |   |-- NORMAL_1.png
|   |   |   |-- ...
|   |   |-- seg
|   |   |   |-- TUMOR_1.png
|   |   |   |-- ...
|   |   |-- tumor
|   |   |   |-- TUMOR_1.jpg
|   |   |   |-- ...
|-- t1
|-- t1ce
|-- t2
```
## Encoder Pretrain with Self-supervised Methods
```
cd ./src/encoder_phase/
python3 pretrain_clnet.py -m t1 --model_type Res18
```
## Train and Test Multi-exit Classifier with Pretrained Encoder
```
cd ./src/encoder_phase/
python3 train_cnet.py -b 256 -m t1 --encoder_pretrained_path SimCLR/Res18_t1_ep100_b512
python3 test_cnet.py -m t1 --pretrained_path Res18_t1_ep10_b256.ME
```
## Train and Test the Activation Aggregation Network with the Contrastive Loss
```
cd ./src/attention_aggregation_network/
python3 train_cnet.py -b 256 -m t1 --encoder_pretrained_path Res18_t1_ep10_b256.ME
python3 test_cnet.py -m t1 --pretrained_path Res18_t1_ep10_b256.AME-CAM
```
## Run the Inference stage of AME-CAM 
```
cd ./src/AME-CAM_inference/
python3 main.py --pretrained_path Res18_t1_ep10_b256.AME-CAM -m t1
```

## Citation
If you use the code or results in your research, please use the following BibTeX entry.
```
@article{chen2023ame,
  title={AME-CAM: Attentive Multiple-Exit CAM for Weakly Supervised Segmentation on MRI Brain Tumor},
  author={Chen, Yu-Jen and Hu, Xinrong and Shi, Yiyu and Ho, Tsung-Yi},
  journal={arXiv preprint arXiv:2306.14505},
  year={2023}
}
```
