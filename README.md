# AME-CAM: Attentive Multiple-Exit CAM for Weakly Supervised Segmentation on MRI Brain Tumor (AME-CAM) [MICCAI 23']
Official code implementation for the AME-CAM paper accepted by MICCAI 2023.

## Dataset
[RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2021](http://braintumorsegmentation.org/)
Download the official BraTS 2021 Dataset Task 1
Preprocess the dataset from 3D volume data into 2D slide with the following script.
```
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
cd ./model_phase/
python3 pretrain_clnet.py -m t1 --model_type Res18
```
## Train and Test Multi-exit Classifier with Pretrained Encoder
```
cd ./me_encoder/
python3 train_cnet.py -b 256 -m t1 --encoder_pretrained_path SimCLR/Res18_t1_ep100_b512
python3 test_cnet.py -m t1 --pretrained_path Res18_t1_ep10_b256.ME
```
## Train and Test the Activation Aggregation Network with the Contrastive Loss
```
cd ./attention_aggregation_network/
python3 train_cnet.py -b 256 -m t1 --encoder_pretrained_path Res18_t1_ep10_b256.ME
python3 test_cnet.py -m t1 --pretrained_path Res18_t1_ep10_b256.AME-CAM
```
## Run the Inference stage of AME-CAM 
```
cd ./AME-CAM_inference/
python3 main.py --pretrained_path Res18_t1_ep10_b256.AME-CAM -m t1
```

## Citation
If you use the code or results in your research, please use the following BibTeX entry.
```
```
