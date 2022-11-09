#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        # python3 pretrain_clnet.py -m t1
        python3 train_cnet.py -b 256 -m t1 --encoder_pretrained_path Res18_t1_ep10_b256.ME -s 1
        python3 test_cnet.py -m t1 --pretrained_path Res18_t1_ep10_b256.ME-CLwn_score_only.1
    elif [ "$1" == "t2" ]; then
        # python3 pretrain_clnet.py -m t2
        python3 train_cnet.py -b 256 -m t2 --encoder_pretrained_path Res18_t2_ep10_b256.ME -s 1
        python3 test_cnet.py -m t2 --pretrained_path Res18_t2_ep10_b256.ME-CLwn_score_only.1
    elif [ "$1" == "t1ce" ]; then
        # python3 pretrain_clnet.py -m t1ce
        python3 train_cnet.py -b 256 -m t1ce --encoder_pretrained_path Res18_t1ce_ep10_b256.ME -s 1
        python3 test_cnet.py -m t1ce --pretrained_path Res18_t1ce_ep10_b256.ME-CLwn_score_only.1
    elif [ "$1" == "flair" ]; then
        # python3 pretrain_clnet.py -m flair
        python3 train_cnet.py -b 256 -m flair --encoder_pretrained_path Res18_flair_ep10_b256.ME -s 1
        python3 test_cnet.py -m flair --pretrained_path Res18_flair_ep10_b256.ME-CLwn_score_only.1
    else
    echo "Error modality. Usage: run.sh [modality]"
    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi