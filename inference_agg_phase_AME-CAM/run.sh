#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        python3 main.py --pretrained_path Res18_t1_ep10_b256.ME-CL_score_only -m t1
    elif [ "$1" == "t2" ]; then
        python3 main.py --pretrained_path Res18_t2_ep10_b256.ME-CL_score_only -m t2
    elif [ "$1" == "t1ce" ]; then
        python3 main.py --pretrained_path Res18_t1ce_ep10_b256.ME-CL_score_only -m t1ce
    elif [ "$1" == "flair" ]; then
        python3 main.py --pretrained_path Res18_flair_ep10_b256.ME-CL_score_only -m flair

    else
    echo "Error modality. Usage: run.sh [modality]"

    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi