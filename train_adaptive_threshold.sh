#!/bin/bash
session_name='voc12_adaptive_t'
data_dir='/workdir/VOCdevkit/VOC2012/JPEGImages'
gt_dir='/workdir/VOCdevkit/VOC2012/SegmentationClass'
output_dir='./results/'
phase=1

while(( $phase<=3 ))
do
    echo $phase
    phase_name=$session_name'_phase'$phase
    let last_phase=phase-1
    last_phase_name=$session_name'_phase'$last_phase
    if [ $phase -le 1 ]
    then
        python3 train.py --session_name $phase_name --data_dir $data_dir --pretrain ./voc12_init.pth --fg_path ./datasets/voc12_init.npy --output_dir $output_dir
    else
        python3 train.py --session_name $phase_name --data_dir $data_dir --pretrain $output_dir$last_phase_name/$last_phase_name'-voc12-final.pth' --fg_path ./datasets/$last_phase_name'_fg.npy' --output_dir $output_dir
    fi
    python3 make_cam.py --session_name $phase_name --data_dir $data_dir --cam_path $output_dir$phase_name'_cam' --output_dir $output_dir --adaptive_t True
    python3 eval.py --type cam --cam_path $output_dir$phase_name'_cam' --gt_path $gt_dir  --adaptive_t True
    python3 make_crf.py --data_dir $data_dir --cam_path $output_dir$phase_name'_cam' --crf_path $output_dir$phase_name'_crf'
    python3 eval.py --type crf --crf_path $output_dir$phase_name'_crf' --gt_path $gt_dir
    python3 create_fg.py --session_name $phase_name --crf_path $output_dir$phase_name'_crf'
    let phase=phase+1
done
