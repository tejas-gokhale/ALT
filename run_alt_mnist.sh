#!/usr/bin/env bash
trap "exit" INT
base_cmd="python train.py --data_name digits -sc mnist10k --feat none --name main "
net_cmd=" --net digit -nc 10 --image_size 32 --trans fcn --pre_epoch 5 "
opt_cmd=" -ni 10000 -vi 250 -bs 32 --lr 0.0001 --val_freq 4 "
hp_cmd=" --alt --cl --clw 0.75 -db kaiming --lr_adv 5e-6 --adv_steps 10 "
rs_vals=(1 2 3 4 5)
# this block runs 5 seeds on 5 different gpus parallely.
# you can change this to suit your compute resources
GPU=0
for rs in "${rs_vals[@]}"
do
	CUDA_VISIBLE_DEVICES=$GPU nohup $base_cmd $net_cmd $opt_cmd $hp_cmd -rs $rs > runs/digits_"$rs".out &
	((GPU=(GPU+1)%8))
done
