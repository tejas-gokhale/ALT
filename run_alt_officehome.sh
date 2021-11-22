#!/usr/bin/env bash
trap "exit" INT
base_cmd="python train.py --data_name officehome --feat none --name main "
net_cmd=" --net resnet50 -nc 65 --image_size 224 --trans fcn --pre_epoch 4 "
opt_cmd=" -ni 2000 -vi 100 -bs 32 --lr 0.004 -sgd -sch CosLR --val_freq 2 "
sc_vals=("real" "art" "clipart" "product")
rs_vals=(1 2 3 4 5)
hp_cmd=" --alt --cl --clw 0.75 -db kaiming --lr_adv 5e-5 --adv_steps 10 "
# this block runs 5 seeds for each source parallely on 8 gpus
# you can change this to suit your compute resources
GPU=0
for sc in "${sc_vals[@]}"
do
	for rs in "${rs_vals[@]}"
	do
		CUDA_VISIBLE_DEVICES=$GPU nohup $base_cmd $net_cmd $opt_cmd $hp_cmd -rs $rs -sc $sc > runs/officehome_"$sc"_"$rs".out &
		((GPU=(GPU+1)%8))
	done
done