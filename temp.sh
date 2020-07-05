#!/usr/bin/env bash

D:/anconda3/python.exe -u  G:/lxb/desktop/easyCS/train_DQBCS.py \
--data_star_num 1 --data_end_num 100 \
--model DQBCS --model_save True --test_model False \
--model_name  DQBCS_v0.t7  --model_log  DQBCS_v0 \
--maskname poisson2d --maskperc  5  \
--epochs 60 --lr 1e-3 --batch_size 4 \
--loss_mse_only True --loss_ssim False --loss_vgg False \
--DQBCS_rate 0.05\