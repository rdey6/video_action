#!/bin/bash
python run.py \
    --ckpt /content/drive/MyDrive/model/video_action_best_model/best_model_wts.pt \
    --cnn_backbone resnet50 \
    --model_type lrcn \
    --rnn_hidden_size 256 \
    --rnn_n_layers 2 \
    --n_classes 50 \
    --batch_size 32 \
    --mode eval \
    --frame_dir "/content/UCF50/UCF50"