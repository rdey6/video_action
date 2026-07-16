#!/bin/bash
python run.py \
    --frame_dir "/content/UCF50/UCF50" \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 256 \
    --rnn_n_layers 2 \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 32 \
    --mode 'train' \
    --best_dir '/content/video_action_best_model'