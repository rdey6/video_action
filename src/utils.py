"""
Module: utils.py

This module provides helper functions for video processing and data transformations 
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

from torchvision import transforms as transforms
from torch.utils.data import DataLoader
#from video_datasets import collate_fn_r3d_18, collate_fn_rnn
import torch
from torch.nn.utils.rnn import pad_sequence


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.
            
    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return frames, 0
    frame_idx = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    for idx, frame in enumerate(frames):
        print("processing")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, "frame{}.jpg".format(idx))
        cv2.imwrite(path_to_frame, frame)


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets.

    The training transforms include data augmentation such as random horizontal flipping and random affine transformations,
    while the validation/test transforms consist solely of resizing, converting to tensor, and normalizing.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn'):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    if model == "lrcn":
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_rnn)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn)
    else:
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_r3d_18)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_r3d_18)
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn'):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    if model == "lrcn":
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_rnn)
    else:
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_r3d_18)
    dataloaders = {'test': test_dl}
    return dataloaders

def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    
    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))
    
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    
    return padded_imgs, labels_tensor

