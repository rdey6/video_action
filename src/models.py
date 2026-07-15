"""
Module: models.py

This module defines an enhanced LRCN (Long-term Recurrent Convolutional Network) and
an I3D (Inflated 3D ConvNet) model for video classification. 

The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature extraction 
from individual frames with a bidirectional LSTM for temporal modeling across frame sequences. 
A multi-head self-attention mechanism refines temporal representations before feature 
aggregation and classification through a fully-connected prediction head.

The I3D architecture extends 2D CNNs into the temporal domain by inflating
2D convolutional filters into 3D convolutions, allowing the model to jointly
learn spatial and temporal features from video clips. The model uses a pretrained I3D backbone 
and replaces the final classification layer for the target dataset.
Input format:
    (batch_size, channels, time_steps, height, width)

Example:
    Batch of 16 RGB clips with 32 frames:
        (16, 3, 32, 224, 224)

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace
              the fully-connected layer of the ResNet backbone.
    TemporalAttention: A multi-head self-attention module for refining LSTM temporal
                       representations.
    LRCN: The main video classification model integrating CNN feature extraction,
          bidirectional LSTM temporal modeling, temporal attention, pooling, and
          fully-connected classification layers.
    I3D: Inflated 3D ConvNet (I3D) for video classification.
"""

import torch.nn as nn
from torchvision import models
import torch
from pytorch-i3d.pytorch_i3d import InceptionI3d

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention module for temporal feature refinement.

    This module applies self-attention across the sequence of LSTM outputs,
    allowing each time step to attend to all other time steps and learn richer
    temporal dependencies. The output is an attention-enhanced sequence of
    features with the same shape as the input.
    """

    def __init__(self, hidden_size, num_heads=4):
        super(TemporalAttention, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, lstm_output):
        """
        Apply multi-head self-attention to the sequence of LSTM outputs.

        Args:
            lstm_output (Tensor): LSTM output tensor of shape
                (batch_size, time_steps, hidden_size).

        Returns:
            Tensor: Attention-enhanced sequence of features with shape
                (batch_size, time_steps, hidden_size).
        """

        attn_out, _ = self.attn(lstm_output, lstm_output, lstm_output)

        return attn_out
    
    
class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet backbone as a 2D CNN feature extractor
    to obtain spatial representations from individual video frames.
    The frame-level features are processed by a bidirectional LSTM to
    capture temporal dependencies. A multi-head self-attention module
    refines the temporal representation, which is combined with average
    temporal pooling before classification. Dropout and fully-connected
    layers are used to produce the final class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
                                   Default is 'resnet34'.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34'):
        super(LRCN, self).__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if cnn_model == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(pretrained=pretrained)
        elif cnn_model == 'resnet50':
            base_cnn = models.resnet50(pretrained=pretrained)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model == 'resnet152':
            # Note: This example uses resnet34 for resnet152 option as a placeholder.
            base_cnn = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features
        
        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features.
        self.rnn = nn.LSTM(num_features, 
                           hidden_size, 
                           n_layers, 
                           batch_first=True, 
                           bidirectional=True,
                           dropout=0.5 if n_layers > 1 else 0)

        self.attention = TemporalAttention(hidden_size * 2)
        
        self.norm = nn.LayerNorm(hidden_size * 2)
       
        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout_rate),
                                nn.Linear(hidden_size, n_classes))

    def forward(self, x):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        Each frame is first processed independently by the CNN backbone to extract spatial
        features. The sequence of frame features is then passed through a bidirectional LSTM
        to model temporal dependencies across the video. A multi-head self-attention mechanism 
        refines the temporal features by allowing each time step to attend to all other time steps. 
        The resulting sequence representation is aggregated using temporal pooling before classification.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        
        # Extract CNN features for every frame

        features = []

        for idx in range(ts):
            y = self.base_model(x[:, idx])
            features.append(y)

        # (B,T,F)
        features = torch.stack(features, dim=1)

        # Process the whole sequence once
        lstm_out, _ = self.rnn(features)

        attn_out = self.attention(lstm_out)

        attn_pool = attn_out.mean(dim=1)

        # Average LSTM features
        avg_pool = lstm_out.mean(dim=1)

        # Layer normalization
        attn_pool = self.norm(attn_pool)
        avg_pool = self.norm(avg_pool)

        # Concatenate
        context = torch.cat([attn_pool, avg_pool], dim=1)

        out = self.fc(context)

        return out
    
class I3D(nn.Module):
    """
    Inflated 3D ConvNet (I3D) for video classification.

    The model uses an I3D backbone pretrained on Kinetics and fine-tunes
    the final classification layer for the target number of classes.

    Args:
        n_classes (int):
            Number of output classes.

        pretrained (bool):
            Whether to load pretrained Kinetics weights.

        dropout_rate (float):
            Dropout probability before classification.

    """

    def __init__(
        self,
        n_classes,
        pretrained=True,
        dropout_rate=0.5
    ):
        super(I3D, self).__init__()

        # RGB input I3D backbone
        self.base_model = InceptionI3d(
            num_classes=400,
            in_channels=3
        )

        # Load pretrained Kinetics weights
        if pretrained:
            self.base_model.load_state_dict(
                torch.load(
                    "../weights/rgb_imagenet.pt",
                    map_location="cpu"
                )
            )

        # Replace final classification layer
        self.base_model.replace_logits(n_classes)

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Video tensor with shape:
                (batch_size, channels, time_steps, height, width)

        Returns:
            Classification logits:
                (batch_size, n_classes)
        """

        x = self.base_model(x)

        x = self.dropout(x)

        return x