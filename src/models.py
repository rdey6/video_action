"""
Module: models.py

This module defines an enhanced LRCN (Long-term Recurrent Convolutional Network)
model for video classification. The model combines a 2D CNN backbone (e.g., ResNet)
for spatial feature extraction from individual frames with a bidirectional LSTM for
temporal modeling across frame sequences. A multi-head self-attention mechanism
refines temporal representations before feature aggregation and classification through
a fully-connected prediction head.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace
              the fully-connected layer of the ResNet backbone.
    TemporalAttention: A multi-head self-attention module for refining LSTM temporal
                       representations.
    LRCN: The main video classification model integrating CNN feature extraction,
          bidirectional LSTM temporal modeling, temporal attention, pooling, and
          fully-connected classification layers.
"""

import torch.nn as nn
from torchvision import models
import torch

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
        self.freeze_cnn_layers()

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

    def freeze_cnn_layers(self):
        """
        Freeze early ResNet layers and unfreeze deeper layers for fine-tuning.

        Strategy:
            - Freeze: conv1, bn1, layer1, layer2
            - Train: layer3, layer4

        This preserves generic ImageNet features while allowing deeper
        convolutional layers to adapt to video action recognition.
        """

        # Freeze entire ResNet backbone first
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze deeper ResNet layers
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True

        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # Keep the backbone classifier replacement (Identity) frozen
        self.base_model.fc.requires_grad = False

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