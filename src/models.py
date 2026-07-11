"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
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
    Learns attention weights over all LSTM outputs.
    """

    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()

        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, time_steps, hidden_size)

        Returns:
            context: (batch_size, hidden_size)
        """

        # Compute attention scores
        scores = self.attention(lstm_output)          # (B,T,1)

        # Normalize scores
        weights = torch.softmax(scores, dim=1)        # (B,T,1)

        # Weighted sum of LSTM outputs
        context = torch.sum(weights * lstm_output, dim=1)

        return context
    
class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

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
        #self.rnn = nn.LSTM(num_features, hidden_size, n_layers)
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True, bidirectional=True)

        self.attention = TemporalAttention(hidden_size * 2)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        #self.fc = nn.Linear(hidden_size, n_classes)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        Each frame is first processed independently by the CNN backbone to extract spatial
        features. The sequence of frame features is then passed through a bidirectional LSTM
        to model temporal dependencies across the video. A temporal attention mechanism learns
        the importance of each time step and computes a weighted combination of the LSTM outputs.
        The resulting context vector is regularized using dropout and passed through the final
        fully-connected layer to produce class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        
        ###Commenting out - start
        # Process the first frame separately to initialize the LSTM hidden and cell states.
        ##idx = 0
        ##y = self.base_model(x[:, idx])
        ##_, (hn, cn) = self.rnn(y.unsqueeze(1))
        
        # Iterate over the remaining frames, feeding each frame's features into the LSTM.
        ##for idx in range(1, ts):
        ##    y = self.base_model(x[:, idx])
        ##    out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        
        # Apply dropout to the output of the final time step.
        ##out = self.dropout(out[:, -1])
        
        # Pass the final output through the fully-connected layer to get class logits.
        ##out = self.fc(out)
        ##return out
        ###Commenting out - end
        # Extract CNN features for every frame

        features = []

        for idx in range(ts):
            y = self.base_model(x[:, idx])
            features.append(y)

        # (B,T,F)
        features = torch.stack(features, dim=1)

        # Process the whole sequence once
        lstm_out, _ = self.rnn(features)

        # Temporal attention
        context = self.attention(lstm_out)

        # Dropout + classifier
        context = self.dropout(context)

        out = self.fc(context)

        return out
