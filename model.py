import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ArtCNN_RNN(nn.Module):
    """CNN-RNN with Attention for Art Classification"""
    
    def __init__(self, num_artists, num_genres, num_styles, 
                 rnn_hidden=512, dropout=0.3):
        super().__init__()
        
        # CNN Backbone
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature dimensions
        self.feature_dim = 2048
        self.spatial_size = 7
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # RNN
        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Classification heads
        self.artist_classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_artists)
        )
        
        self.genre_classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_genres)
        )
        
        self.style_classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_styles)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN features
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, self.feature_dim, -1)
        cnn_features = cnn_features.permute(0, 2, 1)
        
        # RNN
        rnn_output, _ = self.rnn(cnn_features)
        
        # Attention
        attention_weights = self.attention(rnn_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Context vector
        context = torch.sum(attention_weights * rnn_output, dim=1)
        
        # Predictions
        artist_out = self.artist_classifier(context)
        genre_out = self.genre_classifier(context)
        style_out = self.style_classifier(context)
        
        return artist_out, genre_out, style_out, attention_weights