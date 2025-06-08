import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation block for 1D feature maps (time dimension).
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock1d, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, time)
        # Squeeze: global average pooling over time
        y = x.mean(dim=2, keepdim=True)           # shape: (batch, channels, 1)
        y = self.relu(self.fc1(y))                # (batch, channels/reduction, 1)
        y = self.sigmoid(self.fc2(y))             # (batch, channels, 1)
        return x * y                              # scale

class SEConv1d(nn.Module):
    """
    TDNN (1D conv) block with SE attention and residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(SEConv1d, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.tdnn = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, dilation=dilation,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock1d(out_channels)
        self.residual = (in_channels == out_channels)

    def forward(self, x):
        # x: (batch, in_channels, time)
        out = self.relu(self.bn(self.tdnn(x)))
        out = self.se(out)
        if self.residual:
            out = out + x
        return out


class AttentiveStatsPool(nn.Module):
    """
    Attentive Statistics Pooling used in ECAPA‑TDNN.
    Learns an attention weight for each time frame, then computes
    weighted mean and standard deviation over time.
    """
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.tanh   = nn.Tanh()
        self.attn   = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.attn(x)                    # (B, C, T) attention weights
        mu = torch.sum(x * w, dim=2)        # weighted mean (B, C)
        sigma = torch.sqrt(
            torch.sum((x ** 2) * w, dim=2) - mu ** 2 + 1e-9
        )                                   # weighted std  (B, C)
        return torch.cat([mu, sigma], dim=1)  # (B, 2C)

class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN for audio spoof detection.
    Input: (batch, 1, n_mels, time_steps)
    Steps:
      1. Squeeze channel dim and treat as (batch, n_mels, time)
      2. SE-augmented TDNN blocks (depth configurable)
      3. Statistics pooling
      4. FC layers with dropout for classification
    """
    def __init__(self, num_classes=2, n_mels=60,
                 channels=512, emb_dim=192, depth=4,  # depth 3‑5
                 dropout_p=0.6):
        super(ECAPA_TDNN, self).__init__()
        # TDNN + SE blocks
        self.layer1 = SEConv1d(n_mels,    channels, kernel_size=5, dilation=1)
        self.layer2 = SEConv1d(channels,  channels, kernel_size=3, dilation=2)
        self.layer3 = SEConv1d(channels,  channels, kernel_size=3, dilation=3)
        # Additional TDNN blocks (depth selectable)
        self.layer4 = (SEConv1d(channels, channels, kernel_size=3, dilation=4) 
                        if depth >= 4 else None)
        self.layer5 = (SEConv1d(channels, channels, kernel_size=3, dilation=5)
                       if depth >= 5 else None)
        self.layer6 = (SEConv1d(channels, channels, kernel_size=3, dilation=5)
                       if depth >= 6 else None)
        # Statistics pooling
        self.pool = AttentiveStatsPool(channels)
        # Classification head
        self.emb_net = nn.Sequential(
            nn.Linear(channels * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout_p)
        )
        self.fc_out = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, time_steps)
        b, _, n_mels, t = x.size()
        x = x.view(b, n_mels, t)              # (batch, n_mels, time)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4 is not None:
            out = self.layer4(out)
        if self.layer5 is not None:
            out = self.layer5(out)
        if self.layer6 is not None:
            out = self.layer6(out)
        stats = self.pool(out)              # (B, 2C)
        emb   = self.emb_net(stats)         # (B, emb_dim)
        logits = self.fc_out(emb)           # (B, num_classes)
        return emb, logits

def build_model(input_dim=None, hidden1=None, hidden2=None, num_classes=2,
                depth=4, dropout_p=0.6):
    """
    Builder function for ECAPA-TDNN.
    input_dim: number of mel bins (n_mels)
    hidden1: channels
    hidden2: embedding dimension
    depth: number of TDNN blocks (3–5)
    dropout_p: dropout probability for classifier
    """
    n_mels = input_dim if input_dim is not None else 60
    channels = hidden1 if hidden1 is not None else 512
    emb_dim  = hidden2 if hidden2 is not None else 192
    return ECAPA_TDNN(num_classes=num_classes,
                      n_mels=n_mels,
                      channels=channels,
                      emb_dim=emb_dim,
                      depth=depth,
                      dropout_p=dropout_p)