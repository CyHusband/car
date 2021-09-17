import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any
from torchvision import models

class Car(nn.Module):
    def __init__(self):
        super(Car, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for p in self.parameters():
            #p.requires_grad = False
            pass
        self.linear = nn.Linear(49, 5, bias=True)
        self.encoder = TransformerEncoderLayer_(d_model=5, nhead=1, d_out=5)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        
        x = x.flatten(-2)
        x = self.linear(x)
        x = self.encoder(x).sigmoid()
        x = x.mean(1)

        return x

class TransformerEncoderLayer_(nn.Module):

    def __init__(self, d_model, nhead, d_out=5, dropout=0.1):
        super(TransformerEncoderLayer_, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        #src = self.norm(src)
        #src = self.self_attn2(src, src, src, attn_mask=src_mask,
        #                      key_padding_mask=src_key_padding_mask)[0]
        #src = self.norm(src)
        return src
