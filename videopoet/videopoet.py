import math
from contextlib import nullcontext
from functools import partial, wraps

import torch
import torch
from torch import nn, einsum
import torch.nn.functional as F

from videopoet.utils import default

from .t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME


class VideoPoet(nn.Module):
    def __init__(self, 
                 t5_name = DEFAULT_T5_NAME, 
                 text_embed_dim = None, 
                 max_text_len = 128
        ):
        super().__init__()

        # text conditioning
        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name = t5_name)
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len

    def forward(self, x):
        return x
