import torch
import numpy as np
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x

# Discussions on batch norm (https://stats.stackexchange.com/questions/361700/lack-of-batch-normalization-before-last-fully-connected-layer)
class SDRInterpolateHead(nn.Module):
    def __init__(self):
        super(SDRInterpolateHead, self).__init__()
        self._interp = nn.functional.interpolate
        self._head = nn.Sequential(
            #nn.Dropout(p=0.5), # TODO: maybe put it back
            nn.Conv2d(768, 192, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(192, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        #print("weight", torch.sum(self._head[0].weight))
        x = self._interp(x, size=(100, 464), mode="bilinear", align_corners=False)
        x = self._head(x)
        return x.squeeze(1)

    def get_output_type(self):
        return "heatmap"

class SDRDeconvHead(nn.Module):
    def __init__(self):
        super(SDRDeconvHead, self).__init__()
        self._interp = nn.functional.interpolate
        self._head = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 2, stride=2),
            #nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2),
            #nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 1, 2, stride=2),
        )
        
    def forward(self, x, output_size=(100, 464)):
        """
        Output size [bsx58x208]
        """
        x = self._head(x)
        x = self._interp(x, size=output_size, mode="bilinear", align_corners=False)
        return x.squeeze(1)

    def get_output_type(self):
        return "heatmap"


class SDRConcatDeconvHead(nn.Module):
    def __init__(self):
        super(SDRConcatDeconvHead, self).__init__()
        self._interp = nn.functional.interpolate
        self._head = nn.Sequential(
            nn.ConvTranspose2d(768 * 2, 384, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 1, 2, stride=2),
        )
        
    def forward(self, image, text, output_size=(100, 464)):
        """
        Output size [bsx58x208]
        """
        h, w = image.shape[2:]
        tiled_text = text.unsqueeze(-1).unsqueeze(-1).repeat(1,1,h,w) #torch.tile(text.unsqueeze(-1).unsqueeze(-1), (1,1,h,w))
        x = torch.cat([image, tiled_text], 1)
        x = self._head(x)
        x = self._interp(x, size=output_size, mode="bilinear", align_corners=False)
        return x.squeeze(1)

    def get_output_type(self):
        return "heatmap"

class MMax(torch.nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=(2, 3))

class MMean(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(2, 3))

class SDRConcatRegressionHead(nn.Module):
    """
    SDR head desinged for regression objective
    """
    def __init__(self, mode: str = "concat_regression_mean"):
        super(SDRConcatRegressionHead, self).__init__()
        self._head = nn.Sequential(
            nn.Conv2d(768 * 2, 384, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(384, 192, 2, stride=2),
            nn.ReLU(),
            MMax() if mode == "concat_regression_max" else MMean(),
            nn.Linear(192, 2),
        )
        
    def forward(self, image, text):
        """
        Output size [bsx58x208]
        """
        h, w = image.shape[2:]
        tiled_text = torch.tile(text.unsqueeze(-1).unsqueeze(-1), (1,1,h,w))
        x = torch.cat([image, tiled_text], 1)
        x = self._head(x).sigmoid()

        return x
    
    def get_output_type(self):
        return "coords"


class ContrastiveHead(nn.Module):
    def __init__(self, hidden_size: int = 768, num_distractors: int = 5):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_distractors + 1)

    def forward(self, x):
        x = self.fc(x)
        return x

class CLIPContrastiveHead(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.text_transform = nn.Linear(hidden_size, hidden_size)
        self.image_transform = nn.Linear(hidden_size, hidden_size)
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_norm(self, features):
        return features / features.norm(dim=-1, keepdim=True).float()

    def forward(self, image_feats, text_feats):
        # Normalize features
        images_features, texts_features = image_feats, text_feats
        images_features = self.compute_norm(self.image_transform(image_feats))
        texts_features = self.compute_norm(self.text_transform(text_feats))
        """
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * torch.einsum('bij,bj->bi', images_features, texts_features)
        """
        similarity = torch.einsum('bij,bj->bi', images_features, texts_features)

        return similarity