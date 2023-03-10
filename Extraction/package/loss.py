import torch
import torch.nn as nn
import numpy as np
import math

## Loss functions
def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, s = 64., m = 0.5, centers=False):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.weight, std=0.01)

    def classify(self, x, centers=None):   
        weights = self.weight
        if torch.is_tensor(centers):
            weights = centers     
        logits = F.linear(F.normalize(x), F.normalize(weights))
        return self.s * logits

    def forward(self, embbedings, label, centers=None):
        weights = self.weight
        if torch.is_tensor(centers) and centers.shape == self.weight.shape:
            weights = centers
        
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(weights, axis = 1)
        cos_theta = torch.mm(embbedings, torch.transpose(kernel_norm, 0, 1))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        margin_output = cos_theta * self.s
        original_logits = origin_cos * self.s
        return margin_output, original_logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, in_features=128, out_features=10575, s=32.0, m=0.50, easy_margin=False, centers=False):
        super(ArcFace, self).__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        with torch.no_grad():
            origin_cos = cosine.clone()
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        margin_output = origin_cos * self.s
        return margin_output, output
    
class LMCL(nn.Module):
    def __init__(self, in_features=128, out_features=600, s=30.0, m=0.65, centers=False):
        super(LMCL, self).__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.s = s
        self.m = m
        # self.featureMapping_90 = nn.Linear(feature_dim, feature_dim)
        # self.featureMapping_180 = nn.Linear(feature_dim, feature_dim)
        # self.featureMapping_270 = nn.Linear(feature_dim, feature_dim)
        if torch.is_tensor(centers) and centers.shape == (out_features, in_features):
            self.weight = nn.Parameter(centers)
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            # nn.init.xavier_uniform_(self.weight)
            # nn.init.kaiming_uniform_(self.weight)
            # nn.init.normal_(self.weight, std=0.01)

    def classify(self, x, centers=None):   
        weights = self.weight
        if torch.is_tensor(centers):
            weights = centers     
        logits = F.linear(F.normalize(x), F.normalize(weights))
        return self.s * logits

    # def centers(self):     
    #     weights_90 = self.featureMapping_90(self.weight)
    #     weights_180 = self.featureMapping_180(self.weight)
    #     weights_270 = self.featureMapping_270(self.weight)
    #     weights = torch.cat([self.weight, weights_90, weights_180, weights_270], 0)
    #     return weights
    
    def forward(self, x, label, centers=None):
        weights = self.weight
        if torch.is_tensor(centers) and centers.shape == self.weight.shape:
            weights = centers
        # else:
        #     weights = self.centers()
        cosine = F.linear(F.normalize(x), F.normalize(weights))
        with torch.no_grad():
            origin_cos = cosine.clone()
            
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        margin_output = self.s * (cosine - one_hot * self.m)
        original_logits = self.s * origin_cos
        return margin_output, original_logits
    
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()

    def forward(self, feats, label, centers):
        center = centers[label]
        dist = (feats-center).pow(2).sum(dim=-1) / 2
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        return loss

class CenterHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(CenterHuberLoss, self).__init__()
        self.HuberLoss = nn.HuberLoss(delta=delta)
        self.delta = delta

    def forward(self, feats, label, centers):
        center = centers[label]
        # dist = (feats-center).pow(2).sum(dim=-1) / 2
        dist = self.HuberLoss(feats, center)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()