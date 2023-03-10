import torch
import torch.nn as nn
import timm
from package.loss import *
from package.resnet20 import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
def initModel(model_type, head_type, num_classes, feature_dim, loss_func, lamb, lr=0.01, l2=0, mm=0.9, feature_norm=False):
    if model_type == "ResNet20_basic":
        model = resnet20_basic(feature_dim)
    elif model_type == "ResNet18":
        model = models.resnet18(pretrained = False)
        model.avgpool = Identity()
        model.fc = nn.Linear(model.fc.in_features*7*7, feature_dim) 
    elif model_type == "ResNet18_default":
        model = models.resnet18(pretrained = False)
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "pretrained_ResNet18":
        model = models.resnet18(pretrained = True)
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "reduced_ResNet18":
        model = models.resnet18(pretrained = True)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, feature_dim) 
    elif model_type == "SE_ResNeXt26d_pretrained":
        model = timm.create_model('seresnext26d_32x4d', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "SE_ResNeXt26d":
        model = timm.create_model('seresnext26d_32x4d')
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "ResNeSt26d_pretrained":
        model = timm.create_model('resnest26d', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "Reduced_ResNeSt26d_pretrained":
        model = timm.create_model('resnest26d', pretrained=True)
        model.layer4 = Identity()
        model.fc = nn.Linear(1024, feature_dim) 
    elif model_type == "Double_Reduced_ResNeSt26d_pretrained":
        model = timm.create_model('resnest26d', pretrained=True)
        model.layer4 = Identity()
        model.layer3 = Identity()
        model.fc = nn.Linear(512, feature_dim) 
    elif model_type == "ResNeSt26d":
        model = timm.create_model('resnest26d')
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "ResNeSt14d_pretrained":
        model = timm.create_model('resnest14d', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "ResNeSt50d_pretrained":
        model = timm.create_model('resnest50d', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, feature_dim) 
    elif model_type == "Reduced_ResNeSt50d_pretrained":
        model = timm.create_model('resnest50d', pretrained=True)
        model.layer4 = Identity()
        model.fc = nn.Linear(1024, feature_dim)
    elif model_type == "Double_Reduced_ResNeSt50d_pretrained":
        model = timm.create_model('resnest50d', pretrained=True)
        model.layer3 = Identity()
        model.layer4 = Identity()
        model.fc = nn.Linear(512, feature_dim) 
    else:
        raise BaseException("Invalid model type")

    if head_type == "LMCL":
        head = LMCL(in_features=feature_dim, out_features=num_classes, s=30.0, m=0.65, centers=False)
        # head = LMCL_loss(num_classes, feature_dim, s=30.00, m=0.65)
    elif head_type == "CurricularFace":
        head = CurricularFace(in_features=feature_dim, out_features=num_classes, s=30.0, m=0.65, centers=False)
    elif head_type == "ArcFace":
        head = ArcFace(in_features=feature_dim, out_features=num_classes, s=30.0, m=0.65, centers=False)
    else:
        raise BaseException("Invalid loss function type")
        
    if loss_func == "CE+Center":
        criterion = {"Softmax": nn.CrossEntropyLoss(), "CenterLoss": CenterLoss(), "Lambda": lamb}
    elif loss_func == "Focal+Center":
        criterion = {"Softmax": FocalLoss(gamma=2), "CenterLoss": CenterLoss(), "Lambda": lamb}
    elif loss_func == "Focal+Huber":
        criterion = {"Softmax": FocalLoss(gamma=2), "CenterLoss": CenterHuberLoss(delta=12), "Lambda": lamb}
    elif loss_func == "CE+Huber":
        criterion = {"Softmax": nn.CrossEntropyLoss(), "CenterLoss": CenterHuberLoss(delta=10), "Lambda": lamb}
    else:
        raise BaseException("Invalid loss function type")
    

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mm)
    optimizer4center = torch.optim.Adam(head.parameters(), lr=0.1, weight_decay=l2)
    return model, head, criterion, optimizer, optimizer4center