import torch.nn as nn
import timm


class bengalimodel(nn.Module):
    def __init__(self, backbone = 'resnet18'):
        super(bengalimodel,self).__init__()
        self.backbone = timm.create_model(backbone, pretrained = True)
        self.l1 = nn.Linear(1000, 168)
        self.l2 = nn.Linear(1000, 11)
        self.l3 = nn.Linear(1000, 7)
        
    def forward(self,x):
        x = self.backbone(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        
        return l1,l2,l3
    