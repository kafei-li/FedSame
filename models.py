import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, mobilenet_v2

class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_mnist_model(num_classes=10):
    return MNISTNet(num_classes=num_classes)

class CelebAResNet18(nn.Module):
    def __init__(self, num_classes=40):
        super(CelebAResNet18, self).__init__()
        self.backbone = resnet18(pretrained=True)
        # 冻结前16层参数
        ct = 0
        for child in self.backbone.children():
            ct += 1
            if ct <= 6:  # conv1, bn1, layer1, layer2, layer3, layer4
                for param in child.parameters():
                    param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # 40个独立二分类头
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ) for _ in range(num_classes)])
    def forward(self, x):
        feat = self.backbone(x)
        outs = [torch.sigmoid(head(feat)) for head in self.heads]
        outs = torch.cat(outs, dim=1)  # [B, 40]
        return outs

def get_celeba_model(num_classes=40):
    return CelebAResNet18(num_classes=num_classes)


class OphthalmicMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=True).features[:-3]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


        self.classification_head = nn.Sequential(
            nn.Linear(self.backbone.last_channel, num_classes),
            nn.Sigmoid()
        )


        self.amd_head = nn.Linear(self.backbone.last_channel, 3)
        self.dr_head = nn.Linear(self.backbone.last_channel, 5)

    def forward(self, x, task_type):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if task_type == "classification":
            return self.classification_head(x)
        elif task_type == "AMD_grading":
            return self.amd_head(x)
        elif task_type == "DR_grading":
            return self.dr_head(x)

def get_ophthalmic_model(num_classes=10):
    return OphthalmicMobileNetV2(num_classes=num_classes)

class SyntheticNet(nn.Module):
    def __init__(self, num_tasks=4, num_classes=4):
        super(SyntheticNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.shared_fc = nn.Linear(32 * 14 * 14, 64)
        # 每个任务一个独立头
        self.task_heads = nn.ModuleList([nn.Linear(64, num_classes) for _ in range(num_tasks)])
    def forward(self, x, task_id=0):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.shared_fc(x))
        out = self.task_heads[task_id](x)
        return out

def get_synthetic_model(num_tasks=4, num_classes=4):
    return SyntheticNet(num_tasks=num_tasks, num_classes=num_classes)