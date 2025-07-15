import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, Bottleneck

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):  # 原论文ratio=16，调整为8更适合YOLOv8n
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 使用1x1卷积代替全连接层，保持空间信息
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """适配YOLOv8的CBAM模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

        # YOLO风格初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

def insert_cbam(model):
    """在YOLOv8n的每个C3模块后插入CBAM"""
    for name, module in model.named_children():
        if isinstance(module, C2f):  # YOLOv8的C3模块现在是C2f
            # 在C2f后添加CBAM
            cbam = CBAM(module.c)
            model._modules[name] = nn.Sequential(module, cbam)
        elif hasattr(module, 'children'):
            # 递归处理子模块
            insert_cbam(module)

def create_yolov8n_cbam():
    # 加载官方预训练模型
    model = YOLO('yolov8n.pt').model

    # 插入CBAM模块
    insert_cbam(model)

    # 保存完整模型定义
    torch.save({'model': model,
        'model_def': CBAM  # 确保类定义被保存
    }, 'yolov8n_cbam.pt')

create_yolov8n_cbam()