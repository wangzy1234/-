import torch
import torch.nn as nn
from ultralytics import YOLO

# 定义
class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def add_se_to_yolo(model, target_layers=["model.4", "model.6"]):
    """
    在指定层后添加SE模块
    target_layers: 推荐选择Backbone中的C2f输出层
    """
    for name in target_layers:
        try:
            # 获取目标层
            parent = model.model
            for part in name.split('.'):
                parent = getattr(parent, part)

            # 获取输出通道数
            if hasattr(parent, 'cv2'):  # C2f模块
                out_channels = parent.cv2.conv.out_channels
            elif hasattr(parent, 'out_channels'):  # 普通卷积
                out_channels = parent.out_channels
            else:
                print(f"跳过 {name} (无法获取通道数)")
                continue

            # 添加SE模块
            se = SEBlock(out_channels)
            parent.add_module("se", se)
            #print(f"✅ 已在 {name} 添加SE (通道数: {out_channels})")

        except Exception as e:
            print(f"❌ {name} 添加失败: {str(e)}")


# 加载原始模型
model = YOLO("yolov8n.pt")

# 添加SE模块
add_se_to_yolo(model)

# 验证前向传播
test_input = torch.randn(1, 3, 640, 640)
try:
    output = model(test_input)
    #print("✅ 前向传播验证通过")
except Exception as e:
    print(f"❌ 验证失败: {str(e)}")


# 保存模型
model.save("yolov8n_se.pt")