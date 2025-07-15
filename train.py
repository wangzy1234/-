import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import SE_attention
import CBAM_attention

# ---  模型训练与超参数调优 ---
def train_model():
    # 加载自定义模型
    #model = YOLO("yolov8n.pt")  # 未添加注意力机制
    #model = YOLO("yolov8n_se.pt")  # 添加se
    model = YOLO("yolov8n_cbam.pt")  # 添加cbam

    # 训练参数配置
    results = model.train(
        data='datasets/data.yaml',
        epochs=100,
        batch=32,  # 根据GPU显存调整
        imgsz=256,  # 图像尺寸
        optimizer="AdamW",  # 可选SGD/AdamW
        lr0=0.002,  # 初始学习率
        lrf=0.01,  # 最终学习率 = lr0 * lrf
        warmup_epochs=3,  # 添加学习率热身（避免初期不稳定）
        workers=4,  # 多进程
        patience=10,  # 早停等待epoch数
        device="cpu",  # 使用CPU
        name="fire_detection",
        hsv_h=0.1,  # 色调增强强度
        hsv_s=0.7,  # 饱和度增强
        flipud=0.5,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        mosaic=0.5,  # Mosaic数据增强
        exist_ok = True,  # 允许覆盖已有训练结果
    )
    return model

# ---  模型评估 ---
def evaluate_model(model):
    # 验证集评估
    val_metrics = model.val(
        data='datasets/data.yaml',
        conf=0.5,  # 置信度阈值
        iou=0.5,  # IoU阈值
        save_json=True  # 保存JSON结果
    )

    # 测试集独立评估
    test_metrics = model.val(
        data='datasets/data.yaml',
        split="test",
        save_conf=True,  # 保存置信度
        save_hybrid=True  # 保存混合标签
    )

    # 添加混淆矩阵计算
    from ultralytics.utils.metrics import ConfusionMatrix
    cm = ConfusionMatrix(nc=2)
    cm.process_batch(test_metrics.pred, test_metrics.targets)

    return val_metrics, test_metrics,cm


# --- 主流程 ---
if __name__ == "__main__":
    # 训练模型
    model = train_model()
    # 评估模型
    val_metrics, test_metrics, cm = evaluate_model(model)
    print(f"验证集 mAP@0.5: {val_metrics.box.map:.3f}")
    print(f"测试集 mAP@0.5: {test_metrics.box.map:.3f}")