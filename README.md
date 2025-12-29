# 超声图像年龄预测项目

基于深度学习的超声图像年龄预测系统，使用ResNet50等多种架构进行训练。

## 🚀 快速开始

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 配置Python环境
conda activate us
```

### 训练模型
```bash
# 单GPU训练
python train_mae.py --epochs 100 --batch-size 32

# 多GPU训练 (6卡)
torchrun --nproc_per_node=6 train_mae.py --batch-size 32

# 使用年龄分层抽样
python train_mae.py --use-age-stratify --epochs 100

# 256分辨率训练
python train_mae.py --image-size 256

# 集成训练（6个模型并行）
python train_mae.py --ensemble
```

### 评估模型
```bash
# 评估模型
python evaluate.py --model-path outputs/run_xxx/best_model.pth

# 集成预测
python predict_ensemble.py
```

## 📁 项目结构

```
usage_predict/
├── train_mae.py          # 统一训练脚本（支持单模型/集成/DDP）
├── dataset.py            # 数据集加载（支持年龄分层抽样）
├── model.py              # 模型定义
├── evaluate.py           # 模型评估
├── predict_ensemble.py   # 集成预测
├── requirements.txt      # Python依赖
├── docs/                 # 详细文档
├── scripts/              # 工具脚本
├── results/              # 训练结果（图表、摘要、最佳配置）
├── outputs/              # 完整训练输出（不上传Git）
└── data/                 # 数据集（不上传Git）
```

## 📊 最佳模型

- **训练运行**: run_20251226_182738_noturn
- **验证集MAE**: 6.67 years
- **架构**: ResNet50
- **特点**: 无水平翻转（更适合医学图像）
- **权重位置**: `outputs/run_20251226_182738_noturn/best_model.pth`

配置详情见 [`results/best_results/`](results/best_results/)

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | 训练参数详解和使用指南 |
| [docs/DATASET_OPTIMIZATION.md](docs/DATASET_OPTIMIZATION.md) | 数据集划分和增强策略 |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | 项目结构和文件说明 |

## 🔧 工具脚本

```bash
# 查看项目结构
bash scripts/show_structure.sh

# 数据集分析
python scripts/analyze_dataset.py

# 验证数据泄漏
python scripts/verify_no_leakage.py

# 可视化图像尺寸
python scripts/visualize_image_sizes.py

# 绘制误差分析图
python scripts/plot_age_error.py
```

## 🎯 核心特性

### 数据增强
- ✅ RandomRotation(±10°)
- ✅ ColorJitter(亮度/对比度 ±0.2)
- ❌ 无水平翻转（医学图像特性）

### 训练策略
- **损失函数**: MAE/MSE/SmoothL1/Huber可选
- **优化器**: Adam (lr=0.001)
- **学习率调度**: CosineAnnealingLR
- **数据划分**: 按subject ID分组（防止数据泄漏）
- **年龄分层**: 支持按10岁分组的分层抽样

### 模型架构
- ResNet50 (默认)
- EfficientNet-B0/B1
- ConvNeXt-Tiny
- MobileNetV3-Large
- RegNet

## 📈 性能记录

| 模型 | MAE | 训练日期 | 备注 |
|------|-----|---------|------|
| ResNet50 | **6.67** | 2025-12-26 | 🏆 最佳（无翻转） |
| ResNet50 | 6.69 | 2025-12-25 | 含翻转 |
| ResNet50 | 6.72 | 2025-12-26 | 256分辨率 |

## 💡 常见问题

**Q: 如何继续训练？**
```bash
python train_mae.py --resume outputs/run_xxx/checkpoint_epoch_50.pth
```

**Q: 如何查看训练历史？**
```bash
cat outputs/run_xxx/history.json
```

**Q: 如何使用不同损失函数？**
```bash
python train_mae.py --loss mse  # 或 smoothl1, huber
```

**Q: 如何调整学习率？**
```bash
python train_mae.py --lr 0.0001
```

## 📞 技术支持

- 训练问题: 查看 [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- 数据问题: 查看 [docs/DATASET_OPTIMIZATION.md](docs/DATASET_OPTIMIZATION.md)
- 项目结构: 查看 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

---

**最后更新**: 2025-12-29  
**版本**: v1.0
