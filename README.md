# 超声图像年龄预测（usage_predict）

简洁、可复现的超声图像年龄回归训练框架，支持单/多GPU训练、年龄分层抽样、CLAHE 预处理、Top‑3 checkpoint 保存与可复现的命令化实验。

## 🚀 快速开始

### 环境
```bash
# 创建并激活环境
conda create -n us python=3.10 -y
conda activate us

# 安装依赖
pip install -r requirements.txt
```

### 数据准备
- 将图像放入 `data/TA/Healthy/Images`（或在命令行指定 `--image-dir`）
- Excel 标签文件放入 `data/TA/characteristics.xlsx`（或用 `--excel-path` 指定）

### Baseline 训练（完整示例）
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model resnet50 \
  --batch-size 32 \
  --dropout 0.5 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --loss mae \
  --epochs 500 \
  --patience 100 \
  --clahe 0 \
  --image-size 224 \
  --age-bin-width 10 \
  --seed 42 \
  --num-workers 8
```

### 评估模型
```bash
# 使用独立脚本评估某个 checkpoint
python evaluate.py --model-path outputs/run_xxx/best_model.pth --excel-path data/TA/characteristics.xlsx
```

## 📁 项目结构（简化）
```
usage_predict/
├─ train.py            # 主训练脚本（参数化、支持DDP）
├─ dataset.py          # 数据集与变换（CLAHE、年龄分层）
├─ model.py            # 模型定义与构造器
├─ evaluate.py         # 单独评估脚本（不在train结束自动运行）
├─ requirements.txt
├─ scripts/            # 辅助脚本（可视化/分析等）
└─ outputs/            # 训练产物（每次run的文件夹）
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
- ✅ RandomHorizontalFlip(p=0.5) - 默认启用
- ✅ ColorJitter(亮度/对比度 ±0.2)

> 注：如需禁用水平翻转，请手动修改 `dataset.py` 第354行

### 训练策略
- **损失函数**: MAE/MSE/SmoothL1/Huber 可选
- **优化器**: AdamW（默认 lr=1e-4）
- **学习率调度**: CosineAnnealingLR（支持线性 warmup，默认 warmup 5 epochs）
- **数据划分**: 按 subject ID 分组（防止数据泄漏）
- **年龄分层**: 支持按 10 岁分组的分层抽样（默认启用）

### 模型架构
- ResNet50 (默认)
- EfficientNet-B0/B1
- ConvNeXt-Tiny
- MobileNetV3-Large
- RegNet

## 📈 历史与近期结果
- **近期最佳（迭代记录）**：Dropout=0.6, Val MAE **7.016**（run_20260106_161415）
- Baseline (dropout=0.5, no flip): Val MAE **7.050**（run_20260106_154254）
- +CLAHE: Val MAE **7.120**（run_20260106_154708）

> 注：`config.json` 中会保存每次运行的全部参数，所有对比请以 `config.json` 为准。

## 💡 常见问题

**Q: 如何继续训练？**
```bash
python train.py --resume outputs/run_xxx/checkpoint_epoch_50.pth
```

**Q: 如何查看训练历史？**
```bash
cat outputs/run_xxx/history.json
```

**Q: 如何使用不同损失函数？**
```bash
python train.py --loss mse  # 或 smoothl1, huber
```

**Q: 如何调整学习率？**
```bash
python train.py --lr 0.0001
```

## 📞 技术支持

- 训练问题: 查看 [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- 数据问题: 查看 [docs/DATASET_OPTIMIZATION.md](docs/DATASET_OPTIMIZATION.md)
- 项目结构: 查看 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

---

**最后更新**: 2026-01-07  
**版本**: v1.1
