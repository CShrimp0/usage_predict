# 训练脚本使用指南

## 概述

`train_mae.py` 是统一的训练脚本，支持：
- ✅ 单模型训练 / 集成训练
- ✅ 多种损失函数 (MAE, MSE, Smooth L1, Huber)
- ✅ 不同图像分辨率 (224×224, 256×256)
- ✅ 单GPU / 多GPU (DDP) 训练
- ✅ 完整的训练监控和可视化

## 快速开始

### 1. 单GPU训练（默认MAE损失，224×224）
```bash
python train_mae.py --batch-size 96
```

### 2. 多GPU训练（DDP）
```bash
torchrun --nproc_per_node=6 train_mae.py --batch-size 32
```

### 3. 使用MSE损失
```bash
python train_mae.py --loss mse --batch-size 96
```

### 4. 使用256×256分辨率
```bash
python train_mae.py --image-size 256 --batch-size 64
```

### 5. 使用年龄分层抽样（提高泛化能力）
```bash
python train_mae.py --use-age-stratify --age-bin-width 10 --batch-size 96
```

### 6. 集成训练（6个模型并行）
```bash
python train_mae.py --ensemble \
    --ensemble-models resnet50 efficientnet_b0 efficientnet_b1 convnext mobilenet_v3 regnet \
    --ensemble-gpus 0 1 2 3 4 5 \
    --batch-size 96
```

## 完整参数列表

### 数据参数
- `--image-dir`: 图像文件夹路径（默认: `/home/szdx/LNX/data/TA/Healthy/Images`）
- `--excel-path`: Excel标签文件路径（默认: `/home/szdx/LNX/data/TA/characteristics.xlsx`）
- `--output-dir`: 输出目录（默认: `./outputs`）
- `--test-size`: 测试集比例（默认: 0.2）
- `--val-size`: 验证集比例（默认: 0.1）
- `--seed`: 随机种子（默认: 42）
- `--use-age-stratify`: 启用按年龄分层抽样（提高泛化能力）
- `--age-bin-width`: 年龄分组宽度（岁），默认10岁一组

### 模型参数
- `--model`: 模型架构
  - 可选: `resnet50`, `efficientnet_b0`, `efficientnet_b1`, `convnext`, `mobilenet_v3`, `regnet`
  - 默认: `resnet50`
- `--loss`: 损失函数类型
  - 可选: `mae`, `mse`, `smoothl1`, `huber`
  - 默认: `mae`
- `--image-size`: 输入图像尺寸
  - 可选: `224`, `256`
  - 默认: `224`
- `--pretrained`: 使用ImageNet预训练权重（默认: True）
- `--dropout`: Dropout比例（默认: 0.5）

### 训练参数
- `--epochs`: 最大训练轮数（默认: 500）
- `--patience`: 早停耐心值（默认: 100）
- `--batch-size`: 每个GPU的批次大小（默认: 32）
- `--lr`: 初始学习率（默认: 1e-4）
- `--weight-decay`: 权重衰减（默认: 1e-4）
- `--num-workers`: 数据加载线程数（默认: 8）

### 学习率调度参数
- `--eta-min`: 最小学习率（默认: 1e-7）
- `--warmup-epochs`: Warmup轮数（默认: 5）
- `--max-grad-norm`: 梯度裁剪阈值（默认: 1.0）

### 集成训练参数
- `--ensemble`: 启用集成训练模式
- `--ensemble-models`: 集成训练的模型列表（默认: 6个模型）
- `--ensemble-gpus`: 集成训练使用的GPU列表（默认: [0,1,2,3,4,5]）

## 数据集划分策略

### 默认模式：按受试者ID划分
- 防止数据泄露（同一受试者的图像不会同时出现在训练集和测试集）
- 随机划分，不考虑年龄分布

### 年龄分层抽样模式（推荐）
- 按年龄段分层抽样（默认每10岁一组：0-10, 10-20, ..., 80-90）
- **优势**：确保各年龄段在训练/验证/测试集中的比例一致
- **提高泛化能力**：避免某个年龄段被过度采样或欠采样
- 使用方法：添加 `--use-age-stratify` 参数

```bash
# 年龄分层抽样（每10岁一组）
python train_mae.py --use-age-stratify --age-bin-width 10

# 更细粒度的分层（每5岁一组）
python train_mae.py --use-age-stratify --age-bin-width 5
```

## 损失函数对比

| 损失函数 | 特点 | 适用场景 |
|---------|------|---------|
| **MAE (L1)** | 对异常值鲁棒 | 存在噪声数据时推荐 ✅ |
| **MSE (L2)** | 惩罚大误差 | 数据质量好时可用 |
| **Smooth L1** | L1和L2的结合 | 折中方案 |
| **Huber** | 自适应鲁棒损失 | 复杂场景 |

## 图像分辨率对比

| 分辨率 | 显存占用 | 训练速度 | 性能 |
|--------|---------|---------|------|
| **224×224** | 基准 | 快 | 基准 |
| **256×256** | +33% | -25% | 略好 |

**建议**：
- 6×RTX 4090 (48GB): 256×256 + batch_size=64-96
- 单GPU训练: 224×224 + batch_size=96-128

## 输出结构

```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── best_model.pth           # 最佳模型权重
    ├── checkpoint_epoch_*.pth   # 定期检查点（每10轮）
    ├── training_curves.png      # 训练曲线图
    ├── history.json             # 训练历史数据
    └── config.json              # 完整配置信息
```

集成训练输出：
```
outputs/
└── ensemble_YYYYMMDD_HHMMSS/
    ├── resnet50/
    │   ├── best_model.pth
    │   ├── training.log
    │   └── ...
    ├── efficientnet_b0/
    └── ...
```

## 训练监控

训练过程中会实时显示：
- ✅ Loss / MAE
- ✅ 梯度范数
- ✅ 学习率变化
- ✅ 验证集性能
- ✅ 最佳模型更新

每10轮自动绘制训练曲线并保存检查点。

## 常见问题

### Q: 如何选择batch size？
**A**: 根据GPU显存：
- RTX 4090 (48GB): 224→128, 256→64-96
- 多GPU (DDP): 总batch = batch_size × GPU数量

### Q: 训练太慢/太快？
**A**: 调整学习率和warmup：
- 慢: 提高 `--lr` (默认1e-4 → 2e-4)
- 快: 增加 `--warmup-epochs` (默认5 → 10)

### Q: 过拟合怎么办？
**A**: 增强正则化：
- 提高 `--dropout` (0.5 → 0.6-0.7)
- 提高 `--weight-decay` (1e-4 → 5e-4)
- 启用更多数据增强（修改dataset.py）

### Q: 集成训练GPU分配？
**A**: 
```bash
# 自动循环分配（推荐）
--ensemble-gpus 0 1 2 3 4 5

# 手动指定（6个模型需要6个GPU ID）
--ensemble-gpus 0 0 1 1 2 2  # 每张GPU跑2个模型
```

## 最佳实践

1. **首次训练**: 使用默认参数测试
   ```bash
   python train_mae.py --batch-size 96
   ```

2. **性能优化**: 尝试不同损失函数
   ```bash
   python train_mae.py --loss mae  # 基准
   python train_mae.py --loss mse  # 对比
   ```

3. **分辨率实验**: 对比224 vs 256
   ```bash
   python train_mae.py --image-size 224 --batch-size 96
   python train_mae.py --image-size 256 --batch-size 64
   ```

4. **集成学习**: 最终模型
   ```bash
   python train_mae.py --ensemble --batch-size 96
   ```

## 结果可视化

训练完成后，使用以下脚本分析结果：

```bash
# 绘制误差分析图
python plot_age_error.py --model-path outputs/run_YYYYMMDD_HHMMSS/best_model.pth

# 集成预测
python predict_ensemble.py --ensemble-dir outputs/ensemble_YYYYMMDD_HHMMSS
```

## 脚本维护说明

### 已整合的脚本
原有的多个训练脚本和数据集模块已合并：
- ~~`train.py`~~ (MSE损失) → `--loss mse`
- ~~`train_ensemble.py`~~ → `--ensemble`
- ~~`train_mae_256.py`~~ → `--image-size 256`
- ~~`train_ddp.py`~~ → `torchrun`
- ~~`train_mae_noturn.py`~~ → 已删除（重复）
- ~~`dataset_256.py`~~ → 已合并到 `dataset.py`

### 数据集模块
统一的 `dataset.py` 支持：
- **可配置图像尺寸**: 224×224, 256×256或任意尺寸
- **两种划分策略**: 
  - 按受试者ID随机划分（默认）
  - 按年龄分层抽样（`use_age_stratify=True`）
- **灵活的年龄分组**: 可调节分组宽度（默认10岁）

修改数据增强：编辑 `dataset.py` 中的 `train_transform`。

