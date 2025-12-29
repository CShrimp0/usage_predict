# 配置文件优化说明

## 📝 改进概述

优化了三个训练脚本（`train.py`, `train_ddp.py`, `train_mae.py`）的配置文件保存逻辑，现在会保存更详细和完整的训练信息。

---

## 🆚 改进前后对比

### ❌ 之前的配置文件（简单）

```json
{
  "image_dir": "/home/szdx/LNX/data/TA/Healthy/Images",
  "excel_path": "/home/szdx/LNX/data/TA/characteristics.xlsx",
  "output_dir": "./outputs",
  "model": "resnet50",
  "epochs": 500,
  "batch_size": 32,
  "lr": 0.0001,
  "timestamp": "20251225_120000",
  "device": "cuda",
  "world_size": 6,
  "use_ddp": true
}
```

**问题**：
- ❌ 缺少脚本名称和版本信息
- ❌ 没有数据集统计信息（受试者数量、样本分布）
- ❌ 没有记录损失函数类型
- ❌ 没有学习率调度器详情
- ❌ 缺少GPU信息
- ❌ 没有说明数据划分方式

---

### ✅ 优化后的配置文件（完整）

#### **train_mae.py 的配置示例**

```json
{
  "script_name": "train_mae.py",
  "script_version": "3.0",
  "timestamp": "20251225_164104",
  "description": "Optimized training with MAE loss, gradient clipping, and cosine annealing",
  
  "device": "cuda",
  "world_size": 6,
  "use_ddp": true,
  "gpu_names": [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090"
  ],
  "cuda_version": "12.8",
  "pytorch_version": "2.9.1+cu128",
  
  "dataset": {
    "image_dir": "/home/szdx/LNX/data/TA/Healthy/Images",
    "excel_path": "/home/szdx/LNX/data/TA/characteristics.xlsx",
    "total_samples": 3092,
    "train_samples": 2211,
    "val_samples": 251,
    "test_samples": 630,
    "train_subjects": 734,
    "val_subjects": 82,
    "test_subjects": 205,
    "total_subjects": 1021,
    "test_size": 0.2,
    "val_size": 0.1,
    "random_seed": 42,
    "split_method": "by_subject_id",
    "data_leakage_prevention": true
  },
  
  "model": {
    "architecture": "resnet50",
    "pretrained": true,
    "dropout": 0.5,
    "output_dim": 1,
    "task": "age_regression"
  },
  
  "training": {
    "loss_function": "MAE (L1Loss)",
    "optimizer": "AdamW",
    "optimizer_params": {
      "betas": [0.9, 0.999]
    },
    "lr_scheduler": "CosineAnnealingLR",
    "scheduler_params": {
      "T_max": 500,
      "eta_min": 1e-07
    },
    "warmup_epochs": 5,
    "max_grad_norm": 1.0,
    "epochs": 500,
    "patience": 100,
    "batch_size": 32,
    "effective_batch_size": 192,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_workers": 8,
    "plot_interval": 10
  },
  
  "optimizations": {
    "gradient_clipping": true,
    "warmup": true,
    "early_stopping": true,
    "cosine_annealing": true
  },
  
  "output_dir": "./outputs"
}
```

---

## 📊 新增字段说明

### 1️⃣ **脚本信息**
```json
{
  "script_name": "train_mae.py",          // 使用的训练脚本
  "script_version": "3.0",                 // 脚本版本号
  "description": "..."                     // 脚本描述
}
```

**作用**：快速识别实验使用的训练方法

### 2️⃣ **运行环境详情**
```json
{
  "gpu_names": ["NVIDIA GeForce RTX 4090", ...],  // 所有GPU型号
  "cuda_version": "12.8",                          // CUDA版本
  "pytorch_version": "2.9.1+cu128"                 // PyTorch版本
}
```

**作用**：
- 复现实验时确保环境一致
- 排查GPU相关问题
- 记录硬件配置

### 3️⃣ **数据集统计** ⭐重要
```json
{
  "dataset": {
    "total_samples": 3092,           // 总样本数
    "train_samples": 2211,           // 训练样本数
    "val_samples": 251,              // 验证样本数
    "test_samples": 630,             // 测试样本数
    "train_subjects": 734,           // 训练受试者数（新增）
    "val_subjects": 82,              // 验证受试者数（新增）
    "test_subjects": 205,            // 测试受试者数（新增）
    "total_subjects": 1021,          // 总受试者数（新增）
    "split_method": "by_subject_id", // 划分方式（新增）
    "data_leakage_prevention": true  // 防泄露标记（新增）
  }
}
```

**作用**：
- ✅ 确认数据集是否按受试者划分（防数据泄露）
- ✅ 了解数据分布（734:82:205 = 72%:8%:20%）
- ✅ 判断样本量是否充足
- ✅ 对比不同实验的数据使用情况

### 4️⃣ **模型详细配置**
```json
{
  "model": {
    "architecture": "resnet50",
    "pretrained": true,
    "dropout": 0.5,
    "output_dim": 1,              // 输出维度（新增）
    "task": "age_regression"      // 任务类型（新增）
  }
}
```

### 5️⃣ **训练配置详情**
```json
{
  "training": {
    "loss_function": "MAE (L1Loss)",     // 损失函数类型（新增）
    "optimizer": "AdamW",                // 优化器（新增）
    "optimizer_params": {...},           // 优化器参数（新增）
    "lr_scheduler": "CosineAnnealingLR", // 学习率调度器（新增）
    "scheduler_params": {...},           // 调度器参数（新增）
    "warmup_epochs": 5,                  // Warmup轮数（新增）
    "max_grad_norm": 1.0,                // 梯度裁剪阈值（新增）
    "effective_batch_size": 192,         // 有效批次大小（新增）
    "plot_interval": 10                  // 绘图间隔（新增）
  }
}
```

**作用**：
- 📊 知道用的是MSE还是MAE损失
- 📊 了解学习率调度策略
- 📊 确认是否使用梯度裁剪等优化技巧

### 6️⃣ **优化技巧标记**（仅 train_mae.py）
```json
{
  "optimizations": {
    "gradient_clipping": true,
    "warmup": true,
    "early_stopping": true,
    "cosine_annealing": true
  }
}
```

---

## 🔍 三个脚本的配置差异

### **train.py**
- **损失函数**: MSE
- **优化器**: Adam
- **调度器**: ReduceLROnPlateau
- **特点**: 基础版，支持单GPU和DDP

### **train_ddp.py**
- **损失函数**: MSE
- **优化器**: Adam
- **调度器**: ReduceLROnPlateau
  - `mode='min'`
  - `factor=0.5`
  - `patience=10`
- **特点**: 优化的DDP版本，10轮更新一次曲线图

### **train_mae.py** ⭐推荐
- **损失函数**: MAE (L1Loss)
- **优化器**: AdamW
  - `betas=(0.9, 0.999)`
- **调度器**: CosineAnnealingLR
  - `T_max=500`
  - `eta_min=1e-7`
- **特殊优化**:
  - ✅ 梯度裁剪 (max_norm=1.0)
  - ✅ Warmup (5 epochs)
  - ✅ 早停 (patience=100)
  - ✅ 更稳定的学习率 (1e-4)

---

## 💡 使用场景

### 1️⃣ **实验对比**
通过配置文件快速对比不同实验的设置：
```bash
# 查看某次实验的配置
cat outputs/run_20251225_164104/config.json

# 对比两次实验
diff outputs/run_A/config.json outputs/run_B/config.json
```

### 2️⃣ **复现实验**
根据配置文件精确复现实验：
```python
import json
with open('outputs/best_run/config.json') as f:
    config = json.load(f)

# 使用相同的参数
train_model(
    lr=config['training']['learning_rate'],
    batch_size=config['training']['batch_size'],
    loss_fn=config['training']['loss_function'],
    ...
)
```

### 3️⃣ **验证数据划分**
快速检查是否有数据泄露：
```python
import json
with open('config.json') as f:
    config = json.load(f)

if config['dataset']['split_method'] == 'by_subject_id':
    print("✅ 按受试者划分，无数据泄露")
else:
    print("⚠️  可能存在数据泄露")
```

### 4️⃣ **生成实验报告**
自动提取配置信息生成报告：
```python
# 自动生成markdown报告
report = f"""
## 实验配置
- **脚本**: {config['script_name']} v{config['script_version']}
- **时间**: {config['timestamp']}
- **损失函数**: {config['training']['loss_function']}
- **数据集**: {config['dataset']['total_subjects']}个受试者
- **训练样本**: {config['dataset']['train_samples']}
"""
```

---

## 📋 配置文件位置

每次训练运行时，配置文件自动保存在：
```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json          ← 完整的训练配置
    ├── history.json         ← 训练历史
    ├── best_model.pth       ← 最佳模型
    ├── checkpoint_epoch_*.pth
    └── training_curves.png
```

---

## ✅ 总结

### 主要改进
1. ✅ **脚本标识**: 明确记录使用的训练脚本和版本
2. ✅ **数据集详情**: 增加受试者统计，确认数据划分方式
3. ✅ **损失函数**: 明确记录MSE/MAE
4. ✅ **调度器详情**: 记录学习率调度策略和参数
5. ✅ **优化技巧**: 标记使用的优化方法（梯度裁剪、warmup等）
6. ✅ **环境信息**: GPU型号、CUDA版本、PyTorch版本

### 好处
- 🔍 **可追溯性**: 每次实验的所有关键信息都有记录
- 🔄 **可复现性**: 根据配置文件可精确复现实验
- 📊 **便于对比**: 快速对比不同实验的设置差异
- ✅ **质量保证**: 确认数据划分方式，防止数据泄露

---

**更新日期**: 2025-12-25  
**影响文件**: `train.py`, `train_ddp.py`, `train_mae.py`
