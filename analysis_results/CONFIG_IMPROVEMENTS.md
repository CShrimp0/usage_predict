# 配置文件优化说明

## 📝 改进概述

优化了训练脚本 `train.py` 的配置文件保存逻辑，现在会保存更详细和完整的训练信息。

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
  "device": "cuda"
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

```json
{
  "script_name": "train.py",
  "script_version": "4.0",
  "timestamp": "20251226_182738",
  "description": "Unified training script with MAE loss, gradient clipping, and cosine annealing",
  
  "device": "cuda",
  "world_size": 6,
  "use_ddp": false,
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
    "image_size": 224,
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
    "data_leakage_prevention": true,
    "use_age_stratify": false,
    "horizontal_flip": false
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
  "script_name": "train.py",               // 使用的训练脚本
  "script_version": "4.0",                 // 脚本版本号
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

### 3️⃣ **数据集完整统计**
```json
{
  "dataset": {
    "total_samples": 3092,        // 总图像数
    "train_samples": 2211,        // 训练集图像数
    "val_samples": 251,           // 验证集图像数
    "test_samples": 630,          // 测试集图像数
    "total_subjects": 1021,       // 总受试者数
    "train_subjects": 734,        // 训练集受试者数
    "val_subjects": 82,           // 验证集受试者数
    "test_subjects": 205,         // 测试集受试者数
    "split_method": "by_subject_id",           // 数据划分方式
    "data_leakage_prevention": true,           // 是否防止数据泄露
    "use_age_stratify": false,                 // 是否使用年龄分层
    "horizontal_flip": false                   // 是否使用水平翻转
  }
}
```

**作用**：
- 了解数据集规模和分布
- 确认数据划分策略
- 验证是否防止数据泄露
- 记录数据增强策略

### 4️⃣ **模型架构详情**
```json
{
  "model": {
    "architecture": "resnet50",   // 模型类型
    "pretrained": true,           // 是否使用预训练权重
    "dropout": 0.5,               // Dropout比例
    "output_dim": 1,              // 输出维度
    "task": "age_regression"      // 任务类型
  }
}
```

**作用**：明确模型配置

### 5️⃣ **训练超参数详情**
```json
{
  "training": {
    "loss_function": "MAE (L1Loss)",      // 损失函数
    "optimizer": "AdamW",                  // 优化器
    "lr_scheduler": "CosineAnnealingLR",  // 学习率调度器
    "scheduler_params": {...},             // 调度器参数
    "warmup_epochs": 5,                    // 预热轮数
    "max_grad_norm": 1.0,                  // 梯度裁剪阈值
    "patience": 100,                       // 早停耐心值
    "effective_batch_size": 192            // 有效批大小（多GPU）
  }
}
```

**作用**：
- 记录完整训练策略
- 便于调参和对比实验

### 6️⃣ **优化技巧标记**
```json
{
  "optimizations": {
    "gradient_clipping": true,    // 是否启用梯度裁剪
    "warmup": true,               // 是否启用学习率预热
    "early_stopping": true,       // 是否启用早停
    "cosine_annealing": true      // 是否启用余弦退火
  }
}
```

**作用**：快速了解使用了哪些优化技巧

---

## 💡 使用建议

### 1. 复现实验
查看 `config.json` 文件，了解实验的完整配置：
```bash
cat outputs/run_YYYYMMDD_HHMMSS/config.json
```

### 2. 对比不同实验
比较两次实验的配置差异：
```bash
diff outputs/run_A/config.json outputs/run_B/config.json
```

### 3. 调参参考
在新实验中参考之前最优配置：
```bash
# 复制最佳配置作为起点
cp outputs/best_run/config.json my_experiment_config.json
```

---

## 📁 受影响的文件

**更新文件**: `train.py`

**生成位置**: `outputs/run_YYYYMMDD_HHMMSS/config.json`

---

## ✅ 最佳实践

### **推荐使用 train.py**
- ✅ 统一的训练接口
- ✅ 支持单GPU/多GPU/DDP训练
- ✅ 支持多种损失函数（MAE/MSE/SmoothL1/Huber）
- ✅ 完整的配置记录
- ✅ 自动保存详细训练历史
- ✅ 支持年龄分层采样
- ✅ 防止数据泄露的by-subject划分

### 典型训练命令

**单GPU训练**：
```bash
python train.py --model resnet50 --loss mae --epochs 500 --batch-size 32
```

**多GPU训练**（指定GPU）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --model resnet50 --loss mae --batch-size 96
```

**DDP训练**（分布式）：
```bash
torchrun --nproc_per_node=6 train.py --model resnet50 --loss mae --use-ddp
```

**年龄分层训练**：
```bash
python train.py --use-age-stratify --age-bin-width 10
```

---

## 🔄 更新历史

- **2025-12-25**: 初始版本，统一配置文件格式
- **2025-12-26**: 添加水平翻转和年龄分层配置项
- **2025-12-29**: 简化为单一train.py脚本，删除冗余文件

---

## 📌 注意事项

1. **重要性**: `config.json` 对于实验复现至关重要，请妥善保存
2. **版本控制**: 每次重大更新脚本后，应更新 `script_version` 字段
3. **备份**: 建议将最佳模型的 `config.json` 单独备份到 `results/best_results/` 目录
