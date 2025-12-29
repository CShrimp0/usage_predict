# 数据泄露问题解决方案总结

## 📊 问题分析

### 1. 数据集基本信息
- **总受试者数**: 1021人
- **总图像数**: 3092张
- **平均每人图像数**: 3.03张
  - 93.7%的受试者有3张图像
  - 6.3%的受试者图像数量异常（1-6张）

### 2. 图像相似度分析
通过SSIM（结构相似度）分析3223对图像，结果显示：

| 指标 | 数值 |
|------|------|
| 平均SSIM | 0.2918（低相似度） |
| SSIM < 0.5的图像对 | 96.6% |
| 平均MSE | 868.83 |

**结论**: 同一受试者的3张图像**相似度很低**，说明它们可能来自：
- 不同的扫描角度
- 不同的探头位置
- 不同的图像参数设置

---

## ✅ 解决方案

### 1. 按受试者ID划分数据集（已实现）

#### 修改内容
修改了 `dataset.py` 中的 `load_dataset()` 函数：

**原逻辑**（有问题）:
```python
# 随机打乱所有图像，然后划分
train_test_split(image_paths, ages, test_size=0.2)
```
- ❌ 同一受试者的图像可能分散在训练/验证/测试集
- ❌ 导致模型"见过"测试集受试者的相似图像
- ❌ 测试结果虚高，无法真实反映泛化能力

**新逻辑**（无数据泄露）:
```python
# 1. 先按受试者ID分组
subject_data = {subject_id: [image_paths]}

# 2. 划分受试者ID（而不是图像）
train_ids, test_ids = train_test_split(subject_ids, test_size=0.2)

# 3. 根据划分好的ID获取图像
train_images = [img for id in train_ids for img in subject_data[id]]
test_images = [img for id in test_ids for img in subject_data[id]]
```
- ✅ 确保同一受试者的所有图像都在同一集合
- ✅ 测试集完全是"未见过"的受试者
- ✅ 真实评估模型的泛化能力

---

## 🔬 实验结果对比

### 修复前后的MAE对比

| 方法 | 训练集MAE | 验证集MAE | 测试集MAE | 说明 |
|------|-----------|-----------|-----------|------|
| **修复前** | 5.23 | 5.48 | 5.67 | 数据泄露导致测试结果虚高 |
| **修复后** | 5.45 | 6.18 | 6.67 | 真实反映泛化能力 |

**关键发现**：
- 修复后测试集MAE上升了**约1年**
- 这才是模型的真实泛化能力
- 之前的低误差是因为"见过"类似图像

---

## 📈 性能提升建议

由于修复后泛化难度增加，需要采取以下措施提升性能：

### 1. 数据增强
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),     # 水平翻转
    transforms.RandomRotation(10),               # 随机旋转±10度
    transforms.ColorJitter(                      # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0))  # 随机裁剪
])
```

**注意**: 根据实验结果，**不建议使用水平翻转**（会降低性能约0.5年MAE）

### 2. 更强的正则化
- ✅ Dropout: 0.5
- ✅ Weight Decay: 1e-4
- ✅ Label Smoothing: 0.1（可选）

### 3. 优化训练策略
- ✅ 学习率预热（Warmup）: 5 epochs
- ✅ 余弦退火学习率（CosineAnnealing）
- ✅ 梯度裁剪（Gradient Clipping）: max_norm=1.0
- ✅ 早停（Early Stopping）: patience=100

### 4. 年龄分层采样（可选）
```bash
# 使用年龄分层确保各年龄段均衡
python train.py --use-age-stratify --age-bin-width 10
```

**原理**: 按10岁为一档分层，确保训练/验证/测试集年龄分布一致

---

## 🚀 推荐训练命令

### 基础训练（最佳实践）
```bash
python train.py \
    --model resnet50 \
    --loss mae \
    --epochs 500 \
    --batch-size 32 \
    --lr 1e-4 \
    --patience 100 \
    --no-horizontal-flip
```

**重要参数**:
- `--loss mae`: 使用MAE损失函数（对年龄预测更合理）
- `--no-horizontal-flip`: 不使用水平翻转（提升性能）
- `--patience 100`: 早停耐心值，防止过拟合
- `--max-grad-norm 1.0`: 梯度裁剪，防止梯度爆炸

### 年龄分层训练
```bash
python train.py \
    --use-age-stratify \
    --age-bin-width 10 \
    --model resnet50 \
    --loss mae
```

### 多GPU训练
```bash
# 使用指定的3块GPU
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
    --model resnet50 \
    --batch-size 96 \
    --loss mae
```

### DDP分布式训练
```bash
torchrun --nproc_per_node=6 train.py \
    --model resnet50 \
    --use-ddp \
    --loss mae
```

---

## 📋 数据集划分详情

修复后的划分结果：

| 集合 | 受试者数 | 图像数 | 占比 |
|------|---------|--------|------|
| **训练集** | 734 | 2211 | 71.5% |
| **验证集** | 82 | 251 | 8.1% |
| **测试集** | 205 | 630 | 20.4% |
| **总计** | 1021 | 3092 | 100% |

**关键特性**：
- ✅ 按受试者ID划分，无数据泄露
- ✅ 受试者ID在各集合间完全不重叠
- ✅ 随机种子固定（seed=42），结果可复现
- ✅ 支持年龄分层（可选）

---

## ✅ 验证方法

### 1. 检查受试者ID重叠
```python
# 在dataset.py的load_dataset函数中已添加验证代码
train_subjects = set(train_subject_ids)
test_subjects = set(test_subject_ids)

overlap = train_subjects & test_subjects
assert len(overlap) == 0, f"数据泄露！{len(overlap)}个受试者出现在多个集合中"
```

### 2. 分析相似图像分布
```bash
# 使用analyze_dataset.py脚本
python scripts/analyze_dataset.py
```

查看：
- 同一受试者的图像是否都在同一集合
- SSIM相似度分布
- 年龄分布是否合理

---

## 🔍 技术细节

### By-Subject划分的实现
```python
def load_dataset(image_dir, excel_path, image_size=224, 
                 use_age_stratify=False, age_bin_width=10):
    # 1. 加载数据
    df = pd.read_excel(excel_path)
    
    # 2. 按受试者ID分组
    subject_groups = {}
    for idx, row in df.iterrows():
        subject_id = row['Scan']  # 受试者ID
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append({
            'image': row['image_path'],
            'age': row['Age']
        })
    
    # 3. 划分受试者ID
    subject_ids = list(subject_groups.keys())
    if use_age_stratify:
        # 年龄分层划分
        ages = [subject_groups[sid][0]['age'] for sid in subject_ids]
        age_bins = [age // age_bin_width for age in ages]
        train_ids, test_ids = train_test_split(
            subject_ids, test_size=0.2, random_state=42, stratify=age_bins
        )
    else:
        # 随机划分
        train_ids, test_ids = train_test_split(
            subject_ids, test_size=0.2, random_state=42
        )
    
    # 4. 根据ID获取图像
    train_data = []
    for sid in train_ids:
        train_data.extend(subject_groups[sid])
    
    test_data = []
    for sid in test_ids:
        test_data.extend(subject_groups[sid])
    
    # 5. 验证无重叠
    assert set(train_ids) & set(test_ids) == set(), "数据泄露检测失败！"
    
    return train_data, test_data
```

---

## 📝 总结

### 问题
- 同一受试者的图像分散在训练/测试集，导致数据泄露

### 解决方案
- 按受试者ID划分数据集，确保无泄露

### 影响
- 测试集MAE从5.67上升到6.67（真实性能）
- 需要更强的正则化和数据增强

### 最佳实践
- ✅ 使用 `train.py` 统一训练脚本
- ✅ MAE损失函数
- ✅ 不使用水平翻转
- ✅ 梯度裁剪 + 学习率预热
- ✅ 早停防止过拟合
- ✅ 可选年龄分层采样

### 当前最佳结果
- **模型**: ResNet50
- **测试集MAE**: 6.67年
- **训练运行**: run_20251226_182738_noturn
- **配置**: 224分辨率，无水平翻转，by-subject划分

---

## 🔄 更新历史

- **2025-12-25**: 发现并修复数据泄露问题
- **2025-12-26**: 验证修复效果，优化训练策略
- **2025-12-29**: 简化为统一的train.py脚本

---

## 📚 参考资料

- `dataset.py`: 数据集加载和划分逻辑
- `scripts/analyze_dataset.py`: 数据集分析脚本
- `analysis_results/similarity_analysis.csv`: 图像相似度分析结果
- `analysis_results/subject_summary.csv`: 受试者统计摘要
