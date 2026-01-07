# 🧪 超参数优化实验计划

**创建日期**: 2026-01-05  
**优化目标**: Val MAE < 5.5岁 | Test MAE < 7.0岁  
**总实验数**: 31个 (9个批次)

---

## 📊 实验数据表

**详细配置和结果记录**: [`experiment_plan.csv`](experiment_plan.csv)  
**使用方式**: 用Excel/LibreOffice/Google Sheets打开，训练后填写结果列

---

## 🎯 新Baseline配置 (必须包含分层抽样)

### **Baseline - 采用分层抽样的标准配置**
```bash
Model: resnet50 | BS: 32 | Dropout: 0.5 | LR: 1e-4 | WD: 1e-4 | Loss: MAE
分层抽样: ✅ (必须) | CLAHE: ❌ (待测试) | ImageSize: 224
预期: Val MAE 6.3-6.5岁
```

### **旧Baseline对比 (已废弃)**
```bash
run_20251226_182738_noturn: MAE 6.668岁 (无分层抽样, 无CLAHE)
→ 现在所有实验都基于分层抽样！
```

### **通用固定参数**
```bash
--epochs 300 --patience 100 --seed 42 --use-age-stratify --age-bin-width 10
```

---

## 📋 实验批次概览

### **批次1: Baseline + 基础特性测试** (4个实验, 优先级★★★★★)
- **BASELINE**: 分层抽样 + 标准配置 (必跑)
- **1.1**: 添加CLAHE预处理
- **1.2**: 增大图像尺寸到256
- **1.3**: CLAHE + 256尺寸组合

**目的**: 建立新基线，测试CLAHE和图像尺寸影响

---

### **批次2: Dropout系统测试** (5个实验, 优先级★★★★)
测试Dropout: **0.2, 0.3, 0.4, 0.6, 0.7**

**目的**: 找到最优Dropout值，平衡过拟合和欠拟合

---

### **批次3: Weight Decay优化** (3个实验, 优先级★★★★)
测试WeightDecay: **1e-5, 5e-5, 1e-3**

**目的**: 结合最优Dropout，找到最佳权重衰减

---

### **批次4: 学习率调优** (3个实验, 优先级★★★)
测试LR: **5e-5, 2e-4, 3e-4**

**目的**: 优化收敛速度和最终精度

---

### **批次5: 损失函数对比** (3个实验, 优先级★★★)
测试Loss: **SmoothL1, Huber, MSE**

**目的**: 评估不同损失函数对异常值的鲁棒性

---

### **批次6: Batch Size影响** (3个实验, 优先级★★★)
测试BatchSize: **16, 64, 96**

**目的**: 平衡训练稳定性和GPU利用率

---

### **批次7: 模型架构对比** (6个实验, 优先级★★★★★)
测试模型: **EfficientNet-B0/B1/B2, ConvNeXt, MobileNet, RegNet**

**目的**: 找到最适合骨龄预测的架构

---

### **批次8: 最优配置组合** (3个实验, 优先级★★★★★)
基于前7批结果，组合最优参数进行精细调优

---

### **批次9: 集成学习** (2个实验, 优先级★★★★★)
- **9.1**: 3模型集成 (ResNet50 + EfficientNet-B1 + ConvNeXt)
- **9.2**: 6模型全集成

**目的**: 达到最终目标 Val MAE < 5.5岁

---

## 🚀 快速启动脚本

### **第1批: Baseline + 基础特性 (4个实验, 约2-3小时)**

```bash
# GPU 0: BASELINE - 分层抽样基线
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model resnet50 --batch-size 32 --dropout 0.5 \
    --lr 0.0001 --weight-decay 0.0001 --loss mae \
    --use-age-stratify --image-size 224 &

# GPU 1: 实验1.1 - 添加CLAHE
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model resnet50 --batch-size 32 --dropout 0.5 \
    --lr 0.0001 --weight-decay 0.0001 --loss mae \
    --use-age-stratify --use-clahe --image-size 224 &

# GPU 2: 实验1.2 - 增大图像尺寸
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model resnet50 --batch-size 32 --dropout 0.5 \
    --lr 0.0001 --weight-decay 0.0001 --loss mae \
    --use-age-stratify --image-size 256 &

# GPU 3: 实验1.3 - CLAHE + 256尺寸
CUDA_VISIBLE_DEVICES=3 python train.py \
    --model resnet50 --batch-size 32 --dropout 0.5 \
    --lr 0.0001 --weight-decay 0.0001 --loss mae \
    --use-age-stratify --use-clahe --image-size 256 &
```

---

### **第2批: Dropout系统测试 (5个实验)**

```bash
# GPU 0-4: Dropout 0.2, 0.3, 0.4, 0.6, 0.7
for i in 0 1 2 3 4; do
    DROPOUT=(0.2 0.3 0.4 0.6 0.7)
    CUDA_VISIBLE_DEVICES=$i python train.py \
        --model resnet50 --batch-size 32 --dropout ${DROPOUT[$i]} \
        --lr 0.0001 --weight-decay 0.0001 --loss mae \
        --use-age-stratify --use-clahe --image-size 224 &
done
```

---

### **第3批: Weight Decay优化 (3个实验)**

```bash
# 基于第2批最优Dropout (假设为0.3)
# GPU 0: WD=1e-5
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model resnet50 --batch-size 32 --dropout 0.3 \
    --lr 0.0001 --weight-decay 0.00001 --loss mae \
    --use-age-stratify --use-clahe --image-size 224 &

# GPU 1: WD=5e-5
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model resnet50 --batch-size 32 --dropout 0.3 \
    --lr 0.0001 --weight-decay 0.00005 --loss mae \
    --use-age-stratify --use-clahe --image-size 224 &

# GPU 2: WD=1e-3
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model resnet50 --batch-size 32 --dropout 0.3 \
    --lr 0.0001 --weight-decay 0.001 --loss mae \
    --use-age-stratify --use-clahe --image-size 224 &
```

---

### **第7批: 模型架构对比 (6个实验, 全GPU并行)**

```bash
# GPU 0: EfficientNet-B0
CUDA_VISIBLE_DEVICES=0 python train.py --model efficientnet_b0 --batch-size 96 --dropout 0.3 --lr 0.0001 --weight-decay 0.00001 --loss mae --use-age-stratify --use-clahe --image-size 224 &

# GPU 1: EfficientNet-B1
CUDA_VISIBLE_DEVICES=1 python train.py --model efficientnet_b1 --batch-size 96 --dropout 0.3 --lr 0.0001 --weight-decay 0.00001 --loss mae --use-age-stratify --use-clahe --image-size 240 &

# GPU 2: EfficientNet-B2
CUDA_VISIBLE_DEVICES=2 python train.py --model efficientnet_b2 --batch-size 64 --dropout 0.3 --lr 0.0001 --weight-decay 0.00001 --loss mae --use-age-stratify --use-clahe --image-size 260 &

# GPU 3: ConvNeXt-Tiny
CUDA_VISIBLE_DEVICES=3 python train.py --model convnext_tiny --batch-size 64 --dropout 0.3 --lr 0.0001 --weight-decay 0.00001 --loss mae --use-age-stratify --use-clahe --image-size 224 &

# GPU 4: MobileNetV3
CUDA_VISIBLE_DEVICES=4 python train.py --model mobilenet_v3 --batch-size 96 --dropout 0.3 --lr 0.0001 --weight-decay 0.00001 --loss mae --use-age-stratify --use-clahe --image-size 224 &

# GPU 5: RegNet
CUDA_VISIBLE_DEVICES=5 python train.py --model regnet --batch-size 64 --dropout 0.3 --lr 0.0001 --weight-decay 0.00001 --loss mae --use-age-stratify --use-clahe --image-size 224 &
```

---

## 📊 结果分析工具

### **自动提取结果到CSV**

```bash
python << 'EOF'
import json
import csv
from pathlib import Path

# 读取所有训练结果
results = []
for run_dir in sorted(Path('outputs').glob('run_*')):
    config_file = run_dir / 'config.json'
    history_file = run_dir / 'history.json'
    
    if not (config_file.exists() and history_file.exists()):
        continue
    
    with open(config_file) as f:
        config = json.load(f)
    with open(history_file) as f:
        history = json.load(f)
    
    # 提取关键信息
    val_mae = min(history['val_mae'])
    best_epoch = history['val_mae'].index(val_mae) + 1
    
    results.append({
        'RunName': run_dir.name,
        'ValMAE': f"{val_mae:.3f}",
        'BestEpoch': best_epoch,
        'Model': config.get('model', {}).get('architecture', 'N/A'),
        'Dropout': config.get('model', {}).get('dropout', 'N/A'),
        'UseStratify': config.get('dataset', {}).get('use_age_stratify', False),
        'UseCLAHE': config.get('preprocessing', {}).get('use_clahe', False),
    })

# 按MAE排序
results.sort(key=lambda x: float(x['ValMAE']))

# 打印排名
print("\n" + "="*80)
print("训练结果排名 (按Val MAE升序)")
print("="*80)
for i, r in enumerate(results, 1):
    stratify = "✓" if r['UseStratify'] else "✗"
    clahe = "✓" if r['UseCLAHE'] else "✗"
    print(f"{i:2d}. {r['RunName']:<35} | MAE: {r['ValMAE']:>6} | "
          f"Epoch: {r['BestEpoch']:>3} | Model: {r['Model']:<15} | "
          f"Drop: {r['Dropout']} | 分层:{stratify} | CLAHE:{clahe}")
print("="*80 + "\n")
EOF
```

---

### **快速查看CSV表格**

```bash
# 方法1: 用列对齐方式显示
column -t -s',' experiment_plan.csv | less -S

# 方法2: 用Python pandas美化输出
python -c "
import pandas as pd
df = pd.read_csv('experiment_plan.csv', encoding='utf-8')
print(df[['实验ID', 'Model', 'Dropout', 'LR', '分层抽样', 'CLAHE', '预期ValMAE', '状态']].to_string(index=False))
"

# 方法3: 用Excel/LibreOffice打开
libreoffice experiment_plan.csv  # Linux
# open experiment_plan.csv       # Mac
# start experiment_plan.csv      # Windows
```

---

## 📈 关键发现总结 (实验完成后填写)

### **Dropout影响曲线**
```
Dropout  Val MAE   结论
0.2      ___      (过拟合?)
0.3      ___      
0.4      ___      
0.5      ___      (Baseline)
0.6      ___      
0.7      ___      (欠拟合?)

最优值: ___ (在第___批实验中确定)
```

### **模型架构对比**
```
Model              Val MAE   参数量    推理速度
ResNet50           ___       25.6M     ___ms
EfficientNet-B0    ___       5.3M      ___ms
EfficientNet-B1    ___       7.8M      ___ms
ConvNeXt-Tiny      ___       28.6M     ___ms

最优架构: ___ (性能/速度平衡)
```

### **特性影响评估**
```
配置                     Val MAE    提升幅度
Baseline (分层抽样)      ___        -
+ CLAHE                  ___        ± ___
+ 256尺寸                ___        ± ___
+ CLAHE + 256            ___        ± ___
+ 最优Dropout            ___        ± ___
+ 最优WeightDecay        ___        ± ___

最终最优单模型: ___ (MAE: ___)
集成模型: ___ (MAE: ___)
```

---

## 🎯 阶段性目标

### **第1批完成后 (预计2-3小时)**
- [ ] 确认分层抽样 vs 无分层的MAE差异
- [ ] 确认CLAHE预处理的效果
- [ ] 找到最优的Dropout值 (0.3/0.5/0.6)
- [ ] 评估EfficientNet vs ResNet

### **第2批完成后**
- [ ] 锁定最优正则化组合
- [ ] 确认最佳学习率
- [ ] 测试损失函数影响

### **第3批 (集成学习)**
- [ ] 选择Top3模型进行集成
- [ ] 尝试6模型全集成
- [ ] 达成目标: Val MAE < 5.5岁

---

## 💡 实验技巧

### **并行运行监控**
```bash
# 查看所有训练进程
watch -n 5 'ps aux | grep train.py'

# 监控GPU使用
watch -n 1 nvidia-smi

# 查看最新日志
tail -f outputs/run_*/training.log
```

### **快速查看结果**
```bash
# 查看所有运行的最佳MAE
for dir in outputs/run_*/; do
    echo "$dir: $(grep 'best_val_mae' $dir/history.json | tail -1)"
done

# 按MAE排序
python -c "
import json
from pathlib import Path
results = []
for p in Path('outputs').glob('run_*/history.json'):
    with open(p) as f:
        h = json.load(f)
        results.append((p.parent.name, min(h['val_mae'])))
for name, mae in sorted(results, key=lambda x: x[1]):
    print(f'{name}: {mae:.3f}')
"
```

---

## 📝 实验日志

### 2026-01-05
- ✅ 完成实验计划制定
- ✅ 优化config.json保存逻辑（支持分层/CLAHE记录）
- ✅ 添加学习率曲线可视化
- ⬜ 待启动第1批实验

### 2026-01-XX (待填写)
- ⬜ 第1批实验完成
- ⬜ 分析结果，确定第2批配置
- ⬜ ...
