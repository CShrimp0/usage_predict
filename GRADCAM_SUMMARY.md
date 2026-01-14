## ✨ 新增功能：evaluate.py 自动生成 Grad-CAM 热力图

### 📋 功能概述

在 `evaluate.py` 中新增了自动生成 Grad-CAM 热力图可视化的功能。评估完成后会自动保存两张图：

1. **gradcam_best_sample.png** - 最佳预测样本（MAE最小）
2. **gradcam_worst_sample.png** - 最差预测样本（MAE最大）

### 🎨 可视化内容

每张图包含 **3个子图**：

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   原图 + Mask标注    │  原图 + Grad-CAM    │   纯Grad-CAM热力图   │
│   (红色半透明)       │   (热力图叠加)       │                    │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ 标注：真实年龄        │ 标注：预测年龄       │ 标注：MAE          │
│      (绿色/红色)     │      文件名         │                    │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

### 🚀 快速开始

运行评估即可自动生成：

```bash
python evaluate.py \
    --checkpoint outputs/ablation/01_baseline/run_20260108_115437/best_model.pth \
    --image-dir /home/szdx/LNX/data/TA/Healthy/Images \
    --excel-path /home/szdx/LNX/data/TA/characteristics.xlsx
```

### 📂 输出示例

```
evaluation_results/01_baseline_run_20260108_115437/
├── gradcam_best_sample.png    ✨ 最佳样本（绿色标题）
└── gradcam_worst_sample.png   ✨ 最差样本（红色标题）
```

### 🔧 技术实现

- **算法**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **目标层**: ResNet50 的 layer4（最后卷积层）
- **热力图颜色**: JET 色彩映射（蓝→绿→黄→红）
- **Mask颜色**: 红色半透明 (alpha=0.3)

### 📖 详细文档

查看 [GRADCAM_README.md](GRADCAM_README.md) 了解完整技术细节。

### ✅ 已验证

- [x] 语法检查通过
- [x] 导入依赖正确
- [x] 函数逻辑完整
- [x] 错误处理完善

### 💡 使用建议

1. 对比最佳/最差样本的热力图，了解模型学到的特征
2. 检查热力图是否聚焦在 Mask 标注区域
3. 分析最差样本的热力图是否关注了错误区域
4. 用于论文插图和模型解释性分析
