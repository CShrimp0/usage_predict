# Grad-CAM 热力图可视化功能文档

## 📋 功能概述

`evaluate.py` 集成了自动 Grad-CAM 热力图可视化功能。评估完成后会自动生成最佳和最差预测样本的可视化图像，帮助理解模型关注的区域。

### 输出文件

1. **gradcam_best_sample.png** - 预测效果最好的样本（MAE最小）
2. **gradcam_worst_sample.png** - 预测效果最差的样本（MAE最大）

每张图包含 **3个子图**：

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   原图 + Mask标注    │  原图 + Grad-CAM    │   纯Grad-CAM热力图   │
│   (红色半透明)       │   (热力图叠加)       │                    │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ 显示 Mask 标注区域   │ 显示模型关注区域     │ 热力图细节          │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

标注信息：
- 真实年龄（绿色：MAE<5，红色：MAE≥5）
- 预测年龄
- MAE（平均绝对误差）
- 图像文件名

---

## 🚀 使用方法

### 方法 1: 直接运行评估（推荐）

```bash
cd /home/szdx/LNX/usage_predict

python evaluate.py \
    --checkpoint outputs/run_XXXXXXX/best_model.pth \
    --image-dir /path/to/images \
    --excel-path /path/to/characteristics.xlsx
```

评估完成后，Grad-CAM 图像会自动保存到评估结果目录中。

### 方法 2: 查看现有评估结果

```bash
ls evaluation_results/*/gradcam_*.png
```

---

## 📂 输出目录结构

```
evaluation_results/01_baseline_run_20260108_115437/
├── test_metrics.json              # 评估指标
├── predictions.json                # 详细预测结果
├── evaluation_results.png          # 预测散点图和误差分布
├── bland_altman.png               # Bland-Altman图
├── age_group_comparison.png       # 年龄段对比图
├── gradcam_best_sample.png        # ✨ 最佳样本热力图
├── gradcam_worst_sample.png       # ✨ 最差样本热力图
├── high_error_samples.txt         # 高误差样本列表
└── low_error_samples.txt          # 低误差样本列表
```

---

## 🔧 技术细节

### 算法实现

- **算法**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **目标层**: ResNet50 的 `layer4`（最后一个卷积层）
- **热力图颜色**: JET 色彩映射（蓝→绿→黄→红，红色表示高激活）
- **Mask 颜色**: 红色半透明 (alpha=0.3)

### 核心类和函数

#### 1. GradCAM 类 (`evaluate.py`)

```python
class GradCAM:
    """Grad-CAM可视化类"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        # ... 注册前向和反向钩子
    
    def generate_cam(self, input_tensor, target_output=None):
        """生成类激活映射"""
        # ... 计算梯度加权激活图
        return cam  # 返回热力图
```

#### 2. 辅助函数

```python
# 热力图叠加到原图
get_heatmap_overlay(image, heatmap, alpha=0.5)

# 生成纯热力图
get_heatmap_only(heatmap)

# Mask 标注叠加
get_mask_overlay(image, mask, alpha=0.3)

# 完整可视化生成
generate_gradcam_visualization(model, image_path, mask_path, 
                               true_age, pred_age, device)
```

### Mask 文件自动查找

系统会自动在以下位置查找 Mask 文件：

```python
# 假设图像路径: /path/to/Images/anon_123_1.png
# 自动查找: /path/to/Masks/anon_123_1.png
mask_dir = image_dir.replace('/Images/', '/Masks/')
```

支持的格式：`.png`, `.jpg`, `.jpeg`

---

## 💡 使用建议

### 1. 模型调试

对比最佳和最差样本：
- **最佳样本**: 检查模型是否正确关注标注区域
- **最差样本**: 分析模型是否关注了错误特征

### 2. 论文插图

- 选择有代表性的样本展示模型的可解释性
- 对比不同年龄段的热力图分布

### 3. 特征分析

- 观察模型在不同年龄组的关注区域差异
- 验证模型是否学到了医学上有意义的特征

### 4. 常见模式

**良好预测**:
- 热力图高度集中在 Mask 标注区域
- 激活强度分布均匀

**不良预测**:
- 热力图分散在多个区域
- 关注了背景或噪声区域

---

## 📖 参考文献

```bibtex
@inproceedings{selvaraju2017grad,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={618--626},
  year={2017}
}
```

---

## ❓ 常见问题

### Q: 如何生成特定样本的热力图？

A: 修改 `show_gradcam.ipynb` notebook 中的样本索引，或直接调用 `generate_gradcam_visualization()` 函数。

### Q: 热力图颜色如何解读？

A: 
- **红色/黄色**: 高激活区域，模型重点关注
- **绿色/蓝色**: 低激活区域，模型较少关注
- **黑色**: 基本无激活

### Q: Mask 文件找不到怎么办？

A: 确保 Mask 文件夹与 Images 文件夹为同级目录：
```
data/
  ├── Images/
  └── Masks/
```

### Q: 可以生成其他层的热力图吗？

A: 可以。修改 `GradCAM` 初始化时的 `target_layer` 参数：
```python
gradcam = GradCAM(model, target_layer=model.backbone.layer3)  # 使用 layer3
```

---

## 🔗 相关文档

- [README.md](README.md) - 项目总体说明
- [DATA_LEAKAGE_REPORT.md](DATA_LEAKAGE_REPORT.md) - 数据泄露检查报告
- `show_gradcam.ipynb` - 交互式热力图分析 notebook

---

**最后更新**: 2026-01-15
