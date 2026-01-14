# 🎯 Grad-CAM 热力图功能使用指南

## 功能已成功集成到 evaluate.py ✅

### 📦 新增内容

#### 1. 核心代码 (evaluate.py)
- ✅ **GradCAM 类**: 实现梯度加权类激活映射算法
- ✅ **辅助函数**: 
  - `get_heatmap_overlay()` - 热力图叠加
  - `get_heatmap_only()` - 纯热力图生成
  - `get_mask_overlay()` - Mask标注叠加
  - `generate_gradcam_visualization()` - 完整可视化生成

#### 2. 自动化集成
- ✅ 评估完成后自动生成最佳/最差样本的热力图
- ✅ 无需额外参数，开箱即用
- ✅ 自动查找Mask文件（支持多种格式）

#### 3. 文档
- ✅ `GRADCAM_README.md` - 完整技术文档
- ✅ `GRADCAM_SUMMARY.md` - 快速概览
- ✅ 本文档 - 使用指南

#### 4. 测试工具
- ✅ `test_gradcam_eval.sh` - 一键测试脚本
- ✅ `verify_gradcam.sh` - 功能验证脚本

---

## 🚀 使用方法

### 方法 1: 直接运行评估（推荐）

```bash
cd /home/szdx/LNX/usage_predict

python evaluate.py \
    --checkpoint outputs/ablation/01_baseline/run_20260108_115437/best_model.pth \
    --image-dir /home/szdx/LNX/data/TA/Healthy/Images \
    --excel-path /home/szdx/LNX/data/TA/characteristics.xlsx
```

### 方法 2: 使用测试脚本

```bash
cd /home/szdx/LNX/usage_predict
./test_gradcam_eval.sh
```

---

## 📊 输出结果

### 生成的文件

```
evaluation_results/01_baseline_run_20260108_115437/
├── test_metrics.json                  # 评估指标
├── predictions.json                   # 详细预测
├── evaluation_results.png             # 散点图
├── bland_altman.png                  # BA图
├── age_group_comparison.png          # 年龄段对比
├── high_error_samples.txt            # 高误差样本
├── low_error_samples.txt             # 低误差样本
├── 🌟 gradcam_best_sample.png         # 最佳样本热力图
└── 🌟 gradcam_worst_sample.png        # 最差样本热力图
```

### 可视化内容

#### gradcam_best_sample.png (绿色标题)
```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  原图 + Mask标注     │  原图 + Grad-CAM    │   纯Grad-CAM热力图   │
│  (红色区域)          │  (彩色热力图)        │  (JET色彩映射)       │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ 真实年龄: XX.X 岁    │ 预测年龄: XX.X 岁   │ MAE: X.XX 岁        │
│                     │ 文件名              │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

#### gradcam_worst_sample.png (红色标题)
- 同样的布局，但用红色标题突出显示

---

## 💡 使用场景

### 1. 模型解释性分析
- 查看模型是否关注了正确的解剖区域
- 对比最佳/最差样本的注意力差异
- 验证模型学到的特征是否符合临床逻辑

### 2. 论文插图
- 高分辨率输出 (DPI=300)
- 专业的三子图布局
- 清晰的中文标注

### 3. 模型调试
- 识别模型失败的原因
- 发现数据标注问题
- 指导数据增强策略

### 4. 临床验证
- 展示给临床专家验证
- 确认AI关注区域与临床经验一致
- 提高模型可信度

---

## 🔧 技术细节

### Grad-CAM 工作流程

```
输入图像 → ResNet50 → layer4特征图
                      ↓
                 前向传播钩子
                      ↓
                  预测年龄
                      ↓
                 反向传播
                      ↓
                 梯度钩子
                      ↓
         梯度加权 × 特征图
                      ↓
            全局平均池化
                      ↓
         ReLU激活 + 归一化
                      ↓
          热力图 (0-1范围)
                      ↓
       JET色彩映射 (蓝→红)
                      ↓
         叠加到原图 (50%透明度)
```

### 颜色方案

- **Mask标注**: 红色 (RGB: 255, 0, 0), alpha=0.3
- **热力图**: JET色彩映射
  - 蓝色 → 低激活（不重要）
  - 绿色 → 中等激活
  - 黄色 → 高激活
  - 红色 → 最高激活（最重要）

### 目标层选择

- **默认**: ResNet50 的 `layer4`（最后卷积层）
- **原因**: 包含最高级语义信息，最接近决策层
- **自动后备**: 如果找不到layer4，使用最后一个卷积层

---

## ⚠️ 注意事项

### 1. 依赖库
确保安装了 opencv-python:
```bash
pip install opencv-python
```

### 2. Mask 文件位置
脚本自动查找 `{image_dir}/../Masks/` 目录
如果Mask在其他位置，请手动移动或创建符号链接

### 3. GPU 内存
生成热力图需要梯度计算，占用约 2-3GB GPU 内存

### 4. 输出文件名
- `gradcam_best_sample.png` - 固定名称，每次覆盖
- `gradcam_worst_sample.png` - 固定名称，每次覆盖

---

## 🐛 故障排除

### 问题: ImportError: No module named 'cv2'
**解决**: 
```bash
pip install opencv-python
```

### 问题: 未生成 Grad-CAM 图像
**检查**:
1. 是否有测试样本文件名 (filenames list)
2. 图像文件是否存在
3. 查看控制台警告信息

### 问题: Mask 未显示
**解决**:
1. 检查 Mask 目录: `ls /home/szdx/LNX/data/TA/Healthy/Masks/`
2. 确认文件名匹配
3. 查看控制台日志确认查找路径

### 问题: 热力图全黑
**原因**: 目标层梯度为0
**解决**: 检查模型是否正确加载权重

---

## 📈 后续优化建议

### 短期 (1-2天)
- [ ] 添加命令行参数控制是否生成热力图 (`--no-gradcam`)
- [ ] 支持指定要可视化的样本索引
- [ ] 添加进度条显示

### 中期 (1周)
- [ ] 批量生成 top-N 样本的热力图
- [ ] 生成 HTML 报告集成所有可视化
- [ ] 支持其他模型架构（EfficientNet, Vision Transformer）

### 长期 (1月)
- [ ] 实现其他可解释性方法 (Integrated Gradients, SHAP)
- [ ] 交互式Web界面浏览热力图
- [ ] 自动生成分析报告PDF

---

## 📚 参考资料

1. **原始论文**: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
2. **PyTorch实现**: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
3. **医学影像可解释AI**: [Interpretable AI in Healthcare](https://www.nature.com/articles/s41591-018-0300-7)

---

## ✅ 验证清单

运行以下命令验证安装:
```bash
cd /home/szdx/LNX/usage_predict
./verify_gradcam.sh
```

应该看到所有 ✅ 标记。

---

## 📞 技术支持

如有问题，请查看：
1. `GRADCAM_README.md` - 详细技术文档
2. 控制台日志输出
3. 生成的警告信息

---

**🎉 恭喜！Grad-CAM 热力图功能已成功集成到 evaluate.py！**

直接运行评估脚本即可自动生成可视化结果。
