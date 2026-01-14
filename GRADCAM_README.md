# Grad-CAM 热力图可视化功能

## 功能说明

在 `evaluate.py` 脚本中添加了自动生成 Grad-CAM 热力图可视化的功能。评估完成后，会自动生成两张图：

1. **gradcam_best_sample.png** - 预测效果最好的样本（MAE最小）
2. **gradcam_worst_sample.png** - 预测效果最差的样本（MAE最大）

每张图包含三个子图：
- **左图**: 原图叠加 Mask 标注（红色半透明）
- **中图**: 原图叠加 Grad-CAM 热力图（显示模型关注区域）
- **右图**: 纯 Grad-CAM 热力图

图中会标注：
- 真实年龄
- 预测年龄  
- MAE（平均绝对误差）
- 图像文件名

## 使用方法

### 1. 正常运行评估脚本

```bash
python evaluate.py \
    --checkpoint outputs/ablation/01_baseline/run_20260108_115437/best_model.pth \
    --image-dir /home/szdx/LNX/data/TA/Healthy/Images \
    --excel-path /home/szdx/LNX/data/TA/characteristics.xlsx \
    --output-dir evaluation_results/test_run
```

### 2. 或使用测试脚本

```bash
chmod +x test_gradcam_eval.sh
./test_gradcam_eval.sh
```

## 输出文件

在输出目录中会生成以下文件：

```
evaluation_results/01_baseline_run_20260108_115437/
├── test_metrics.json              # 评估指标
├── predictions.json                # 详细预测结果
├── evaluation_results.png          # 预测散点图和误差分布
├── bland_altman.png               # Bland-Altman图
├── age_group_comparison.png       # 年龄段对比图
├── high_error_samples.txt         # 高误差样本列表
├── low_error_samples.txt          # 低误差样本列表
├── gradcam_best_sample.png        # ✨ 最佳样本Grad-CAM可视化
└── gradcam_worst_sample.png       # ✨ 最差样本Grad-CAM可视化
```

## 技术细节

### Grad-CAM 原理

Grad-CAM（Gradient-weighted Class Activation Mapping）通过以下步骤生成热力图：

1. 对目标层（ResNet50的layer4）注册前向和反向钩子
2. 前向传播获取特征图
3. 反向传播获取梯度
4. 计算梯度加权的特征图
5. ReLU激活并归一化生成热力图

### 关键函数

- `GradCAM`: 核心类，实现Grad-CAM算法
- `generate_gradcam_visualization()`: 生成完整的三子图可视化
- `get_heatmap_overlay()`: 热力图叠加到原图
- `get_mask_overlay()`: Mask标注叠加到原图
- `get_heatmap_only()`: 生成纯热力图

### Mask 文件查找规则

脚本会自动在以下位置查找 Mask 文件：
1. `{image_dir}/../Masks/{filename}`
2. `{image_dir}/../Masks/{basename}.png`
3. `{image_dir}/../Masks/{basename}.jpg`

如果未找到 Mask 文件，第一个子图将显示原图。

## 可视化示例

### 最佳预测样本
- 标题颜色：绿色
- 显示模型在该样本上关注的正确区域
- MAE 接近 0

### 最差预测样本
- 标题颜色：红色
- 显示模型可能关注了错误区域或图像质量问题
- MAE 较大

## 注意事项

1. **依赖库**: 确保安装了 opencv-python (`cv2`)
   ```bash
   pip install opencv-python
   ```

2. **GPU 支持**: Grad-CAM 计算需要反向传播，建议使用GPU加速

3. **模型架构**: 当前实现针对 ResNet50 的 layer4 优化，其他架构可能需要调整目标层

4. **内存占用**: 生成热力图需要保留梯度信息，会占用一定显存

## 故障排除

### 问题：生成的热力图为空或全黑

**原因**: 未找到合适的卷积层

**解决**: 检查模型架构，确保模型包含 `layer4` 或其他卷积层

### 问题：Mask 未显示

**原因**: Mask 文件路径不正确或文件不存在

**解决**: 
1. 检查 Mask 目录是否存在: `{image_dir}/../Masks`
2. 检查 Mask 文件名是否与图像文件名匹配
3. 查看控制台日志确认 Mask 查找路径

### 问题：图像加载失败

**原因**: 图像文件路径不正确

**解决**: 确保 `--image-dir` 参数指向正确的图像目录

## 未来改进

- [ ] 支持批量生成多个样本的热力图
- [ ] 添加命令行参数控制是否生成 Grad-CAM
- [ ] 支持其他可视化方法（Integrated Gradients, SHAP等）
- [ ] 自动识别不同模型架构的最佳目标层
- [ ] 生成 GIF 动画展示不同层的热力图演变

## 参考资料

- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [PyTorch Grad-CAM 实现教程](https://github.com/jacobgil/pytorch-grad-cam)
