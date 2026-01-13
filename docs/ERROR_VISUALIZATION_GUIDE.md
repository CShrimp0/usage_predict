# 错误样本可视化分析工具使用指南

## 功能简介

`analyze_error_samples.py` 是一个用于生成交互式HTML错误样本分析报告的工具。它可以帮助你快速可视化检查模型的高错误、低错误和离群样本，无需手动复制图像文件。

## 主要特性

✅ **交互式HTML报告** - 在浏览器中打开，支持标签页切换  
✅ **图像统计特征** - 自动计算每张图像的灰度、清晰度、对比度等  
✅ **三类样本展示**:
- 高错误样本(Top-N MAE最大)
- 低错误样本(Top-N MAE最小)
- 离群样本(误差超过3倍标准差)

✅ **无需复制文件** - 直接使用图像路径，节省磁盘空间  
✅ **排序功能** - 支持按误差、真实年龄、预测年龄排序  
✅ **统计摘要** - 显示每组样本的平均图像统计特征

## 快速开始

### 1. 前置条件

确保已运行 `evaluate.py` 生成错误样本分析文件:

```bash
python evaluate.py --model-path outputs/run_xxx/best_model.pth
```

这会在 `evaluation_results/` 目录下生成:
- `high_error_samples.txt`
- `low_error_samples.txt`
- `outlier_samples.txt`

### 2. 生成HTML报告

```bash
python analyze_error_samples.py \
  --result-dir evaluation_results/01_baseline_run_20260108_115437 \
  --image-dir ../data/TA \
  --max-samples 30
```

**参数说明:**
- `--result-dir`: 评估结果目录（包含3个txt文件）
- `--image-dir`: 图像根目录（会递归搜索子目录）
- `--max-samples`: 每类显示的最大样本数（默认50）
- `--output`: 可选，HTML输出路径（默认保存到结果目录）

### 3. 查看报告

脚本会自动生成 `error_analysis_report.html`，用浏览器打开即可:

```bash
firefox evaluation_results/xxx/error_analysis_report.html
# 或
google-chrome evaluation_results/xxx/error_analysis_report.html
```

## HTML报告界面说明

### 标签页

1. **⚠️ 高错误样本** - 模型预测误差最大的样本
   - 可能包含数据质量问题
   - 可能是模型难以处理的边缘情况
   
2. **✅ 低错误样本** - 模型预测最准确的样本
   - 代表模型表现最佳的情况
   - 可用于分析成功案例

3. **🚨 离群样本** - 统计意义上的异常样本
   - 误差超过 `mean + 3*std`
   - 需要特别关注的异常情况

### 样本卡片信息

每张样本卡片显示:
- **图像缩略图** (200px高度)
- **文件名** - 完整的图像文件名
- **误差** - MAE（平均绝对误差），单位：岁
- **真实年龄** vs **预测年龄**
- **图像统计**:
  - 灰度: 平均灰度值 (0-255)
  - 清晰度: Laplacian方差
  - 对比度: 灰度标准差/均值

### 交互功能

- **排序下拉菜单**: 
  - 按误差排序（默认）
  - 按真实年龄排序
  - 按预测年龄排序

- **悬停效果**: 鼠标悬停在卡片上会有放大和阴影效果

## 使用场景

### 1. 模型调试

对比不同实验的高错误样本，找到共性问题:

```bash
# 实验1
python analyze_error_samples.py \
  --result-dir evaluation_results/01_baseline_run_xxx \
  --image-dir ../data/TA

# 实验2
python analyze_error_samples.py \
  --result-dir evaluation_results/02_dropout_run_xxx \
  --image-dir ../data/TA
```

### 2. 数据质量检查

查看离群样本，识别标注错误或图像质量问题:

```bash
python analyze_error_samples.py \
  --result-dir evaluation_results/01_baseline_run_xxx \
  --image-dir ../data/TA \
  --max-samples 100  # 显示更多离群样本
```

### 3. 论文图表准备

找到典型的成功/失败案例用于论文展示:

```bash
# 只看前10个最佳/最差案例
python analyze_error_samples.py \
  --result-dir evaluation_results/best_model \
  --image-dir ../data/TA \
  --max-samples 10
```

## 图像统计特征说明

| 特征 | 含义 | 正常范围 | 用途 |
|------|------|----------|------|
| **平均灰度** | 图像平均亮度 | 0-255 | 评估曝光是否合适 |
| **灰度标准差** | 像素值离散程度 | 通常10-80 | 评估对比度 |
| **清晰度** | Laplacian方差 | 越大越清晰 | 检测模糊图像 |
| **对比度** | std/mean | 0.1-0.5 | 评估图像对比度 |
| **偏度** | 灰度分布偏态 | -1到+1 | 识别过曝/欠曝 |

## 常见问题

### Q1: 图像无法显示怎么办？

确保 `--image-dir` 指向正确的目录，脚本会递归搜索子目录。如果还是找不到:

```bash
# 检查文件是否存在
find /home/szdx/LNX/data/TA -name "anon_1006_3.png"
```

### Q2: 统计特征显示"无法计算"？

可能是opencv无法读取图像，检查:
1. 图像文件是否损坏
2. 图像格式是否支持 (PNG/JPG/BMP等)

### Q3: HTML太大打开慢？

减少 `--max-samples` 数量:

```bash
python analyze_error_samples.py \
  --result-dir evaluation_results/xxx \
  --image-dir ../data/TA \
  --max-samples 20  # 减少到20
```

### Q4: 如何在无图形界面的服务器上使用？

生成HTML后，用 `scp` 或其他方式下载到本地查看:

```bash
scp user@server:/path/to/error_analysis_report.html ./
```

## 技术细节

- **图像搜索**: 使用 `os.walk()` 递归搜索，支持嵌套目录
- **清晰度计算**: Laplacian算子的方差，值越大越清晰
- **HTML生成**: 纯Python生成，无需额外模板引擎
- **图像显示**: 使用 `file://` 协议，不复制文件
- **响应式设计**: 使用CSS Grid自动调整布局

## 扩展建议

如果需要更高级的功能，可以考虑:

1. **添加过滤器**: 按年龄范围、误差范围过滤
2. **导出功能**: 导出选中样本到CSV
3. **批量对比**: 同时显示多个实验的相同样本
4. **热力图**: 使用GradCAM显示模型关注区域

## 相关脚本

- `evaluate.py` - 生成错误样本txt文件（前置步骤）
- `scripts/plot_age_error.py` - 绘制误差分布图
- `scripts/analyze_dataset.py` - 分析数据集统计信息
