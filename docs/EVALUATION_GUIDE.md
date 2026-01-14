# 评估脚本使用指南

## 快速开始

### 1. 评估单个模型
```bash
# 基本用法（自动从checkpoint读取训练配置）
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth

# 输出位置：evaluation_results/run_xxx/
```

### 2. 限制年龄范围评估
```bash
# 只评估18-88岁的受试者
python evaluate.py \
  --checkpoint outputs/run_xxx/best_model.pth \
  --min-age 18 \
  --max-age 88

# 只评估老年人（60岁以上）
python evaluate.py \
  --checkpoint outputs/run_xxx/best_model.pth \
  --min-age 60
```

**注意**: 
- 如果checkpoint中保存了年龄范围（训练时使用了`--min-age/--max-age`），评估会自动使用相同范围
- 可以通过命令行参数覆盖checkpoint中的年龄范围

### 3. 对比多个模型
```bash
# 对比所有评估结果
python compare_evaluations.py evaluation_results/*/test_metrics.json

# 对比特定日期的runs
python compare_evaluations.py evaluation_results/run_202601*/test_metrics.json

# 保存对比结果到CSV
python compare_evaluations.py evaluation_results/*/test_metrics.json --output comparison.csv
```

### 4. 生成HTML错误分析报告
```bash
# 生成交互式HTML报告
python analyze_error_samples.py \
  --result-dir evaluation_results/run_xxx \
  --image-dir data/TA \
  --max-samples 30

# 在VS Code中查看
# 右键 error_analysis_report.html → "Open with Live Server"
# 或直接在浏览器打开
```

## 评估输出文件详解

评估完成后，`evaluation_results/run_xxx/` 目录包含：

### 1. test_metrics.json（结构化指标）
包含以下部分：
- **evaluation_info**: 评估元数据（checkpoint路径、时间、设备）
- **model_config**: 模型配置（架构、dropout、最佳epoch、验证集MAE）
- **dataset_config**: 数据集配置（划分比例、种子、年龄范围、样本数）
- **overall_metrics**: 总体指标（MAE、RMSE、相关系数、准确率）
- **age_group_analysis**: 年龄段分析（每个年龄段的MAE、RMSE、样本数等）
- **error_analysis**: 误差分析（异常值统计、误差分布）

详见: [docs/TEST_METRICS_FORMAT.md](TEST_METRICS_FORMAT.md)

### 2. predictions.json（详细预测）
每个样本的详细信息：
```json
{
  "predictions": [预测年龄1, 预测年龄2, ...],
  "targets": [真实年龄1, 真实年龄2, ...],
  "filenames": ["图像1.jpg", "图像2.jpg", ...]
}
```

### 3. 误差样本列表
- **high_error_samples.txt**: 高误差样本（默认前50个，异常值用⚠️标记）
- **low_error_samples.txt**: 低误差样本（默认前50个）

格式：
```
filename	true_age	pred_age	error	abs_error	outlier_flag
image1.jpg	25.5	45.2	19.7	19.7	⚠️
image2.jpg	70.3	55.1	-15.2	15.2	
```

### 4. image_feature_analysis.txt（图像特征分析）
对比高误差和低误差样本的图像特征差异：
- 平均亮度、标准差、清晰度、对比度
- 偏度、熵等统计特征
- 显著差异的特征（>10%）及优化建议

### 5. 可视化图表
- **evaluation_results.png**: 4合1图（散点图、残差图、误差分布、年龄分布）
- **bland_altman.png**: Bland-Altman图（偏差分析）
- **age_group_comparison.png**: 年龄段对比（真实vs预测均值）

### 6. error_analysis_report.html（交互式报告）
由 `analyze_error_samples.py` 生成，包含：
- 高误差样本卡片（缩略图、误差、图像统计）
- 低误差样本卡片
- 异常样本专区
- 支持按误差大小排序
- 点击图像放大查看

## 读取评估结果示例

### Python 脚本
```python
import json

# 读取metrics
with open('evaluation_results/run_xxx/test_metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

# 访问总体指标
mae = metrics['overall_metrics']['MAE']['value']
rmse = metrics['overall_metrics']['RMSE']['value']
corr = metrics['overall_metrics']['Correlation']['value']

print(f"MAE: {mae:.2f} years")
print(f"RMSE: {rmse:.2f} years")
print(f"Correlation: {corr:.4f}")

# 访问年龄段分析
for group in metrics['age_group_analysis']:
    age_range = group['age_range']
    mae = group['mae']
    count = group['count']
    print(f"{age_range}: MAE={mae:.2f}, n={count}")

# 访问异常值信息
outlier_pct = metrics['error_analysis']['outlier_count']['percentage']
print(f"异常样本占比: {outlier_pct:.2f}%")
```

### Pandas 批量分析
```python
import json
import pandas as pd
from pathlib import Path

# 收集所有评估结果
results = []
for json_path in Path('evaluation_results').glob('*/test_metrics.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 提取关键信息
    results.append({
        'run': json_path.parent.name,
        'model': data['model_config']['architecture'],
        'age_range': data['dataset_config']['age_range'],
        'test_samples': data['dataset_config']['total_samples'],
        'MAE': data['overall_metrics']['MAE']['value'],
        'RMSE': data['overall_metrics']['RMSE']['value'],
        'Correlation': data['overall_metrics']['Correlation']['value']
    })

# 转为DataFrame并排序
df = pd.DataFrame(results)
df = df.sort_values('MAE')
print(df.to_string(index=False))
```

## 常见问题

### Q1: 评估时年龄范围与训练不一致？
**A**: 评估脚本会自动从checkpoint读取训练时的年龄范围。如果需要评估不同年龄范围（例如只评估老年人），可以使用`--min-age`和`--max-age`参数显式指定。

### Q2: 如何复现某次评估？
**A**: test_metrics.json 中记录了完整的评估配置（checkpoint路径、年龄范围、数据划分参数等），可以根据这些信息重新评估。

### Q3: 异常样本（⚠️标记）是什么？
**A**: 异常样本是误差超过 `mean + 2*std` 的样本。这些样本可能包含标注错误、图像质量问题或特殊情况，需要人工审查。

### Q4: 不同模型的年龄段分析结果不能直接对比？
**A**: 是的。如果训练时使用了不同的年龄范围（如一个是0-100岁，另一个是18-88岁），测试集的年龄分布会不同，年龄段统计也不可比。建议：
1. 使用相同年龄范围训练
2. 或者对所有模型使用相同的年龄范围进行评估（用`--min-age/--max-age`统一）

### Q5: HTML报告中的图片不显示？
**A**: 
1. 确保使用 VS Code 的 Live Server 插件打开
2. 或者使用浏览器直接打开（不要通过file://协议）
3. 检查图像路径是否正确（`--image-dir`参数）

## 推荐工作流

### 单模型评估
```bash
# 1. 评估模型
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth

# 2. 查看数值结果
cat evaluation_results/run_xxx/test_metrics.json

# 3. 查看可视化图表
open evaluation_results/run_xxx/*.png

# 4. 生成HTML报告（可选）
python analyze_error_samples.py \
  --result-dir evaluation_results/run_xxx \
  --image-dir data/TA
```

### 多模型对比
```bash
# 1. 评估所有模型
for checkpoint in outputs/*/best_model.pth; do
  python evaluate.py --checkpoint "$checkpoint"
done

# 2. 对比所有结果
python compare_evaluations.py evaluation_results/*/test_metrics.json

# 3. 保存对比表格
python compare_evaluations.py \
  evaluation_results/*/test_metrics.json \
  --output model_comparison.csv

# 4. 在Excel中查看
open model_comparison.csv
```

### 年龄范围实验
```bash
# 评估不同年龄范围
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth --min-age 0 --max-age 100
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth --min-age 18 --max-age 88
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth --min-age 60 --max-age 100

# 对比结果
python compare_evaluations.py evaluation_results/run_xxx/test_metrics.json
```

## 相关文档
- [docs/TEST_METRICS_FORMAT.md](TEST_METRICS_FORMAT.md) - test_metrics.json 格式详解
- [docs/ERROR_VISUALIZATION_GUIDE.md](ERROR_VISUALIZATION_GUIDE.md) - HTML报告使用指南
- [docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 训练参数说明
