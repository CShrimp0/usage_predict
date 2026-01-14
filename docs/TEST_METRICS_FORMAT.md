# test_metrics.json 格式说明

## 概述
优化后的 `test_metrics.json` 提供了更丰富的评估信息，包括评估元数据、模型配置、数据集配置和详细的性能指标。

## JSON 结构

```json
{
  "evaluation_info": {
    "checkpoint_path": "模型检查点路径",
    "evaluation_time": "评估时间 (YYYY-MM-DD HH:MM:SS)",
    "device": "计算设备 (cuda/cpu)"
  },
  
  "model_config": {
    "architecture": "模型架构 (如 resnet50)",
    "dropout": "Dropout率",
    "best_epoch": "最佳训练轮次",
    "val_mae": "验证集MAE"
  },
  
  "dataset_config": {
    "test_size": "测试集比例",
    "val_size": "验证集比例",
    "seed": "随机种子",
    "use_age_stratify": "是否使用年龄分层抽样",
    "age_bin_width": "年龄分箱宽度",
    "age_range": "年龄范围 (如 0-100)",
    "total_samples": "测试样本总数"
  },
  
  "overall_metrics": {
    "MAE": {
      "value": "平均绝对误差值",
      "description": "平均绝对误差 (Mean Absolute Error)",
      "unit": "years"
    },
    "RMSE": {
      "value": "均方根误差值",
      "description": "均方根误差 (Root Mean Square Error)",
      "unit": "years"
    },
    "Correlation": {
      "value": "相关系数值",
      "description": "皮尔逊相关系数 (Pearson Correlation Coefficient)",
      "range": "[-1, 1]"
    },
    "Accuracy_5years": {
      "value": "准确率百分比",
      "description": "5年误差内准确率",
      "unit": "%"
    },
    "Accuracy_10years": {
      "value": "准确率百分比",
      "description": "10年误差内准确率",
      "unit": "%"
    },
    "Accuracy_15years": {
      "value": "准确率百分比",
      "description": "15年误差内准确率",
      "unit": "%"
    }
  },
  
  "age_group_analysis": [
    {
      "age_range": "年龄段 (如 10-20)",
      "start_age": "起始年龄",
      "end_age": "结束年龄",
      "count": "样本数量",
      "mae": "该年龄段的MAE",
      "rmse": "该年龄段的RMSE",
      "mean_true_age": "真实年龄均值",
      "mean_pred_age": "预测年龄均值",
      "true_mean": "真实年龄均值 (同上)",
      "pred_mean": "预测年龄均值 (同上)"
    }
  ],
  
  "error_analysis": {
    "description": "误差分析结果 (Error Analysis)",
    "outlier_threshold": {
      "value": "异常值阈值",
      "description": "异常值阈值 (mean + 2*std)",
      "unit": "years"
    },
    "outlier_count": {
      "value": "异常样本数量",
      "description": "异常样本数量",
      "percentage": "异常样本百分比"
    },
    "statistics": {
      "mean_mae": "误差均值",
      "std_mae": "误差标准差",
      "description": "误差统计 (均值 ± 标准差)"
    },
    "high_error_samples_file": "高误差样本文件路径",
    "low_error_samples_file": "低误差样本文件路径"
  }
}
```

## 关键改进

### 1. 评估元数据 (evaluation_info)
- **checkpoint_path**: 追踪使用的模型检查点
- **evaluation_time**: 记录评估时间，便于版本管理
- **device**: 记录计算设备

### 2. 模型配置 (model_config)
- **architecture**: 模型架构信息
- **dropout**: Dropout配置
- **best_epoch**: 最佳训练轮次
- **val_mae**: 验证集性能，便于与测试集对比

### 3. 数据集配置 (dataset_config)
- **完整的数据划分参数**: 确保实验可复现
- **age_range**: 明确测试的年龄范围
- **total_samples**: 测试样本总数

### 4. 结构化指标 (overall_metrics)
每个指标包含：
- **value**: 数值
- **description**: 中文说明
- **unit/range**: 单位或取值范围

这种结构便于：
- 程序化读取特定指标
- 人工阅读理解
- 生成报告和可视化

### 5. 详细的误差分析 (error_analysis)
- **outlier统计**: 包含数量和百分比
- **误差分布**: 均值和标准差
- **文件引用**: 指向详细的误差样本文件

## 使用示例

### Python 读取
```python
import json

# 读取metrics
with open('test_metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

# 访问总体MAE
mae = metrics['overall_metrics']['MAE']['value']
print(f"MAE: {mae} {metrics['overall_metrics']['MAE']['unit']}")

# 访问年龄段分析
for group in metrics['age_group_analysis']:
    print(f"{group['age_range']}: MAE={group['mae']:.2f}, n={group['count']}")

# 访问异常值信息
outlier_pct = metrics['error_analysis']['outlier_count']['percentage']
print(f"异常样本占比: {outlier_pct}%")
```

### 对比不同模型
```python
import json
import pandas as pd

# 加载多个评估结果
results = []
for path in ['run1/test_metrics.json', 'run2/test_metrics.json']:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        results.append({
            'run': data['evaluation_info']['checkpoint_path'],
            'MAE': data['overall_metrics']['MAE']['value'],
            'RMSE': data['overall_metrics']['RMSE']['value'],
            'Correlation': data['overall_metrics']['Correlation']['value'],
            'outliers': data['error_analysis']['outlier_count']['percentage']
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

## 相关文件
- **test_metrics.json**: 总体评估指标（本文档）
- **predictions.json**: 详细的预测结果（每个样本的真实值、预测值、文件名）
- **high_error_samples.txt**: 高误差样本列表（异常值用⚠️标记）
- **low_error_samples.txt**: 低误差样本列表
- **image_feature_analysis.txt**: 高/低误差样本的图像特征对比分析
- **error_analysis_report.html**: 可视化误差分析报告

## 版本历史
- **v2.0** (2026-01-14): 添加评估元数据、模型配置、数据集配置，结构化指标说明
- **v1.0**: 初始版本，仅包含基本指标
