# 最佳模型结果

## 模型信息

- **训练运行**: run_20251226_182738_noturn
- **验证集MAE**: 6.67 years
- **训练日期**: 2025-12-26

## 文件说明

- `config.json` - 完整训练配置（超参数、数据增强等）
- `history.json` - 训练历史（每个epoch的loss、MAE、RMSE）

## 权重文件

权重文件位于 `outputs/run_20251226_182738_noturn/best_model.pth`

**注意**: 权重文件不在Git中，需要单独备份。

## 可视化结果

相关图表已保存在 `results/figures/`:
- `best_training_curves.png` - 训练曲线
- `age_vs_error.png` - 年龄误差分析
- `true_vs_pred.png` - 预测散点图
- 其他分析图表

## 复现方法

使用相同配置训练：

```bash
python train_mae.py \
    --arch resnet50 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --loss mae
```

详细参数见 `config.json`
