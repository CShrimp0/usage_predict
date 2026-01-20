# 项目文件索引

## 📋 文件清单

### 🔧 核心训练与评估 (5个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `train.py` | 53K | 训练脚本，支持DDP、年龄分层、Top-3保存 |
| `evaluate.py` | 42K | 评估脚本，自动生成Grad-CAM热力图 |
| `dataset.py` | 26K | 数据加载，受试者级划分防数据泄露 |
| `model.py` | 12K | 模型定义（ResNet50/EfficientNet/ConvNeXt等） |
| `auxiliary_features.py` | 11K | 多模态辅助特征提取（性别/BMI/纹理特征） |

### 🛠️ 工具脚本 (4个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `analyze_error_samples.py` | 21K | 交互式错误分析，生成HTML报告 |
| `verify_no_leakage.py` | 12K | 数据泄露验证，重新执行划分逻辑 |
| `compare_evaluations.py` | 7.6K | 多模型评估结果对比 |
| `summarize_ablation_results.py` | 5.9K | 消融实验结果汇总 |

### 📚 文档 (5个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `README.md` | 8.4K | 项目说明、快速开始、使用指南 |
| `GRADCAM.md` | 6.0K | Grad-CAM热力图可视化功能文档 |
| `DATA_LEAKAGE_REPORT.md` | 5.7K | 数据泄露三重验证报告（run_20260113_164941） |
| `ABLATION_RESULTS.md` | 11K | 消融实验结果（baseline vs 多模态） |
| `MULTIMODAL_GUIDE.md` | 9.6K | 多模态训练完整指南 |

### 🔨 辅助脚本 (1个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `run_ablation_study.sh` | 5.0K | 批量运行消融实验 |

### 📊 Jupyter Notebooks (2个)
| 文件 | 说明 |
|------|------|
| `show_gradcam.ipynb` | 交互式Grad-CAM分析，包含5个cell：<br>1. 模型加载与可视化<br>2. 热力图一致性定量分析（7指标）<br>3. 三方法对比（Grad-CAM/++/SmoothGrad）<br>4. SmoothGrad批量可视化<br>5. 受试者级预测一致性分析 |
| `杂七杂八.ipynb` | 实验结果可视化（年龄分组MAE对比图） |

---

## 📂 目录结构

```
usage_predict/
├── *.py                          # 核心脚本和工具（15个文件，210K）
├── *.md                          # 文档（5个文件，41K）
├── *.sh                          # Shell脚本（1个文件）
├── *.ipynb                       # Jupyter notebooks（2个）
├── requirements.txt              # Python依赖
├── data/                         # 数据集（不在版本控制中）
├── outputs/                      # 训练输出（每次运行独立文件夹）
└── evaluation_results/           # 评估结果（预测文件、热力图等）
```

---

## 🎯 快速导航

### 我想...

- **开始训练**: 看 [README.md](README.md) → 快速开始
- **评估模型**: 看 [README.md](README.md) → 评估模型
- **查看热力图**: 看 [GRADCAM.md](GRADCAM.md)
- **检查数据泄露**: 运行 `python verify_no_leakage.py evaluation_results/run_XXX`
- **分析错误样本**: 运行 `python analyze_error_samples.py --result-dir evaluation_results/run_XXX`
- **对比多个模型**: 运行 `python compare_evaluations.py evaluation_results/*/test_metrics.json`
- **交互式分析**: 打开 `show_gradcam.ipynb` notebook
- **多模态训练**: 看 [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md)

---

## 🗑️ 已清理的冗余文件

以下文件已在 2026-01-15 清理：
- ~~`GRADCAM_README.md`~~ → 合并到 `GRADCAM.md`
- ~~`GRADCAM_GUIDE.md`~~ → 合并到 `GRADCAM.md`
- ~~`GRADCAM_SUMMARY.md`~~ → 合并到 `GRADCAM.md`
- ~~`check_data_leakage.py`~~ → 功能被 `verify_no_leakage.py` 覆盖
- ~~`check_data_leakage_deep.py`~~ → 功能被 `verify_no_leakage.py` 覆盖
- ~~`test_gradcam_eval.sh`~~ → 测试脚本，功能已验证
- ~~`verify_gradcam.sh`~~ → 测试脚本，功能已验证

---

**最后更新**: 2026-01-15
