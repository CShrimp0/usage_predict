#!/bin/bash
# 测试evaluate.py中的Grad-CAM功能

echo "=========================================="
echo "测试 evaluate.py 的 Grad-CAM 可视化功能"
echo "=========================================="

cd /home/szdx/LNX/usage_predict

python evaluate.py \
    --checkpoint outputs/ablation/01_baseline/run_20260108_115437/best_model.pth \
    --image-dir /home/szdx/LNX/data/TA/Healthy/Images \
    --excel-path /home/szdx/LNX/data/TA/characteristics.xlsx \
    --output-dir evaluation_results/01_baseline_run_20260108_115437 \
    --batch-size 32 \
    --num-workers 4

echo ""
echo "=========================================="
echo "检查生成的Grad-CAM可视化图像："
echo "=========================================="
ls -lh evaluation_results/01_baseline_run_20260108_115437/gradcam_*.png

echo ""
echo "完成！请查看以下文件："
echo "  - gradcam_best_sample.png  (最佳预测样本)"
echo "  - gradcam_worst_sample.png (最差预测样本)"
