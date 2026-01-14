#!/bin/bash
# 快速验证 Grad-CAM 功能集成

echo "🔍 验证 Grad-CAM 功能集成"
echo "======================================"

cd /home/szdx/LNX/usage_predict

# 1. 检查必要的导入
echo ""
echo "1️⃣  检查导入语句..."
if grep -q "import cv2" evaluate.py && \
   grep -q "from PIL import Image" evaluate.py && \
   grep -q "from torchvision import transforms" evaluate.py; then
    echo "   ✅ 所有必要的导入已添加"
else
    echo "   ❌ 缺少部分导入"
    exit 1
fi

# 2. 检查 GradCAM 类
echo ""
echo "2️⃣  检查 GradCAM 类..."
if grep -q "class GradCAM:" evaluate.py; then
    echo "   ✅ GradCAM 类已定义"
else
    echo "   ❌ 未找到 GradCAM 类"
    exit 1
fi

# 3. 检查辅助函数
echo ""
echo "3️⃣  检查辅助函数..."
functions=(
    "get_heatmap_overlay"
    "get_heatmap_only"
    "get_mask_overlay"
    "generate_gradcam_visualization"
)

for func in "${functions[@]}"; do
    if grep -q "def $func" evaluate.py; then
        echo "   ✅ $func 已定义"
    else
        echo "   ❌ 未找到 $func"
        exit 1
    fi
done

# 4. 检查主函数调用
echo ""
echo "4️⃣  检查主函数中的调用..."
if grep -q "gradcam_best_sample.png" evaluate.py && \
   grep -q "gradcam_worst_sample.png" evaluate.py; then
    echo "   ✅ 主函数已集成 Grad-CAM 调用"
else
    echo "   ❌ 主函数未调用 Grad-CAM"
    exit 1
fi

# 5. 语法检查
echo ""
echo "5️⃣  Python 语法检查..."
if python -m py_compile evaluate.py 2>/dev/null; then
    echo "   ✅ 语法检查通过"
else
    echo "   ❌ 语法错误"
    exit 1
fi

# 6. 检查文档
echo ""
echo "6️⃣  检查文档..."
if [ -f "GRADCAM_README.md" ] && [ -f "GRADCAM_SUMMARY.md" ]; then
    echo "   ✅ 文档已创建"
else
    echo "   ⚠️  部分文档缺失"
fi

# 7. 检查测试脚本
echo ""
echo "7️⃣  检查测试脚本..."
if [ -x "test_gradcam_eval.sh" ]; then
    echo "   ✅ 测试脚本已就绪"
else
    echo "   ⚠️  测试脚本不可执行"
fi

echo ""
echo "======================================"
echo "✅ 所有核心功能验证通过！"
echo ""
echo "📝 功能摘要："
echo "   • 自动生成最佳/最差样本的 Grad-CAM 可视化"
echo "   • 每张图包含 3 个子图：Mask标注 + 热力图叠加 + 纯热力图"
echo "   • 标注真实年龄、预测年龄、MAE"
echo "   • 输出文件："
echo "     - gradcam_best_sample.png  (绿色标题)"
echo "     - gradcam_worst_sample.png (红色标题)"
echo ""
echo "🚀 运行测试："
echo "   ./test_gradcam_eval.sh"
echo ""
