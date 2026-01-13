"""
评估脚本
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

from dataset import load_dataset
from model import get_model


def evaluate(model, test_loader, device, image_paths=None):
    """评估模型
    
    Args:
        model: 待评估模型
        test_loader: 测试数据加载器
        device: 计算设备
        image_paths: 可选，图像路径列表，用于追踪每个样本
    
    Returns:
        metrics: 评估指标字典
        all_preds: 预测值数组
        all_targets: 真实值数组
        filenames: 文件名列表（如果提供了image_paths）
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算指标
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    # 计算相关系数
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    # 计算在N年误差范围内的准确率
    errors = np.abs(all_preds - all_targets)
    acc_5 = np.mean(errors <= 5) * 100
    acc_10 = np.mean(errors <= 10) * 100
    acc_15 = np.mean(errors <= 15) * 100
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'Correlation': float(correlation),
        'Accuracy_5years': float(acc_5),
        'Accuracy_10years': float(acc_10),
        'Accuracy_15years': float(acc_15),
        'predictions': [float(x) for x in all_preds.tolist()],
        'targets': [float(x) for x in all_targets.tolist()]
    }
    
    # 提取文件名
    filenames = None
    if image_paths is not None:
        filenames = [Path(p).name for p in image_paths]
    
    return metrics, all_preds, all_targets, filenames


def setup_chinese_font():
    """设置中文字体支持"""
    import matplotlib.font_manager as fm
    
    # 尝试多种中文字体
    chinese_fonts = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 
        'Noto Sans CJK SC',
        'Noto Sans CJK',
        'Source Han Sans CN',
        'SimHei',
        'Microsoft YaHei',
        'AR PL UMing CN',
        'DejaVu Sans'  # fallback
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f'使用字体: {font}')
            return True
    
    # 如果没有中文字体，使用英文
    print('警告: 未找到中文字体，使用英文标签')
    return False


def calculate_age_group_mae(predictions, targets, bin_width=10):
    """计算每个年龄段的MAE"""
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # 确定年龄范围
    min_age = int(np.floor(targets.min() / bin_width) * bin_width)
    max_age = int(np.ceil(targets.max() / bin_width) * bin_width)
    
    age_group_results = []
    
    for start_age in range(min_age, max_age, bin_width):
        end_age = start_age + bin_width
        mask = (targets >= start_age) & (targets < end_age)
        
        if mask.sum() > 0:
            group_preds = predictions[mask]
            group_targets = targets[mask]
            group_mae = np.mean(np.abs(group_preds - group_targets))
            group_rmse = np.sqrt(np.mean((group_preds - group_targets) ** 2))
            
            age_group_results.append({
                'age_range': f'{start_age}-{end_age}',
                'start_age': start_age,
                'end_age': end_age,
                'count': int(mask.sum()),
                'mae': float(group_mae),
                'rmse': float(group_rmse),
                'mean_true_age': float(group_targets.mean()),
                'mean_pred_age': float(group_preds.mean()),
                'true_mean': float(group_targets.mean()),  # 添加用于绘图
                'pred_mean': float(group_preds.mean())     # 添加用于绘图
            })
    
    return age_group_results


def save_error_analysis(predictions, targets, filenames, output_dir, top_n=50, mae_threshold=None):
    """保存误差分析结果
    
    保存误差最大和误差最小的样本列表到txt文件，便于后续分析。
    
    Args:
        predictions: 预测值数组
        targets: 真实值数组
        filenames: 文件名列表
        output_dir: 输出目录
        top_n: 保存前N个最大/最小误差的样本（默认50）
        mae_threshold: 可选，超过此阈值的样本视为异常大
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算每个样本的误差
    errors = np.abs(predictions.flatten() - targets.flatten())
    signed_errors = predictions.flatten() - targets.flatten()  # 有符号误差
    
    # 创建样本信息列表
    samples = []
    for i, (fname, pred, target, err, signed_err) in enumerate(zip(
            filenames, predictions.flatten(), targets.flatten(), errors, signed_errors)):
        samples.append({
            'filename': fname,
            'true_age': target,
            'pred_age': pred,
            'mae': err,
            'error': signed_err  # 正值=预测偏大, 负值=预测偏小
        })
    
    # 按MAE降序排列
    samples_sorted = sorted(samples, key=lambda x: x['mae'], reverse=True)
    
    # 计算异常阈值
    if mae_threshold is None:
        mae_threshold = errors.mean() + 2 * errors.std()
    outlier_count = sum(1 for s in samples if s['mae'] > mae_threshold)
    
    # 保存高误差样本（误差最大的前N个，异常样本用⚠️标记）
    high_error_path = output_dir / 'high_error_samples.txt'
    with open(high_error_path, 'w', encoding='utf-8') as f:
        f.write(f"# 高误差样本列表（按MAE降序排列）\n")
        f.write(f"# 总样本数: {len(samples)}, 显示前{min(top_n, len(samples))}个\n")
        f.write(f"# 平均MAE: {errors.mean():.2f}岁, 标准差: {errors.std():.2f}岁\n")
        f.write(f"# 异常阈值: {mae_threshold:.2f}岁 (均值+2×标准差), 异常样本数: {outlier_count} ({100*outlier_count/len(samples):.1f}%)\n")
        f.write(f"# ----------------------------------------------------------------\n")
        f.write(f"# 文件名 | 真实年龄 | 预测年龄 | MAE | 误差方向 | 异常标记\n")
        f.write(f"# ----------------------------------------------------------------\n\n")
        
        for sample in samples_sorted[:top_n]:
            direction = "偏大" if sample['error'] > 0 else "偏小"
            outlier_flag = "⚠️异常" if sample['mae'] > mae_threshold else ""
            f.write(f"{sample['filename']}\t"
                   f"{sample['true_age']:.1f}\t"
                   f"{sample['pred_age']:.1f}\t"
                   f"{sample['mae']:.2f}\t"
                   f"{direction}({sample['error']:+.2f})\t"
                   f"{outlier_flag}\n")
    
    print(f'高误差样本已保存: {high_error_path}')
    print(f'  - 包含 {outlier_count} 个异常样本（MAE > {mae_threshold:.2f}岁）')
    
    # 保存低误差样本（误差最小的前N个）
    low_error_path = output_dir / 'low_error_samples.txt'
    with open(low_error_path, 'w', encoding='utf-8') as f:
        f.write(f"# 低误差样本列表（按MAE升序排列）\n")
        f.write(f"# 总样本数: {len(samples)}, 显示前{min(top_n, len(samples))}个\n")
        f.write(f"# 平均MAE: {errors.mean():.2f}岁, 标准差: {errors.std():.2f}岁\n")
        f.write(f"# ----------------------------------------------------------------\n")
        f.write(f"# 文件名 | 真实年龄 | 预测年龄 | MAE | 误差方向\n")
        f.write(f"# ----------------------------------------------------------------\n\n")
        
        # 升序排列（误差最小的）
        samples_low = sorted(samples, key=lambda x: x['mae'])
        for sample in samples_low[:top_n]:
            direction = "偏大" if sample['error'] > 0 else "偏小"
            f.write(f"{sample['filename']}\t"
                   f"{sample['true_age']:.1f}\t"
                   f"{sample['pred_age']:.1f}\t"
                   f"{sample['mae']:.2f}\t"
                   f"{direction}({sample['error']:+.2f})\n")
    
    print(f'低误差样本已保存: {low_error_path}')
    
    # 计算异常阈值但不再单独生成outlier文件，而是在高误差文件中标记
    if mae_threshold is None:
        mae_threshold = errors.mean() + 2 * errors.std()  # 默认使用2个标准差
    
    outlier_count = sum(1 for s in samples if s['mae'] > mae_threshold)
    print(f'异常样本统计: 共{outlier_count}个 ({100*outlier_count/len(samples):.1f}%), 阈值={mae_threshold:.2f}岁')
    print(f'  (异常样本已在 high_error_samples.txt 中用⚠️标记)')
    
    # 返回样本数据供后续特征分析使用
    return {
        'high_error_samples': samples_sorted[:top_n],
        'low_error_samples': samples_low[:top_n],
        'all_samples': samples,
        'outlier_threshold': float(mae_threshold),
        'outlier_count': outlier_count,
        'mean_mae': float(errors.mean()),
        'std_mae': float(errors.std())
    }


def analyze_image_features(sample_info, image_dir, output_dir):
    """
    分析高错误和低错误样本的图像特征差异
    如果发现明显差异，可考虑使用直方图匹配等预处理方法
    
    Args:
        sample_info: save_error_analysis返回的样本信息字典
        image_dir: 图像文件目录
        output_dir: 输出目录
    """
    try:
        import cv2
    except ImportError:
        print('警告: 未安装opencv-python，跳过图像特征分析')
        return None
    
    output_dir = Path(output_dir)
    high_samples = sample_info['high_error_samples']
    low_samples = sample_info['low_error_samples']
    
    def compute_image_stats(filename):
        """计算单张图像的统计特征"""
        # 递归搜索图像
        img_path = None
        for root, dirs, files in os.walk(image_dir):
            if filename in files:
                img_path = os.path.join(root, filename)
                break
        
        if img_path is None or not os.path.exists(img_path):
            return None
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # 计算统计特征
            mean_intensity = float(np.mean(img))
            std_intensity = float(np.std(img))
            
            # 清晰度 (Laplacian方差)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            clarity = float(np.var(laplacian))
            
            # 对比度
            contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            # 偏度
            skewness = float(np.mean(((img - mean_intensity) / std_intensity) ** 3)) if std_intensity > 0 else 0
            
            # 直方图统计
            hist, _ = np.histogram(img, bins=256, range=(0, 256))
            hist = hist / hist.sum()  # 归一化
            entropy = -np.sum(hist * np.log(hist + 1e-10))  # 熵
            
            return {
                'mean': mean_intensity,
                'std': std_intensity,
                'clarity': clarity,
                'contrast': contrast,
                'skewness': skewness,
                'entropy': float(entropy)
            }
        except Exception as e:
            return None
    
    # 计算高错误样本的特征
    print('\n正在计算高错误样本的图像特征...')
    high_features = []
    for sample in high_samples:
        stats = compute_image_stats(sample['filename'])
        if stats:
            high_features.append(stats)
    
    # 计算低错误样本的特征
    print('正在计算低错误样本的图像特征...')
    low_features = []
    for sample in low_samples:
        stats = compute_image_stats(sample['filename'])
        if stats:
            low_features.append(stats)
    
    if len(high_features) == 0 or len(low_features) == 0:
        print('警告: 无法计算足够的图像特征')
        return None
    
    # 统计分析
    feature_names = ['mean', 'std', 'clarity', 'contrast', 'skewness', 'entropy']
    
    summary_path = output_dir / 'image_feature_analysis.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 图像特征分析：高错误样本 vs 低错误样本\n")
        f.write(f"# 分析样本数: 高错误={len(high_features)}, 低错误={len(low_features)}\n")
        f.write("# ================================================================\n\n")
        
        f.write("## 统计摘要\n\n")
        f.write(f"{'特征':<15} | {'高错误(均值±标准差)':<25} | {'低错误(均值±标准差)':<25} | 差异百分比\n")
        f.write("-" * 95 + "\n")
        
        comparison_results = {}
        for feat in feature_names:
            high_vals = [f[feat] for f in high_features]
            low_vals = [f[feat] for f in low_features]
            
            high_mean = np.mean(high_vals)
            high_std = np.std(high_vals)
            low_mean = np.mean(low_vals)
            low_std = np.std(low_vals)
            
            # 计算差异百分比
            diff_pct = abs(high_mean - low_mean) / low_mean * 100 if low_mean != 0 else 0
            
            comparison_results[feat] = {
                'high_mean': high_mean,
                'high_std': high_std,
                'low_mean': low_mean,
                'low_std': low_std,
                'diff_pct': diff_pct
            }
            
            f.write(f"{feat:<15} | {high_mean:>10.2f} ± {high_std:<10.2f} | "
                   f"{low_mean:>10.2f} ± {low_std:<10.2f} | {diff_pct:>8.1f}%\n")
        
        f.write("\n\n## 分析结论\n\n")
        
        # 识别显著差异的特征
        significant_features = [(k, v['diff_pct']) for k, v in comparison_results.items() 
                               if v['diff_pct'] > 10]  # 差异超过10%视为显著
        
        if significant_features:
            f.write("发现以下特征存在显著差异（>10%）:\n")
            for feat, diff in sorted(significant_features, key=lambda x: x[1], reverse=True):
                f.write(f"  - {feat}: {diff:.1f}%\n")
            
            f.write("\n💡 建议:\n")
            f.write("  1. 图像特征存在明显差异，可能影响模型性能\n")
            f.write("  2. 考虑使用以下预处理方法:\n")
            
            if any(feat in ['mean', 'std', 'contrast'] for feat, _ in significant_features):
                f.write("     - 直方图均衡化 (CLAHE)\n")
                f.write("     - 对比度归一化\n")
            
            if any(feat == 'clarity' for feat, _ in significant_features):
                f.write("     - 锐化滤波\n")
                f.write("     - 数据增强（模糊/去噪）\n")
            
            f.write("  3. 参考低错误样本的图像风格进行直方图匹配\n")
        else:
            f.write("未发现显著的图像特征差异（<10%）\n")
            f.write("图像质量可能不是主要影响因素，建议从模型架构或标注质量方面分析。\n")
    
    print(f'图像特征分析已保存: {summary_path}')
    
    # 返回比较结果供后续使用
    return {
        'high_features': high_features,
        'low_features': low_features,
        'comparison': comparison_results,
        'significant_features': significant_features if significant_features else []
    }


def plot_results(predictions, targets, output_dir, use_chinese=True):
    """绘制结果图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置样式
    sns.set_style('whitegrid')
    has_chinese = setup_chinese_font() if use_chinese else False
    
    # 1. 预测值 vs 真实值散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：散点图 + 回归线
    ax = axes[0]
    ax.scatter(targets, predictions, alpha=0.5, s=30)
    
    # 绘制完美预测线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 拟合线性回归
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    ax.plot(targets, p(targets), 'g-', lw=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # 标签文字
    if has_chinese:
        ax.set_xlabel('真实年龄 (岁)', fontsize=12)
        ax.set_ylabel('预测年龄 (岁)', fontsize=12)
        ax.set_title('预测年龄 vs 真实年龄', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('True Age (years)', fontsize=12)
        ax.set_ylabel('Predicted Age (years)', fontsize=12)
        ax.set_title('Predicted vs True Age', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右图：误差分布直方图
    ax = axes[1]
    errors = predictions - targets
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax.axvline(errors.mean(), color='g', linestyle='--', lw=2, 
               label=f'Mean Error: {errors.mean():.2f}')
    if has_chinese:
        ax.set_xlabel('预测误差 (岁)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('预测误差分布', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Prediction Error (years)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f'图表已保存: {output_dir / "evaluation_results.png"}')
    plt.close()
    
    # 2. Bland-Altman图
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_age = (predictions + targets) / 2
    diff_age = predictions - targets
    
    ax.scatter(mean_age, diff_age, alpha=0.5, s=30)
    ax.axhline(0, color='r', linestyle='-', lw=2, label='Mean Difference')
    ax.axhline(diff_age.mean(), color='g', linestyle='--', lw=2, 
               label=f'Bias: {diff_age.mean():.2f}')
    ax.axhline(diff_age.mean() + 1.96 * diff_age.std(), color='orange', 
               linestyle='--', lw=2, label=f'±1.96 SD')
    ax.axhline(diff_age.mean() - 1.96 * diff_age.std(), color='orange', 
               linestyle='--', lw=2)
    
    if has_chinese:
        ax.set_xlabel('平均年龄 (岁)', fontsize=12)
        ax.set_ylabel('差异 (预测 - 真实, 岁)', fontsize=12)
        ax.set_title('Bland-Altman图', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Mean Age (years)', fontsize=12)
        ax.set_ylabel('Difference (Predicted - True, years)', fontsize=12)
        ax.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bland_altman.png', dpi=300, bbox_inches='tight')
    print(f'图表已保存: {output_dir / "bland_altman.png"}')
    plt.close()
    
    # 3. 年龄分段：真实年龄 vs 预测年龄对比柱状图
    age_group_mae = calculate_age_group_mae(predictions, targets, bin_width=10)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(age_group_mae))
    labels = [g['age_range'] for g in age_group_mae]
    true_means = [g['true_mean'] for g in age_group_mae]
    pred_means = [g['pred_mean'] for g in age_group_mae]
    counts = [g['count'] for g in age_group_mae]
    
    # 分组柱状图：真实年龄 vs 预测年龄
    width = 0.35
    x_pos = np.arange(len(labels))
    
    bars1 = ax.bar(x_pos - width/2, true_means, width, 
                   label='真实年龄' if has_chinese else 'True Age',
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, pred_means, width,
                   label='预测年龄' if has_chinese else 'Predicted Age', 
                   color='coral', alpha=0.8, edgecolor='black')
    
    # 标注样本数量
    for i, (true_m, pred_m, count) in enumerate(zip(true_means, pred_means, counts)):
        max_height = max(true_m, pred_m)
        ax.text(i, max_height + 1.5, f'n={count}',
                ha='center', va='bottom', fontsize=9, color='gray')
    
    # 添加参考线（完美预测线）
    ax.plot([-0.5, len(labels)-0.5], [-0.5, len(labels)-0.5],
            'g--', alpha=0.3, linewidth=1.5, label='完美预测' if has_chinese else 'Perfect Prediction')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    if has_chinese:
        ax.set_xlabel('年龄段 (岁)', fontsize=12)
        ax.set_ylabel('平均年龄 (岁)', fontsize=12)
        ax.set_title('各年龄段真实年龄 vs 预测年龄对比', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Age Group (years)', fontsize=12)
        ax.set_ylabel('Mean Age (years)', fontsize=12)
        ax.set_title('True vs Predicted Age by Age Group', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'age_group_comparison.png', dpi=300, bbox_inches='tight')
    print(f'图表已保存: {output_dir / "age_group_comparison.png"}')
    plt.close()
    
    return age_group_mae


def get_run_name_from_checkpoint(checkpoint_path):
    """从checkpoint路径提取运行名称
    
    支持的路径格式:
    - outputs/run_20260108_115437/best_model.pth -> run_20260108_115437
    - outputs/ablation/01_baseline/run_20260108_115437/best_model.pth -> 01_baseline_run_20260108_115437
    """
    checkpoint_path = Path(checkpoint_path)
    parent = checkpoint_path.parent
    
    # 检查父目录是否是 run_xxx 格式
    if parent.name.startswith('run_'):
        # 检查上一级是否是 ablation 的子目录（如 01_baseline）
        grandparent = parent.parent
        if grandparent.name != 'outputs' and grandparent.name != 'ablation':
            # 这是 ablation 的子目录，如 01_baseline
            return f"{grandparent.name}_{parent.name}"
        else:
            return parent.name
    else:
        # 直接返回父目录名
        return parent.name


def main(args):
    """主评估函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 先加载checkpoint获取训练时的配置（防止数据泄漏）
    print(f'加载模型配置: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get('args', {})
    
    # 使用训练时的划分参数，确保数据划分一致
    test_size = train_args.get('test_size', args.test_size)
    val_size = train_args.get('val_size', args.val_size)
    seed = train_args.get('seed', args.seed)
    use_age_stratify = train_args.get('use_age_stratify', True)  # 默认使用分层抽样
    age_bin_width = train_args.get('age_bin_width', 10)
    min_age = train_args.get('min_age', args.min_age)
    max_age = train_args.get('max_age', args.max_age)
    
    print(f'\n⚠️  使用与训练一致的数据划分参数（防止数据泄漏）:')
    print(f'    test_size={test_size}, val_size={val_size}, seed={seed}')
    print(f'    use_age_stratify={use_age_stratify}, age_bin_width={age_bin_width}')
    print(f'    age_range={min_age}-{max_age}')
    
    # 加载数据
    print('\n加载数据集...')
    _, _, test_dataset = load_dataset(
        args.image_dir, 
        args.excel_path,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
        use_age_stratify=use_age_stratify,
        age_bin_width=age_bin_width,
        min_age=min_age,
        max_age=max_age
    )
    
    # 获取测试集的image_paths用于误差分析
    test_image_paths = test_dataset.image_paths
    
    # 创建DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model_name = train_args.get('model', 'resnet50')
    dropout = train_args.get('dropout', 0.5)
    
    print(f'\n创建模型: {model_name}, dropout={dropout}')
    model = get_model(model_name, pretrained=False, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f'模型训练轮次: {checkpoint["epoch"]}')
    print(f'验证集MAE: {checkpoint["val_mae"]:.2f} years')
    
    # 评估（传入image_paths获取文件名）
    print('\n开始评估...')
    metrics, predictions, targets, filenames = evaluate(model, test_loader, device, test_image_paths)
    
    # 打印结果
    print('\n' + '='*60)
    print('测试集评估结果')
    print('='*60)
    print(f'MAE (平均绝对误差):       {metrics["MAE"]:.2f} 岁')
    print(f'RMSE (均方根误差):        {metrics["RMSE"]:.2f} 岁')
    print(f'相关系数:                 {metrics["Correlation"]:.4f}')
    print(f'±5年内准确率:            {metrics["Accuracy_5years"]:.1f}%')
    print(f'±10年内准确率:           {metrics["Accuracy_10years"]:.1f}%')
    print(f'±15年内准确率:           {metrics["Accuracy_15years"]:.1f}%')
    print('='*60)
    
    # 确定输出目录
    if args.output_dir == './evaluation_results':
        # 如果使用默认输出目录，则自动从checkpoint路径推断
        run_name = get_run_name_from_checkpoint(args.checkpoint)
        output_dir = Path('./evaluation_results') / run_name
        print(f'\n自动设置输出目录: {output_dir}')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算年龄分段MAE
    age_group_mae = calculate_age_group_mae(predictions, targets, bin_width=10)
    
    # 打印年龄分段结果
    print('\n各年龄段MAE:')
    print('-' * 50)
    for group in age_group_mae:
        print(f"  {group['age_range']:>8}岁: MAE={group['mae']:.2f}, RMSE={group['rmse']:.2f}, n={group['count']}")
    print('-' * 50)
    
    # 添加年龄分段MAE到metrics
    metrics['age_group_mae'] = age_group_mae
    
    # 移除predictions和targets以减小文件大小
    metrics_save = {k: v for k, v in metrics.items() if k not in ['predictions', 'targets']}
    metrics_save['total_samples'] = len(predictions)
    
    with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_save, f, indent=2, ensure_ascii=False)
    print(f'\n指标已保存: {output_dir / "test_metrics.json"}')
    
    # 保存详细预测结果
    predictions_data = {
        'predictions': [float(x) for x in predictions.tolist()],
        'targets': [float(x) for x in targets.tolist()],
        'filenames': filenames if filenames else []
    }
    with open(output_dir / 'predictions.json', 'w') as f:
        json.dump(predictions_data, f)
    print(f'预测详情已保存: {output_dir / "predictions.json"}')
    
    # 保存误差分析结果（高误差/低误差样本列表）
    if filenames:
        print('\n进行误差分析...')
        error_analysis = save_error_analysis(
            predictions, targets, filenames, output_dir,
            top_n=args.top_n
        )
        # 将误差分析统计添加到metrics
        metrics_save['error_analysis'] = {
            'outlier_threshold': error_analysis['outlier_threshold'],
            'outlier_count': error_analysis['outlier_count'],
            'mean_mae': error_analysis['mean_mae'],
            'std_mae': error_analysis['std_mae']
        }
        # 更新保存的metrics
        with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_save, f, indent=2, ensure_ascii=False)
        
        # 分析图像特征差异
        print('\n分析高错误/低错误样本的图像特征差异...')
        feature_analysis = analyze_image_features(
            error_analysis, args.image_dir, output_dir
        )
        if feature_analysis and feature_analysis['significant_features']:
            print(f"\n⚠️  发现 {len(feature_analysis['significant_features'])} 个显著差异的图像特征")
            print("   详见: image_feature_analysis.txt")
    
    # 绘制图表
    print('\n绘制结果图表...')
    plot_results(predictions, targets, output_dir)
    
    print('\n评估完成!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估TA超声图像年龄预测模型')
    
    # 数据参数
    parser.add_argument('--image-dir', type=str, 
                       default='/home/szdx/LNX/data/TA/Healthy/Images',
                       help='图像文件夹路径')
    parser.add_argument('--excel-path', type=str,
                       default='/home/szdx/LNX/data/TA/characteristics.xlsx',
                       help='Excel标签文件路径')
    parser.add_argument('--checkpoint', type=str, 
                       default='./outputs/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--output-dir', type=str, 
                       default='./evaluation_results',
                       help='输出目录')
    
    # 数据集划分（默认从checkpoint自动读取，与训练保持一致）
    parser.add_argument('--test-size', type=float, default=0.15, help='测试集比例（默认从checkpoint读取）')
    parser.add_argument('--val-size', type=float, default=0.15, help='验证集比例（默认从checkpoint读取）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认从checkpoint读取）')
    parser.add_argument('--min-age', type=float, default=0, help='最小年龄（默认从checkpoint读取）')
    parser.add_argument('--max-age', type=float, default=100, help='最大年龄（默认从checkpoint读取）')
    
    # 评估参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--top-n', type=int, default=50, help='保存误差最大/最小的前N个样本（默认50）')
    
    args = parser.parse_args()
    
    main(args)
