"""
评估脚本
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

from dataset import load_dataset, get_dataloaders
from model import get_model


def evaluate(model, test_loader, device):
    """评估模型"""
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
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': correlation,
        'Accuracy_5years': acc_5,
        'Accuracy_10years': acc_10,
        'Accuracy_15years': acc_15,
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist()
    }
    
    return metrics, all_preds, all_targets


def plot_results(predictions, targets, output_dir):
    """绘制结果图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置样式
    sns.set_style('whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False
    
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
    
    ax.set_xlabel('真实年龄 (岁)', fontsize=12)
    ax.set_ylabel('预测年龄 (岁)', fontsize=12)
    ax.set_title('预测年龄 vs 真实年龄', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右图：误差分布直方图
    ax = axes[1]
    errors = predictions - targets
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax.axvline(errors.mean(), color='g', linestyle='--', lw=2, 
               label=f'Mean Error: {errors.mean():.2f}')
    ax.set_xlabel('预测误差 (岁)', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('预测误差分布', fontsize=14, fontweight='bold')
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
    
    ax.set_xlabel('平均年龄 (岁)', fontsize=12)
    ax.set_ylabel('差异 (预测 - 真实, 岁)', fontsize=12)
    ax.set_title('Bland-Altman图', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bland_altman.png', dpi=300, bbox_inches='tight')
    print(f'图表已保存: {output_dir / "bland_altman.png"}')
    plt.close()


def main(args):
    """主评估函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print('加载数据集...')
    _, _, test_dataset = load_dataset(
        args.image_dir, 
        args.excel_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    _, _, test_loader = get_dataloaders(
        None, None, test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 加载模型
    print(f'加载模型: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 创建模型
    model_args = checkpoint.get('args', {})
    model_name = model_args.get('model', 'resnet50')
    dropout = model_args.get('dropout', 0.5)
    
    model = get_model(model_name, pretrained=False, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f'模型训练轮次: {checkpoint["epoch"]}')
    print(f'验证集MAE: {checkpoint["val_mae"]:.2f} years')
    
    # 评估
    print('\n开始评估...')
    metrics, predictions, targets = evaluate(model, test_loader, device)
    
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
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\n指标已保存: {output_dir / "test_metrics.json"}')
    
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
    
    # 数据集划分（需要与训练时一致）
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 评估参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    
    args = parser.parse_args()
    
    main(args)
