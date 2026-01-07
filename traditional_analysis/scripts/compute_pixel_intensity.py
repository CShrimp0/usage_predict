"""
计算超声图像的平均灰度值并分析与年龄的相关性

功能：
1. 读取所有超声图像
2. 计算每张图的平均灰度值（Mean Pixel Intensity）
3. 结合年龄标签
4. 绘制散点图并计算相关系数
5. 保存结果到CSV和图表
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# 设置中文字体 - 使用Linux系统字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def compute_mean_intensity(image_path):
    """
    计算图像的平均灰度值
    
    Args:
        image_path: 图像路径
    
    Returns:
        mean_intensity: 平均灰度值 (0-255)
    """
    try:
        img = Image.open(image_path)
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        # 计算平均值
        img_array = np.array(img)
        mean_intensity = np.mean(img_array)
        
        return mean_intensity
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_age_labels(excel_path):
    """
    从Excel文件加载年龄标签
    
    Args:
        excel_path: Excel文件路径
    
    Returns:
        age_dict: {subject_id: age}
    """
    df = pd.read_excel(excel_path)
    
    age_dict = {}
    
    # 处理Healthy列
    healthy_df = df[['Healthy', 'Unnamed: 1']].copy()
    healthy_df.columns = ['Number', 'Age']
    healthy_df = healthy_df[1:].dropna()
    
    for _, row in healthy_df.iterrows():
        try:
            subject_id = str(int(float(row['Number'])))
            age = float(row['Age'])
            age_dict[subject_id] = age
        except (ValueError, TypeError):
            continue
    
    # 处理Pathological列
    path_df = df[['Pathological', 'Unnamed: 3']].copy()
    path_df.columns = ['Number', 'Age']
    path_df = path_df[1:].dropna()
    
    for _, row in path_df.iterrows():
        try:
            subject_id = str(int(float(row['Number'])))
            age = float(row['Age'])
            age_dict[subject_id] = age
        except (ValueError, TypeError):
            continue
    
    return age_dict


def extract_subject_id(filename):
    """
    从文件名提取受试者ID
    
    Args:
        filename: 文件名
    
    Returns:
        subject_id: 受试者ID
    """
    # 去除扩展名
    name = Path(filename).stem
    
    # 尝试多种模式
    # 模式1: subject001_xxx
    if '_' in name:
        parts = name.split('_')
        potential_id = parts[0]
        if potential_id.isdigit():
            return potential_id
        # 尝试去除前缀
        for part in parts:
            if part.isdigit():
                return part
    
    # 模式2: 纯数字
    if name.isdigit():
        return name
    
    # 模式3: 提取数字部分
    import re
    numbers = re.findall(r'\d+', name)
    if numbers:
        return numbers[0]
    
    return None


def analyze_pixel_intensity(image_dir, excel_path, output_dir, muscle_name='TA'):
    """
    主分析函数
    
    Args:
        image_dir: 图像目录
        excel_path: Excel标签文件
        output_dir: 输出目录
        muscle_name: 肌肉名称（用于文件夹命名）
    """
    print("="*60)
    print(f"超声图像平均灰度值分析 - {muscle_name}肌肉")
    print("="*60)
    
    # 创建肌肉特定的输出目录
    muscle_dir = Path(output_dir) / muscle_name
    data_dir = muscle_dir / 'data'
    figures_dir = muscle_dir / 'figures'
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载年龄标签
    print("\n1. 加载年龄标签...")
    age_dict = load_age_labels(excel_path)
    print(f"   加载了 {len(age_dict)} 个受试者的年龄标签")
    
    # 遍历所有图像
    print("\n2. 计算图像平均灰度值...")
    image_dir = Path(image_dir)
    
    results = []
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    print(f"   找到 {len(image_files)} 张图像")
    
    for img_path in tqdm(image_files, desc="处理图像"):
        # 提取subject_id
        subject_id = extract_subject_id(img_path.name)
        
        if subject_id is None:
            continue
        
        # 获取年龄
        if subject_id not in age_dict:
            continue
        
        age = age_dict[subject_id]
        
        # 计算平均灰度值
        mean_intensity = compute_mean_intensity(img_path)
        
        if mean_intensity is not None:
            results.append({
                'image_name': img_path.name,
                'subject_id': subject_id,
                'age': age,
                'mean_intensity': mean_intensity
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n3. 成功处理 {len(df)} 张图像")
    print(f"   年龄范围: {df['age'].min():.1f} - {df['age'].max():.1f} 岁")
    print(f"   灰度范围: {df['mean_intensity'].min():.1f} - {df['mean_intensity'].max():.1f}")
    
    # 保存原始数据
    csv_path = data_dir / 'pixel_intensity.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n4. 原始数据已保存: {csv_path}")
    
    # 计算统计信息
    print("\n5. 统计分析:")
    print(f"   平均灰度值: {df['mean_intensity'].mean():.2f} ± {df['mean_intensity'].std():.2f}")
    print(f"   平均年龄: {df['age'].mean():.2f} ± {df['age'].std():.2f}")
    
    # 计算相关系数
    pearson_corr, pearson_pvalue = stats.pearsonr(df['age'], df['mean_intensity'])
    spearman_corr, spearman_pvalue = stats.spearmanr(df['age'], df['mean_intensity'])
    
    print(f"\n6. 相关性分析:")
    print(f"   Pearson相关系数: r = {pearson_corr:.4f}, p = {pearson_pvalue:.4e}")
    print(f"   Spearman相关系数: ρ = {spearman_corr:.4f}, p = {spearman_pvalue:.4e}")
    
    if abs(pearson_corr) < 0.1:
        print("   ⚠️  相关性很弱（|r| < 0.1）")
    elif abs(pearson_corr) < 0.3:
        print("   📊 相关性较弱（0.1 ≤ |r| < 0.3）")
    elif abs(pearson_corr) < 0.5:
        print("   📈 中等相关性（0.3 ≤ |r| < 0.5）")
    else:
        print("   🎯 强相关性（|r| ≥ 0.5）")
    
    # 绘制散点图
    print("\n7. 绘制散点图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：散点图 + 回归线
    ax1 = axes[0]
    ax1.scatter(df['age'], df['mean_intensity'], alpha=0.5, s=20)
    
    # 添加回归线
    z = np.polyfit(df['age'], df['mean_intensity'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['age'].min(), df['age'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
             label=f'回归线: y={z[0]:.3f}x+{z[1]:.2f}')
    
    ax1.set_xlabel('年龄 (岁)', fontsize=12)
    ax1.set_ylabel('平均灰度值', fontsize=12)
    ax1.set_title(f'年龄 vs 平均灰度值\nPearson r={pearson_corr:.4f}, p={pearson_pvalue:.2e}', 
                  fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：分箱统计
    ax2 = axes[1]
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    df['age_group'] = pd.cut(df['age'], bins=age_bins)
    
    # 计算每个年龄组的平均灰度值和标准差
    grouped = df.groupby('age_group', observed=True)['mean_intensity'].agg(['mean', 'std', 'count'])
    
    x_pos = range(len(grouped))
    ax2.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                          for interval in grouped.index], rotation=45)
    ax2.set_xlabel('年龄组 (岁)', fontsize=12)
    ax2.set_ylabel('平均灰度值', fontsize=12)
    ax2.set_title('各年龄组的平均灰度值', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加样本数标注
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax2.text(i, row['mean'] + row['std'] + 2, f"n={int(row['count'])}", 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = figures_dir / 'age_vs_intensity.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   图表已保存: {fig_path}")
    
    # 保存统计摘要
    summary_path = data_dir / 'analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("超声图像平均灰度值分析 - 统计摘要\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"年龄范围: {df['age'].min():.1f} - {df['age'].max():.1f} 岁\n")
        f.write(f"平均年龄: {df['age'].mean():.2f} ± {df['age'].std():.2f} 岁\n\n")
        
        f.write(f"灰度值范围: {df['mean_intensity'].min():.1f} - {df['mean_intensity'].max():.1f}\n")
        f.write(f"平均灰度值: {df['mean_intensity'].mean():.2f} ± {df['mean_intensity'].std():.2f}\n\n")
        
        f.write("相关性分析:\n")
        f.write(f"  Pearson相关系数: r = {pearson_corr:.4f}\n")
        f.write(f"  P值: {pearson_pvalue:.4e}\n")
        f.write(f"  Spearman相关系数: ρ = {spearman_corr:.4f}\n")
        f.write(f"  P值: {spearman_pvalue:.4e}\n\n")
        
        f.write("线性回归方程:\n")
        f.write(f"  灰度值 = {z[0]:.4f} × 年龄 + {z[1]:.2f}\n\n")
        
        if pearson_pvalue < 0.05:
            f.write("✓ 相关性显著 (p < 0.05)\n")
        else:
            f.write("✗ 相关性不显著 (p ≥ 0.05)\n")
    
    print(f"   统计摘要已保存: {summary_path}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    
    return df


if __name__ == '__main__':
    # 默认路径（可以通过命令行参数修改）
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        excel_path = sys.argv[2] if len(sys.argv) > 2 else '/home/szdx/LNX/data/TA/characteristics.xlsx'
        muscle_name = sys.argv[3] if len(sys.argv) > 3 else 'TA'
    else:
        image_dir = '/home/szdx/LNX/data/TA/Healthy/Images'
        excel_path = '/home/szdx/LNX/data/TA/characteristics.xlsx'
        muscle_name = 'TA'
    
    output_dir = Path(__file__).parent.parent / 'results'
    
    # 运行分析
    df = analyze_pixel_intensity(image_dir, excel_path, output_dir, muscle_name)
    
    print(f"\n结果保存位置:")
    print(f"  - 数据: {output_dir}/data/")
    print(f"  - 图表: {output_dir}/figures/")
