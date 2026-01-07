"""
计算超声图像的纹理特征并分析与年龄的相关性

纹理特征包括：
- 统计特征：均值、标准差、偏度、峰度
- GLCM特征：对比度、相关性、能量、同质性
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm


def compute_statistical_features(img_array):
    """计算统计特征"""
    return {
        'mean': np.mean(img_array),
        'std': np.std(img_array),
        'skewness': skew(img_array.flatten()),
        'kurtosis': kurtosis(img_array.flatten()),
        'entropy': -np.sum(np.histogram(img_array, bins=256, density=True)[0] * 
                          np.log2(np.histogram(img_array, bins=256, density=True)[0] + 1e-10))
    }


def compute_glcm_features(img_array):
    """计算GLCM纹理特征"""
    # 归一化到0-255
    img_normalized = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    
    # 计算GLCM (距离=1, 4个方向)
    glcm = graycomatrix(img_normalized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256, symmetric=True, normed=True)
    
    # 提取特征（取4个方向的平均值）
    return {
        'contrast': graycoprops(glcm, 'contrast')[0].mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0].mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity')[0].mean(),
        'energy': graycoprops(glcm, 'energy')[0].mean(),
        'correlation': graycoprops(glcm, 'correlation')[0].mean(),
        'ASM': graycoprops(glcm, 'ASM')[0].mean()
    }


def compute_all_features(image_path):
    """计算所有纹理特征"""
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img)
        
        # 统计特征
        stat_features = compute_statistical_features(img_array)
        
        # GLCM特征
        glcm_features = compute_glcm_features(img_array)
        
        # 合并所有特征
        features = {**stat_features, **glcm_features}
        
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_age_labels(excel_path):
    """从Excel文件加载年龄标签"""
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
    """从文件名提取受试者ID"""
    name = Path(filename).stem
    if '_' in name:
        parts = name.split('_')
        potential_id = parts[0]
        if potential_id.isdigit():
            return potential_id
        for part in parts:
            if part.isdigit():
                return part
    if name.isdigit():
        return name
    import re
    numbers = re.findall(r'\d+', name)
    if numbers:
        return numbers[0]
    return None


def analyze_texture_features(image_dir, excel_path, output_dir, muscle_name='TA'):
    """主分析函数"""
    print("="*60)
    print(f"超声图像纹理特征分析 - {muscle_name}肌肉")
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
    print("\n2. 计算纹理特征...")
    image_dir = Path(image_dir)
    
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    print(f"   找到 {len(image_files)} 张图像")
    
    for img_path in tqdm(image_files, desc="处理图像"):
        subject_id = extract_subject_id(img_path.name)
        if subject_id is None or subject_id not in age_dict:
            continue
        
        age = age_dict[subject_id]
        features = compute_all_features(img_path)
        
        if features is not None:
            result = {
                'image_name': img_path.name,
                'subject_id': subject_id,
                'age': age,
                **features
            }
            results.append(result)
    
    df = pd.DataFrame(results)
    print(f"\n3. 成功处理 {len(df)} 张图像")
    
    # 保存原始数据
    csv_path = data_dir / 'texture_features.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n4. 原始数据已保存: {csv_path}")
    
    # 计算相关性
    print("\n5. 相关性分析:")
    feature_cols = [col for col in df.columns if col not in ['image_name', 'subject_id', 'age']]
    
    correlations = []
    for feature in feature_cols:
        corr, pvalue = stats.pearsonr(df['age'], df[feature])
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'p_value': pvalue,
            'abs_correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    print("\n   特征排名（按相关性强度）:")
    for _, row in corr_df.iterrows():
        print(f"   {row['feature']:20s}: r = {row['correlation']:7.4f}, p = {row['p_value']:.2e}")
    
    # 保存相关性数据
    corr_path = data_dir / 'correlations.csv'
    corr_df.to_csv(corr_path, index=False, encoding='utf-8-sig')
    
    # 绘制相关性图
    print("\n6. 绘制可视化图表...")
    
    # 选择top 9个特征绘制
    top_features = corr_df.head(9)['feature'].tolist()
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        ax.scatter(df['age'], df[feature], alpha=0.5, s=10)
        
        # 添加回归线
        z = np.polyfit(df['age'], df[feature], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['age'].min(), df['age'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # 获取相关系数
        corr = corr_df[corr_df['feature'] == feature]['correlation'].values[0]
        pval = corr_df[corr_df['feature'] == feature]['p_value'].values[0]
        
        ax.set_xlabel('年龄 (岁)', fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.set_title(f'{feature}\nr={corr:.4f}, p={pval:.2e}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = figures_dir / 'texture_features_correlation.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   相关性散点图已保存: {fig_path}")
    plt.close()
    
    # 绘制特征相关性热图
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f'{muscle_name}肌肉 - 特征间相关性热图', fontsize=14, pad=20)
    plt.tight_layout()
    heatmap_path = figures_dir / 'feature_correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"   特征相关性热图已保存: {heatmap_path}")
    plt.close()
    
    # 保存统计摘要
    summary_path = data_dir / 'texture_analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"超声图像纹理特征分析 - {muscle_name}肌肉\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"年龄范围: {df['age'].min():.1f} - {df['age'].max():.1f} 岁\n\n")
        
        f.write("特征相关性排名:\n")
        f.write("-" * 60 + "\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row['feature']:20s}: r = {row['correlation']:7.4f}, p = {row['p_value']:.2e}\n")
        
        f.write("\n" + "="*60 + "\n")
        
        # 找出最强相关特征
        best_feature = corr_df.iloc[0]
        f.write(f"\n最强相关特征: {best_feature['feature']}\n")
        f.write(f"相关系数: r = {best_feature['correlation']:.4f}\n")
        f.write(f"P值: {best_feature['p_value']:.2e}\n")
        
        if best_feature['p_value'] < 0.05:
            f.write("✓ 相关性显著 (p < 0.05)\n")
        else:
            f.write("✗ 相关性不显著 (p ≥ 0.05)\n")
    
    print(f"   统计摘要已保存: {summary_path}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    
    return df, corr_df


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        excel_path = sys.argv[2] if len(sys.argv) > 2 else '/home/szdx/LNX/data/TA/characteristics.xlsx'
        muscle_name = sys.argv[3] if len(sys.argv) > 3 else 'TA'
    else:
        image_dir = '/home/szdx/LNX/data/TA/Healthy/Images'
        excel_path = '/home/szdx/LNX/data/TA/characteristics.xlsx'
        muscle_name = 'TA'
    
    output_dir = Path(__file__).parent.parent / 'results'
    
    df, corr_df = analyze_texture_features(image_dir, excel_path, output_dir, muscle_name)
    
    print(f"\n结果保存位置:")
    print(f"  - 数据: {output_dir}/{muscle_name}/data/")
    print(f"  - 图表: {output_dir}/{muscle_name}/figures/")
