"""
验证数据集划分是否存在数据泄露
检查训练集、验证集、测试集之间是否有相同的受试者ID
"""

from dataset import load_dataset
from pathlib import Path


def extract_subject_id(image_path):
    """从图像路径提取受试者ID"""
    filename = Path(image_path).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None


def verify_no_leakage(train_dataset, val_dataset, test_dataset):
    """
    验证数据集划分没有泄露
    
    Args:
        train_dataset, val_dataset, test_dataset: 数据集对象
    """
    # 提取每个数据集的受试者ID
    train_subjects = set([extract_subject_id(p) for p in train_dataset.image_paths])
    val_subjects = set([extract_subject_id(p) for p in val_dataset.image_paths])
    test_subjects = set([extract_subject_id(p) for p in test_dataset.image_paths])
    
    # 移除None（如果有解析失败的）
    train_subjects.discard(None)
    val_subjects.discard(None)
    test_subjects.discard(None)
    
    print("\n" + "="*80)
    print("数据泄露检查报告")
    print("="*80)
    
    print(f"\n【数据集统计】")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_subjects)} 个受试者")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_subjects)} 个受试者")
    print(f"  测试集: {len(test_dataset)} 样本, {len(test_subjects)} 个受试者")
    print(f"  总计: {len(train_dataset) + len(val_dataset) + len(test_dataset)} 样本, "
          f"{len(train_subjects) + len(val_subjects) + len(test_subjects)} 个受试者")
    
    # 检查重叠
    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects
    
    print(f"\n【数据泄露检查】")
    
    has_leakage = False
    
    if train_val_overlap:
        print(f"  ❌ 训练集 ∩ 验证集: {len(train_val_overlap)} 个受试者重叠!")
        print(f"     重叠ID: {sorted(list(train_val_overlap))[:10]}...")
        has_leakage = True
    else:
        print(f"  ✅ 训练集 ∩ 验证集: 无重叠")
    
    if train_test_overlap:
        print(f"  ❌ 训练集 ∩ 测试集: {len(train_test_overlap)} 个受试者重叠!")
        print(f"     重叠ID: {sorted(list(train_test_overlap))[:10]}...")
        has_leakage = True
    else:
        print(f"  ✅ 训练集 ∩ 测试集: 无重叠")
    
    if val_test_overlap:
        print(f"  ❌ 验证集 ∩ 测试集: {len(val_test_overlap)} 个受试者重叠!")
        print(f"     重叠ID: {sorted(list(val_test_overlap))[:10]}...")
        has_leakage = True
    else:
        print(f"  ✅ 验证集 ∩ 测试集: 无重叠")
    
    # 总结
    print(f"\n【总结】")
    if has_leakage:
        print(f"  ⚠️  发现数据泄露！需要修复数据集划分逻辑")
    else:
        print(f"  ✅ 未发现数据泄露，数据集划分正确！")
        print(f"  👍 所有受试者的图像都被正确分配到独立的数据集中")
    
    print("="*80 + "\n")
    
    return not has_leakage


if __name__ == '__main__':
    print("加载数据集...")
    train_dataset, val_dataset, test_dataset = load_dataset(
        base_path='/home/szdx/LNX/data/TA/Healthy/Images',
        excel_path='/home/szdx/LNX/data/TA/characteristics.xlsx',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # 验证
    is_valid = verify_no_leakage(train_dataset, val_dataset, test_dataset)
    
    if is_valid:
        print("🎉 数据集可以安全用于训练！")
    else:
        print("❌ 数据集存在问题，请检查！")
