"""
统一训练脚本 - 支持单模型/集成训练、多种损失函数、DDP
使用方法:
    # 单GPU训练（MAE损失）
    python train_mae.py
    
    # 多GPU训练（DDP）
    torchrun --nproc_per_node=6 train_mae.py --batch-size 32
    
    # 使用MSE损失
    python train_mae.py --loss mse
    
    # 256x256分辨率
    python train_mae.py --image-size 256
    
    # 集成训练（6个模型并行）
    python train_mae.py --ensemble --ensemble-models resnet50 efficientnet_b0 efficientnet_b1 convnext mobilenet_v3 regnet
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from model import get_model


def get_loss_function(loss_type='mae'):
    """获取损失函数
    
    Args:
        loss_type: 损失函数类型 ('mae', 'mse', 'smoothl1', 'huber')
    
    Returns:
        损失函数实例
    """
    loss_type = loss_type.lower()
    if loss_type == 'mae' or loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse' or loss_type == 'l2':
        return nn.MSELoss()
    elif loss_type == 'smoothl1':
        return nn.SmoothL1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}。支持: mae, mse, smoothl1, huber")


def load_dataset_module(image_size=224, use_age_stratify=False, age_bin_width=10):
    """动态加载数据集，支持图像尺寸和年龄分层配置
    
    Args:
        image_size: 图像尺寸 (224 或 256)
        use_age_stratify: 是否使用年龄分层抽样
        age_bin_width: 年龄分组宽度（默认10岁）
    
    Returns:
        配置好的 load_dataset 函数
    """
    from dataset import load_dataset as load_dataset_func
    
    def configured_load_dataset(image_dir, excel_path, test_size=0.2, val_size=0.1, random_state=42):
        return load_dataset_func(
            image_dir=image_dir,
            excel_path=excel_path,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            image_size=image_size,
            use_age_stratify=use_age_stratify,
            age_bin_width=age_bin_width
        )
    
    print(f"使用 {image_size}×{image_size} 分辨率" + 
          (f"，年龄分层抽样（每{age_bin_width}岁）" if use_age_stratify else ""))
    
    return configured_load_dataset


def setup_ddp():
    """初始化DDP环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def plot_training_curves(history, output_dir, epoch=None):
    """绘制训练曲线
    
    Args:
        history: 训练历史字典
        output_dir: 输出目录
        epoch: 当前epoch（可选，用于标题显示）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 设置标题
    if epoch is not None:
        fig.suptitle(f'Training Progress (Epoch {epoch})', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Training Progress - Final', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=11)
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE曲线
    axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('MAE (years)', fontsize=11)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. RMSE曲线
    axes[1, 0].plot(epochs, history['val_rmse'], 'g-', label='Val RMSE', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('RMSE (years)', fontsize=11)
    axes[1, 0].set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 最佳指标信息（文本显示）
    axes[1, 1].axis('off')
    best_val_mae = min(history['val_mae'])
    best_epoch_idx = history['val_mae'].index(best_val_mae) + 1
    best_val_rmse = history['val_rmse'][best_epoch_idx - 1]
    
    current_epoch = len(history['train_loss'])
    current_val_mae = history['val_mae'][-1]
    current_val_rmse = history['val_rmse'][-1]
    
    metrics_text = f"""Training Summary:

Current Epoch: {current_epoch}
Current Val MAE: {current_val_mae:.2f} years
Current Val RMSE: {current_val_rmse:.2f} years

━━━━━━━━━━━━━━━━━━━━━━━

Best Performance:
Best Epoch: {best_epoch_idx}
Best Val MAE: {best_val_mae:.2f} years
Best Val RMSE: {best_val_rmse:.2f} years

━━━━━━━━━━━━━━━━━━━━━━━

Improvement:
MAE Reduction: {history['val_mae'][0] - best_val_mae:.2f} years
({(1 - best_val_mae/history['val_mae'][0]) * 100:.1f}% improvement)
"""
    
    axes[1, 1].text(0.1, 0.5, metrics_text, 
                    fontsize=11, 
                    verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, is_main_process, max_grad_norm=1.0):
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    maes = AverageMeter()
    grad_norms = AverageMeter()  # 记录梯度范数
    
    if is_main_process:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    else:
        pbar = train_loader
        
    for images, ages in pbar:
        images = images.to(device)
        ages = ages.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, ages)
        
        # 计算MAE
        mae = torch.abs(outputs - ages).mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norms.update(grad_norm.item())
        
        optimizer.step()
        
        # 更新统计
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        maes.update(mae.item(), batch_size)
        
        # 更新进度条
        if is_main_process:
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'MAE': f'{maes.avg:.2f}',
                'grad': f'{grad_norms.avg:.2f}'
            })
    
    return losses.avg, maes.avg, grad_norms.avg


def validate(model, val_loader, criterion, device, epoch, is_main_process):
    """验证模型"""
    model.eval()
    
    losses = AverageMeter()
    maes = AverageMeter()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        if is_main_process:
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        else:
            pbar = val_loader
            
        for images, ages in pbar:
            images = images.to(device)
            ages = ages.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            # 计算MAE
            mae = torch.abs(outputs - ages).mean()
            
            # 更新统计
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            maes.update(mae.item(), batch_size)
            
            # 保存预测结果
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
            
            if is_main_process:
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'MAE': f'{maes.avg:.2f}'
                })
    
    # 计算RMSE
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return losses.avg, maes.avg, rmse


def train(args, model_name=None, gpu_id=None, is_ensemble=False):
    """主训练函数
    
    Args:
        args: 命令行参数
        model_name: 模型名称（集成训练时使用）
        gpu_id: GPU ID（集成训练时使用）
        is_ensemble: 是否为集成训练模式
    """
    # 初始化DDP
    rank, world_size, local_rank = setup_ddp()
    is_main_process = (rank == 0)
    
    # 设置设备
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
    elif world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process:
            print(f'使用DDP训练，World Size: {world_size}')
            print(f'GPU列表: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if is_main_process:
            print(f'使用设备: {device}')
    
    # 创建带时间戳的输出目录（仅主进程）
    if is_main_process:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f'run_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'输出目录: {output_dir}')
    else:
        output_dir = None
    
    # 同步所有进程
    if world_size > 1:
        dist.barrier()
    
    # 加载数据
    if is_main_process:
        print('加载数据集...')
    
    # 动态加载数据集模块
    load_dataset = load_dataset_module(args.image_size, args.use_age_stratify, args.age_bin_width)
    
    train_dataset, val_dataset, test_dataset = load_dataset(
        args.image_dir, 
        args.excel_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    # 保存完整配置文件（包含数据集信息）
    if is_main_process:
        # 统计数据集中的受试者信息
        from collections import defaultdict
        train_subjects = defaultdict(int)
        val_subjects = defaultdict(int)
        test_subjects = defaultdict(int)
        
        for img_path in train_dataset.image_paths:
            subject_id = Path(img_path).stem.split('_')[1]
            train_subjects[subject_id] += 1
        for img_path in val_dataset.image_paths:
            subject_id = Path(img_path).stem.split('_')[1]
            val_subjects[subject_id] += 1
        for img_path in test_dataset.image_paths:
            subject_id = Path(img_path).stem.split('_')[1]
            test_subjects[subject_id] += 1
        
        config = {
            # 脚本信息
            'script_name': 'train_mae.py',
            'script_version': '3.0',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'description': 'Optimized training with MAE loss, gradient clipping, and cosine annealing',
            
            # 运行环境
            'device': str(device),
            'world_size': world_size,
            'use_ddp': world_size > 1,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            
            # 数据集信息
            'dataset': {
                'image_dir': args.image_dir,
                'excel_path': args.excel_path,
                'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset),
                'train_subjects': len(train_subjects),
                'val_subjects': len(val_subjects),
                'test_subjects': len(test_subjects),
                'total_subjects': len(train_subjects) + len(val_subjects) + len(test_subjects),
                'test_size': args.test_size,
                'val_size': args.val_size,
                'random_seed': args.seed,
                'split_method': 'by_subject_id',
                'data_leakage_prevention': True,
            },
            
            # 模型配置
            'model': {
                'architecture': args.model,
                'pretrained': args.pretrained,
                'dropout': args.dropout,
                'output_dim': 1,
                'task': 'age_regression',
            },
            
            # 训练配置
            'training': {
                'loss_function': args.loss.upper(),
                'image_size': args.image_size,
                'optimizer': 'AdamW',
                'optimizer_params': {
                    'betas': (0.9, 0.999),
                },
                'lr_scheduler': 'CosineAnnealingLR',
                'scheduler_params': {
                    'T_max': args.epochs,
                    'eta_min': args.eta_min,
                },
                'warmup_epochs': args.warmup_epochs,
                'max_grad_norm': args.max_grad_norm,
                'epochs': args.epochs,
                'patience': args.patience,
                'batch_size': args.batch_size,
                'effective_batch_size': args.batch_size * world_size,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'num_workers': args.num_workers,
                'plot_interval': 10,
            },
            
            # 优化技巧
            'optimizations': {
                'gradient_clipping': True,
                'warmup': True,
                'early_stopping': True,
                'cosine_annealing': True,
            },
            
            # 其他参数
            'output_dir': args.output_dir,
        }
        
        with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f'配置已保存: {output_dir / "config.json"}')
    
    # 使用DistributedSampler for DDP
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                             shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # 创建模型
    current_model = model_name if model_name else args.model
    if is_main_process:
        print(f'创建模型: {current_model}')
    model = get_model(current_model, pretrained=args.pretrained, dropout=args.dropout)
    model = model.to(device)
    
    # 使用DDP进行多GPU训练
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main_process:
            print(f'模型已包装为DDP，将在 {world_size} 个GPU上训练')
    
    # 定义损失函数和优化器
    criterion = get_loss_function(args.loss)
    if is_main_process:
        loss_name_map = {'mae': 'MAE (L1)', 'mse': 'MSE (L2)', 'smoothl1': 'Smooth L1', 'huber': 'Huber'}
        print(f'使用损失函数: {loss_name_map.get(args.loss.lower(), args.loss)}')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # 使用普通余弦退火（更平滑）
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,  # 整个训练周期
        eta_min=args.eta_min  # 最小学习率
    )
    if is_main_process:
        print(f'\n学习率配置:')
        print(f'  初始学习率: {args.lr:.2e}')
        print(f'  使用CosineAnnealingLR: T_max={args.epochs}, eta_min={args.eta_min:.2e}')
        print(f'  梯度裁剪: max_norm={args.max_grad_norm}')
        print(f'  Warmup epochs: {args.warmup_epochs}\n')
    
    # 训练循环
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0  # 早停计数器
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }
    
    if is_main_process:
        print('\n开始训练...')
    for epoch in range(1, args.epochs + 1):
        # 设置sampler的epoch（DDP需要）
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # Warmup阶段：线性增加学习率
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # 训练
        train_loss, train_mae, avg_grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, is_main_process, args.max_grad_norm
        )
        
        # 验证
        val_loss, val_mae, val_rmse = validate(
            model, val_loader, criterion, device, epoch, is_main_process
        )
        
        # 更新学习率（warmup后才使用余弦退火）
        if epoch > args.warmup_epochs:
            scheduler.step()
        
        # 记录历史（转换为Python原生float类型）
        history['train_loss'].append(float(train_loss))
        history['train_mae'].append(float(train_mae))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_rmse'].append(float(val_rmse))
        
        # 打印统计（仅主进程）
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            warmup_status = ' [Warmup]' if epoch <= args.warmup_epochs else ''
            print(f'\nEpoch {epoch}/{args.epochs} (lr={current_lr:.2e}, grad={avg_grad_norm:.2f}){warmup_status}:')
            print(f'  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f} years')
            print(f'  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f} years, RMSE: {val_rmse:.2f} years')
        
        # 保存最佳模型（仅主进程）
        if is_main_process:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                patience_counter = 0  # 重置早停计数器
                # 保存模型时处理DDP包装
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'args': vars(args)
                }, output_dir / 'best_model.pth')
                print(f'  ✓ 保存最佳模型 (MAE: {val_mae:.2f} years)')
            else:
                patience_counter += 1
                print(f'  ⚠ 没有改善 ({patience_counter}/{args.patience})')
                
                # 早停检查
                if patience_counter >= args.patience:
                    print(f'\n早停触发！连续 {args.patience} 个epoch没有改善，停止训练。')
                    print(f'最佳模型: Epoch {best_epoch}, MAE: {best_val_mae:.2f} years')
                    break
            
            # 每10轮保存检查点并绘制训练曲线
            if epoch % 10 == 0:
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_rmse': val_rmse
                }, output_dir / f'checkpoint_epoch_{epoch}.pth')
                
                # 绘制当前训练曲线
                print(f'  ✓ 更新训练曲线图 (Epoch {epoch})')
                plot_training_curves(history, output_dir, epoch=epoch)
    
    # 保存训练历史（仅主进程）
    if is_main_process:
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f'训练历史已保存: {output_dir / "history.json"}')
        
        # 绘制最终训练曲线
        print('\n绘制最终训练曲线...')
        plot_training_curves(history, output_dir)
        
        print(f'\n训练完成!')
        print(f'最佳模型: Epoch {best_epoch}, MAE: {best_val_mae:.2f} years')
        print(f'所有结果保存在: {output_dir}')
        print(f'  - 最佳模型: best_model.pth')
        print(f'  - 训练曲线: training_curves.png')
        print(f'  - 训练历史: history.json')
        print(f'  - 配置文件: config.json')
    
    # 清理DDP
    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练TA超声图像年龄预测模型（支持DDP）')
    
    # 数据参数
    parser.add_argument('--image-dir', type=str, 
                       default='/home/szdx/LNX/data/TA/Healthy/Images',
                       help='图像文件夹路径')
    parser.add_argument('--excel-path', type=str,
                       default='/home/szdx/LNX/data/TA/characteristics.xlsx',
                       help='Excel标签文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='./outputs',
                       help='输出目录')
    
    # 数据集划分
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use-age-stratify', action='store_true', 
                       help='使用按年龄分层抽样（提高泛化能力）')
    parser.add_argument('--age-bin-width', type=int, default=10,
                       help='年龄分组宽度（岁），仅在--use-age-stratify时有效')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet_b0', 'efficientnet_b1', 'convnext', 'mobilenet_v3', 'regnet'],
                       help='模型架构')
    parser.add_argument('--loss', type=str, default='mae',
                       choices=['mae', 'mse', 'smoothl1', 'huber'],
                       help='损失函数类型')
    parser.add_argument('--image-size', type=int, default=224,
                       choices=[224, 256],
                       help='输入图像尺寸')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='使用ImageNet预训练权重')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=500, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值（连续多少轮没有改善就停止）')
    parser.add_argument('--batch-size', type=int, default=32, help='每个GPU的批次大尋')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率（默认1e-4，更稳定）')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=8, help='数据加载线程数')
    
    # 学习率调度参数
    parser.add_argument('--eta-min', type=float, default=1e-7, help='最小学习率')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup轮数（线性增加lr）')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 集成训练参数
    parser.add_argument('--ensemble', action='store_true', help='启用集成训练模式')
    parser.add_argument('--ensemble-models', nargs='+', 
                       default=['resnet50', 'efficientnet_b0', 'efficientnet_b1', 'convnext', 'mobilenet_v3', 'regnet'],
                       help='集成训练的模型列表')
    parser.add_argument('--ensemble-gpus', nargs='+', type=int,
                       default=[0, 1, 2, 3, 4, 5],
                       help='集成训练使用的GPU列表')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 开始训练
    if args.ensemble:
        # 集成训练模式
        import multiprocessing as mp
        
        print(f"\n{'='*80}")
        print(f"启动集成训练: {len(args.ensemble_models)} 个模型")
        print(f"模型列表: {', '.join(args.ensemble_models)}")
        print(f"GPU分配: {args.ensemble_gpus}")
        print(f"{'='*80}\n")
        
        # 创建进程池
        processes = []
        for i, model_name in enumerate(args.ensemble_models):
            gpu_id = args.ensemble_gpus[i % len(args.ensemble_gpus)]
            p = mp.Process(target=train, args=(args, model_name, gpu_id, True))
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print(f"\n{'='*80}")
        print(f"集成训练完成！")
        print(f"{'='*80}")
    else:
        # 单模型训练
        train(args)
