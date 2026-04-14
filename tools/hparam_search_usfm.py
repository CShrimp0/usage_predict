import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import train as train_module

MIN_AGE_FIXED = 18
SEARCH_ROOT = REPO_ROOT / 'outputs' / 'hparam_search_usfm'
TRACKED_TRAIN_KEYS = [
    'model',
    'pretrained_path',
    'freeze_backbone',
    'usfm_global_pool',
    'use_aux_features',
    'aux_gender',
    'aux_bmi',
    'aux_skewness',
    'aux_intensity',
    'aux_clarity',
    'aux_hidden_dim',
    'loss',
    'image_size',
    'batch_size',
    'dropout',
    'lr',
    'weight_decay',
    'optimizer',
    'scheduler',
    'warmup_epochs',
    'eta_min',
    'lr_patience',
    'lr_factor',
    'lr_min_lr',
    'max_grad_norm',
    'seed',
    'min_age',
    'max_age',
    'age_bin_width',
]

PROFILE_SPECS = {
    'full_ft_phase1': {
        'description': 'USFM 图像-only 全量微调粗搜',
        'mode': 'image',
        'freeze_backbone': False,
        'default_n_trials': 32,
        'default_max_epochs': 40,
        'default_final_epochs': 200,
        'default_final_patience': 40,
        'default_pruner_warmup_epochs': 8,
        'default_pruner_startup_trials': 8,
        'batch_size_choices': [8, 16, 24],
        'lr_range': (5e-6, 2e-5),
        'weight_decay_range': (1e-5, 5e-4),
        'dropout_range': (0.10, 0.40),
        'scheduler_choices': ['cosine', 'plateau'],
        'warmup_epochs_range': (3, 8),
        'eta_min_range': (1e-7, 2e-6),
        'lr_patience_range': (4, 8),
        'lr_factor_range': (0.20, 0.50),
        'lr_min_lr_range': (1e-7, 2e-6),
        'max_grad_norm_choices': [0.5, 1.0, 2.0],
        'usfm_global_pool_choices': ['avg', 'token'],
    },
    'full_ft_phase2': {
        'description': 'USFM 图像-only 全量微调细搜',
        'mode': 'image',
        'freeze_backbone': False,
        'default_n_trials': 16,
        'default_max_epochs': 80,
        'default_final_epochs': 220,
        'default_final_patience': 50,
        'default_pruner_warmup_epochs': 12,
        'default_pruner_startup_trials': 6,
        'batch_size_choices': [16, 24],
        'lr_range': (7e-6, 1.5e-5),
        'weight_decay_range': (3e-5, 2e-4),
        'dropout_range': (0.15, 0.35),
        'scheduler_choices': ['cosine', 'plateau'],
        'warmup_epochs_range': (4, 8),
        'eta_min_range': (1e-7, 1e-6),
        'lr_patience_range': (4, 7),
        'lr_factor_range': (0.25, 0.50),
        'lr_min_lr_range': (1e-7, 1e-6),
        'max_grad_norm_choices': [0.5, 1.0],
        'usfm_global_pool_choices': ['avg', 'token'],
    },
    'frozen_phase1': {
        'description': 'USFM 图像-only 冻结主干粗搜',
        'mode': 'image',
        'freeze_backbone': True,
        'default_n_trials': 24,
        'default_max_epochs': 35,
        'default_final_epochs': 160,
        'default_final_patience': 35,
        'default_pruner_warmup_epochs': 6,
        'default_pruner_startup_trials': 6,
        'batch_size_choices': [16, 32, 48],
        'lr_range': (3e-5, 3e-4),
        'weight_decay_range': (1e-6, 1e-3),
        'dropout_range': (0.20, 0.50),
        'scheduler_choices': ['cosine', 'plateau'],
        'warmup_epochs_range': (2, 6),
        'eta_min_range': (1e-7, 5e-6),
        'lr_patience_range': (3, 6),
        'lr_factor_range': (0.20, 0.50),
        'lr_min_lr_range': (1e-7, 5e-6),
        'max_grad_norm_choices': [0.5, 1.0, 2.0],
        'usfm_global_pool_choices': ['avg', 'token'],
    },
    'multimodal_phase1': {
        'description': 'USFM 全量微调 + 辅助特征粗搜',
        'mode': 'multimodal',
        'freeze_backbone': False,
        'default_n_trials': 12,
        'default_max_epochs': 45,
        'default_final_epochs': 200,
        'default_final_patience': 45,
        'default_pruner_warmup_epochs': 8,
        'default_pruner_startup_trials': 6,
        'batch_size_choices': [8, 16],
        'lr_range': (5e-6, 2e-5),
        'weight_decay_range': (1e-5, 5e-4),
        'dropout_range': (0.10, 0.40),
        'scheduler_choices': ['cosine', 'plateau'],
        'warmup_epochs_range': (3, 8),
        'eta_min_range': (1e-7, 2e-6),
        'lr_patience_range': (4, 8),
        'lr_factor_range': (0.20, 0.50),
        'lr_min_lr_range': (1e-7, 2e-6),
        'max_grad_norm_choices': [0.5, 1.0, 2.0],
        'usfm_global_pool_choices': ['avg', 'token'],
        'aux_hidden_dim_choices': [16, 32, 64],
        'aux_feature_sets': ['demographics', 'image_stats', 'all'],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='USFM 专用 Optuna 超参搜索')
    parser.add_argument('--study-name', type=str, default='usfm_search', help='Optuna study 名称')
    parser.add_argument('--profile', type=str, default='full_ft_phase1',
                        choices=sorted(PROFILE_SPECS.keys()), help='USFM 搜索 profile')
    parser.add_argument('--pretrained-path', type=str, required=True,
                        help='USFM 官方预训练权重路径')
    parser.add_argument('--n-trials', type=int, default=None, help='trial 数量；默认取 profile 推荐值')
    parser.add_argument('--max-epochs', type=int, default=None, help='短跑搜索的最大 epoch；默认取 profile 推荐值')
    parser.add_argument('--final-epochs', type=int, default=None, help='导出的 full-run 推荐 epoch')
    parser.add_argument('--final-patience', type=int, default=None, help='导出的 full-run 推荐早停 patience')
    parser.add_argument('--gpus', nargs='*', type=int, default=None, help='可用 GPU 列表')
    parser.add_argument('--n-workers', type=int, default=None, help='并行 worker 数；默认=GPU数')
    parser.add_argument('--timeout', type=int, default=None, help='搜索超时（秒）')
    parser.add_argument('--storage', type=str, default=None, help='Optuna sqlite 存储路径')
    parser.add_argument('--pruner-warmup-epochs', type=int, default=None, help='剪枝 warmup epoch')
    parser.add_argument('--pruner-startup-trials', type=int, default=None, help='剪枝 startup trials')
    parser.add_argument('--base-params', type=str, default=None,
                        help='基础训练参数 JSON，适合覆盖数据路径/年龄范围/num_workers 等')
    parser.add_argument('--worker', action='store_true', help='仅运行 worker 模式')
    opt_args, train_args = parser.parse_known_args()

    spec = PROFILE_SPECS[opt_args.profile]
    if opt_args.n_trials is None:
        opt_args.n_trials = spec['default_n_trials']
    if opt_args.max_epochs is None:
        opt_args.max_epochs = spec['default_max_epochs']
    if opt_args.final_epochs is None:
        opt_args.final_epochs = spec['default_final_epochs']
    if opt_args.final_patience is None:
        opt_args.final_patience = spec['default_final_patience']
    if opt_args.pruner_warmup_epochs is None:
        opt_args.pruner_warmup_epochs = spec['default_pruner_warmup_epochs']
    if opt_args.pruner_startup_trials is None:
        opt_args.pruner_startup_trials = spec['default_pruner_startup_trials']

    return opt_args, train_args


def resolve_gpus(explicit_gpus):
    if explicit_gpus:
        return explicit_gpus
    env_gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env_gpus:
        return [int(x) for x in env_gpus.split(',') if x.strip()]
    count = torch.cuda.device_count()
    if count <= 0:
        return [None]
    return list(range(count))


def make_storage_url(storage_path):
    return f"sqlite:///{storage_path}"


def load_base_params(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_params(args, params):
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)


def reset_aux_feature_flags(args):
    args.use_aux_features = False
    args.aux_gender = False
    args.aux_bmi = False
    args.aux_skewness = False
    args.aux_intensity = False
    args.aux_clarity = False


def apply_aux_feature_set(args, feature_set):
    reset_aux_feature_flags(args)
    args.use_aux_features = True
    if feature_set == 'demographics':
        args.aux_gender = True
        args.aux_bmi = True
    elif feature_set == 'image_stats':
        args.aux_skewness = True
        args.aux_intensity = True
        args.aux_clarity = True
    elif feature_set == 'all':
        args.aux_gender = True
        args.aux_bmi = True
        args.aux_skewness = True
        args.aux_intensity = True
        args.aux_clarity = True
    else:
        raise ValueError(f'未知辅助特征组合: {feature_set}')


def apply_usfm_profile_defaults(args, opt_args, profile_name):
    spec = PROFILE_SPECS[profile_name]

    args.model = 'usfm'
    args.pretrained_path = opt_args.pretrained_path
    args.pretrained = True
    args.image_size = 224
    args.loss = 'mae'
    args.optimizer = 'adamw'
    args.min_age = MIN_AGE_FIXED
    args.freeze_backbone = spec['freeze_backbone']
    args.ensemble = False
    args.use_ema = False
    args.no_save = True
    args.deterministic = False

    reset_aux_feature_flags(args)
    if spec['mode'] == 'multimodal':
        args.use_aux_features = True
    else:
        args.use_aux_features = False

    return spec


def suggest_usfm_trial_params(trial, args, profile_name):
    spec = PROFILE_SPECS[profile_name]

    args.lr = trial.suggest_float('lr', *spec['lr_range'], log=True)
    args.weight_decay = trial.suggest_float('weight_decay', *spec['weight_decay_range'], log=True)
    args.batch_size = trial.suggest_categorical('batch_size', spec['batch_size_choices'])
    args.dropout = trial.suggest_float('dropout', *spec['dropout_range'])
    args.scheduler = trial.suggest_categorical('scheduler', spec['scheduler_choices'])
    args.max_grad_norm = trial.suggest_categorical('max_grad_norm', spec['max_grad_norm_choices'])
    args.usfm_global_pool = trial.suggest_categorical('usfm_global_pool', spec['usfm_global_pool_choices'])

    if spec['mode'] == 'multimodal':
        aux_feature_set = trial.suggest_categorical('aux_feature_set', spec['aux_feature_sets'])
        apply_aux_feature_set(args, aux_feature_set)
        args.aux_hidden_dim = trial.suggest_categorical('aux_hidden_dim', spec['aux_hidden_dim_choices'])
    else:
        reset_aux_feature_flags(args)

    if args.scheduler == 'cosine':
        args.warmup_epochs = trial.suggest_int('warmup_epochs', *spec['warmup_epochs_range'])
        args.eta_min = trial.suggest_float('eta_min', *spec['eta_min_range'], log=True)
    elif args.scheduler == 'plateau':
        args.lr_patience = trial.suggest_int('lr_patience', *spec['lr_patience_range'])
        args.lr_factor = trial.suggest_float('lr_factor', *spec['lr_factor_range'])
        args.lr_min_lr = trial.suggest_float('lr_min_lr', *spec['lr_min_lr_range'], log=True)


def extract_effective_args(args):
    data = {}
    for key in TRACKED_TRAIN_KEYS:
        if hasattr(args, key):
            data[key] = getattr(args, key)
    return data


def build_objective(base_args, opt_args):
    def objective(trial):
        args = copy.deepcopy(base_args)
        args.epochs = opt_args.max_epochs
        args.patience = opt_args.max_epochs + 1
        args.no_save = True
        args.ensemble = False
        args.min_age = MIN_AGE_FIXED

        suggest_usfm_trial_params(trial, args, opt_args.profile)
        trial.set_user_attr('effective_train_args', extract_effective_args(args))

        def reporter(epoch, val_mae):
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                trial.set_user_attr('fail_reason', 'pruned')
                raise optuna.TrialPruned(f'Pruned at epoch {epoch}')

        try:
            result = train_module.train(args, reporter=reporter)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc).lower()
            if 'out of memory' in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trial.set_user_attr('fail_reason', 'oom')
                raise
            trial.set_user_attr('fail_reason', f'runtime_error: {str(exc)}')
            raise
        except Exception as exc:
            trial.set_user_attr('fail_reason', f'exception: {str(exc)}')
            raise

        if not result or 'best_val_mae' not in result:
            trial.set_user_attr('fail_reason', 'missing_metric')
            raise RuntimeError('训练未返回有效的 best_val_mae')

        best_val_mae = result['best_val_mae']
        if not torch.isfinite(torch.tensor(best_val_mae)):
            trial.set_user_attr('fail_reason', 'nan_or_inf')
            raise RuntimeError('best_val_mae 为 NaN/Inf')

        trial.set_user_attr('best_epoch', result.get('best_epoch'))
        trial.set_user_attr('train_time', result.get('train_time'))
        return best_val_mae

    return objective


def acquire_trial_slot(study, output_root, max_trials):
    budget_path = output_root / 'trial_budget.json'
    lock_path = output_root / 'trial_budget.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'w', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        issued = 0
        if budget_path.exists():
            with open(budget_path, 'r', encoding='utf-8') as f:
                try:
                    issued = int(json.load(f).get('issued', 0))
                except Exception:
                    issued = 0
        waiting_state = getattr(TrialState, 'WAITING', None)
        running_states = [TrialState.RUNNING]
        if waiting_state is not None:
            running_states.append(waiting_state)
        total_trials = len(study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, *running_states),
        ))
        issued = max(issued, total_trials)
        if issued >= max_trials:
            return False
        issued += 1
        with open(budget_path, 'w', encoding='utf-8') as f:
            json.dump({'issued': issued}, f)
        return True


def worker_loop(gpu_id, opt_args, train_args, storage_url, log_path):
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    parser = train_module.create_arg_parser()
    base_args = parser.parse_args(train_args)
    base_params = load_base_params(opt_args.base_params)
    apply_params(base_args, base_params)
    apply_usfm_profile_defaults(base_args, opt_args, opt_args.profile)

    objective = build_objective(base_args, opt_args)
    try:
        sampler = optuna.samplers.TPESampler(seed=base_args.seed, multivariate=True, group=True)
    except TypeError:
        sampler = optuna.samplers.TPESampler(seed=base_args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=opt_args.pruner_startup_trials,
        n_warmup_steps=opt_args.pruner_warmup_epochs,
    )
    study = optuna.create_study(
        study_name=opt_args.study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    start_time = time.time()
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'[worker {gpu_id}] start\n')

    while True:
        if opt_args.timeout and (time.time() - start_time) >= opt_args.timeout:
            break
        if not acquire_trial_slot(study, SEARCH_ROOT / opt_args.study_name, opt_args.n_trials):
            break
        study.optimize(objective, n_trials=1, catch=(Exception,))

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'[worker {gpu_id}] done\n')


def save_results(study, output_root):
    results = []
    for trial in study.trials:
        results.append({
            'number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
            'fail_reason': trial.user_attrs.get('fail_reason'),
        })
    with open(output_root / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    try:
        df = study.trials_dataframe(attrs=('number', 'value', 'state', 'params', 'user_attrs'))
        df['fail_reason'] = df['user_attrs'].apply(lambda x: x.get('fail_reason') if isinstance(x, dict) else None)
        df.to_csv(output_root / 'results.csv', index=False)
        ranked = df[df['state'] == 'COMPLETE'].sort_values('value')
        ranked.to_csv(output_root / 'results_ranked.csv', index=False)
    except Exception:
        pass


def summarize_param_effects(completed_trials):
    metrics = {}
    if not completed_trials:
        return metrics

    values = [t.value for t in completed_trials]
    params = [t.params for t in completed_trials]
    keys = sorted({k for p in params for k in p})

    for key in keys:
        col = [p.get(key) for p in params]
        if all(isinstance(x, (int, float)) for x in col):
            ranked = sorted(zip(col, values), key=lambda x: x[0])
            xs = [x for x, _ in ranked]
            ys = [y for _, y in ranked]
            n = len(xs)
            if n < 3 or max(xs) == min(xs):
                metrics[key] = {'type': 'numeric', 'spearman': 0.0, 'range': (min(xs), max(xs))}
                continue
            xranks = {v: i for i, v in enumerate(sorted(set(xs)))}
            yranks = {v: i for i, v in enumerate(sorted(set(ys)))}
            xranked = [xranks[v] for v in xs]
            yranked = [yranks[v] for v in ys]
            mean_x = sum(xranked) / n
            mean_y = sum(yranked) / n
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xranked, yranked))
            var_x = sum((x - mean_x) ** 2 for x in xranked)
            var_y = sum((y - mean_y) ** 2 for y in yranked)
            spearman = 0.0 if var_x == 0 or var_y == 0 else cov / (var_x ** 0.5 * var_y ** 0.5)
            metrics[key] = {'type': 'numeric', 'spearman': spearman, 'range': (min(xs), max(xs))}
        else:
            buckets = {}
            for v, y in zip(col, values):
                buckets.setdefault(v, []).append(y)
            means = {k: sum(v) / len(v) for k, v in buckets.items()}
            spread = max(means.values()) - min(means.values()) if means else 0.0
            metrics[key] = {'type': 'categorical', 'means': means, 'spread': spread}
    return metrics


def detect_high_perf_region(completed_trials, top_frac=0.1):
    if not completed_trials:
        return {}
    top_n = max(1, int(len(completed_trials) * top_frac))
    top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_n]
    params = [t.params for t in top_trials]
    region = {}
    keys = sorted({k for p in params for k in p})
    for key in keys:
        vals = [p.get(key) for p in params]
        if all(isinstance(x, (int, float)) for x in vals):
            region[key] = {'min': min(vals), 'max': max(vals)}
        else:
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            region[key] = {'top_categories': sorted(counts.items(), key=lambda x: x[1], reverse=True)}
    return region


def summarize_failure_bias(trials, states):
    filtered = [t for t in trials if t.state in states]
    if not filtered:
        return {}
    params = [t.params for t in filtered]
    keys = sorted({k for p in params for k in p})
    bias = {}
    for key in keys:
        vals = [p.get(key) for p in params]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        bias[key] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return bias


def save_summary(study, output_root, opt_args, base_params):
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    best = study.best_trial if completed else None
    param_effects = summarize_param_effects(completed)
    high_region = detect_high_perf_region(completed)
    fail_bias = summarize_failure_bias(study.trials, {TrialState.FAIL})
    prune_bias = summarize_failure_bias(study.trials, {TrialState.PRUNED})
    spec = PROFILE_SPECS[opt_args.profile]

    lines = [
        '# USFM Optuna 搜索总结',
        f'- profile: {opt_args.profile}',
        f'- description: {spec["description"]}',
        f'- pretrained_path: {opt_args.pretrained_path}',
        f'- min_age 固定为 {MIN_AGE_FIXED}',
        f'- 试验总数: {len(study.trials)}',
        f'- 完成: {len(completed)}',
        f'- 剪枝: {len(pruned)}',
        f'- 失败: {len(failed)}',
        f'- 搜索 epochs: {opt_args.max_epochs}',
        f'- 推荐 full-run epochs: {opt_args.final_epochs}',
        f'- 推荐 full-run patience: {opt_args.final_patience}',
    ]
    if best:
        lines.append(f'- 最佳 MAE: {best.value}')
        lines.append(f'- 最佳 trial: {best.number}')
        lines.append(f'- 最佳搜索参数: {json.dumps(best.params, ensure_ascii=False)}')
    if completed:
        strongest = sorted(
            [(k, abs(v.get('spearman', 0.0)), v) for k, v in param_effects.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        lines.append('## 参数影响（基于完成 trial）')
        for key, _, meta in strongest:
            if meta['type'] == 'numeric':
                lines.append(f'- {key}: spearman={meta["spearman"]:.3f} range={meta["range"]}')
            else:
                lines.append(f'- {key}: spread={meta["spread"]:.3f} means={meta["means"]}')
        lines.append('## 高性能区域（Top10% trial）')
        lines.append(json.dumps(high_region, ensure_ascii=False))
        lines.append('## 失败/剪枝集中趋势')
        lines.append(f'- fail_top: {json.dumps(fail_bias, ensure_ascii=False)}')
        lines.append(f'- pruned_top: {json.dumps(prune_bias, ensure_ascii=False)}')
        lines.append('## 可信度说明')
        lines.append('- 这里只是 short-run 排名，最终仍应对 Top 配置做 full-run + 多 seed 复核。')
    if base_params:
        lines.append(f'- base_params: {json.dumps(base_params, ensure_ascii=False)}')

    with open(output_root / 'summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def apply_trial_params_to_args(args, params, profile_name):
    profile = PROFILE_SPECS[profile_name]
    if profile['mode'] == 'multimodal':
        feature_set = params.get('aux_feature_set', 'all')
        apply_aux_feature_set(args, feature_set)
    else:
        reset_aux_feature_flags(args)

    for key, value in params.items():
        if key == 'aux_feature_set':
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def save_best_artifacts(study, output_root, opt_args, train_args):
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return

    parser = train_module.create_arg_parser()
    best_args = parser.parse_args(train_args)
    base_params = load_base_params(opt_args.base_params)
    apply_params(best_args, base_params)
    apply_usfm_profile_defaults(best_args, opt_args, opt_args.profile)
    apply_trial_params_to_args(best_args, study.best_trial.params, opt_args.profile)
    best_args.no_save = False
    best_args.epochs = opt_args.final_epochs
    best_args.patience = opt_args.final_patience

    effective_args = extract_effective_args(best_args)
    effective_args['epochs'] = best_args.epochs
    effective_args['patience'] = best_args.patience
    effective_args['profile'] = opt_args.profile
    effective_args['best_trial'] = study.best_trial.number
    effective_args['best_search_mae'] = study.best_trial.value

    with open(output_root / 'best_train_args.json', 'w', encoding='utf-8') as f:
        json.dump(effective_args, f, indent=2, ensure_ascii=False)

    command = train_module.generate_command_line(best_args)
    with open(output_root / 'best_command.sh', 'w', encoding='utf-8') as f:
        f.write('#!/bin/bash\n')
        f.write(f'# USFM profile: {opt_args.profile}\n')
        f.write(f'# Best short-run trial: {study.best_trial.number}\n')
        f.write(f'# Best short-run val MAE: {study.best_trial.value}\n\n')
        f.write(command)
        f.write('\n')
    os.chmod(output_root / 'best_command.sh', 0o775)


def save_profile_metadata(output_root, opt_args):
    spec = PROFILE_SPECS[opt_args.profile]
    metadata = {
        'profile': opt_args.profile,
        'description': spec['description'],
        'pretrained_path': opt_args.pretrained_path,
        'n_trials': opt_args.n_trials,
        'max_epochs': opt_args.max_epochs,
        'final_epochs': opt_args.final_epochs,
        'final_patience': opt_args.final_patience,
        'pruner_warmup_epochs': opt_args.pruner_warmup_epochs,
        'pruner_startup_trials': opt_args.pruner_startup_trials,
        'search_space': spec,
    }
    with open(output_root / 'profile.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    opt_args, train_args = parse_args()
    output_root = SEARCH_ROOT / opt_args.study_name
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / 'search.log'
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('[main] start\n')

    save_profile_metadata(output_root, opt_args)

    storage_path = Path(opt_args.storage) if opt_args.storage else output_root / 'study.sqlite3'
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = make_storage_url(storage_path.resolve())

    gpus = resolve_gpus(opt_args.gpus)
    if opt_args.n_workers is None:
        opt_args.n_workers = len(gpus)
    opt_args.n_workers = max(1, min(opt_args.n_workers, len(gpus)))

    if opt_args.worker:
        worker_loop(gpus[0], opt_args, train_args, storage_url, log_path)
        return

    if opt_args.n_workers == 1:
        worker_loop(gpus[0], opt_args, train_args, storage_url, log_path)
    else:
        processes = []
        for i in range(opt_args.n_workers):
            gpu_id = gpus[i]
            cmd = [
                sys.executable, str(__file__),
                '--study-name', opt_args.study_name,
                '--profile', opt_args.profile,
                '--pretrained-path', opt_args.pretrained_path,
                '--n-trials', str(opt_args.n_trials),
                '--max-epochs', str(opt_args.max_epochs),
                '--final-epochs', str(opt_args.final_epochs),
                '--final-patience', str(opt_args.final_patience),
                '--pruner-warmup-epochs', str(opt_args.pruner_warmup_epochs),
                '--pruner-startup-trials', str(opt_args.pruner_startup_trials),
                '--storage', str(storage_path),
                '--gpus', str(gpu_id),
                '--worker',
            ]
            if opt_args.base_params:
                cmd += ['--base-params', opt_args.base_params]
            if opt_args.timeout is not None:
                cmd += ['--timeout', str(opt_args.timeout)]
            cmd += train_args
            p = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
            processes.append(p)
        for p in processes:
            p.wait()

    study = optuna.load_study(study_name=opt_args.study_name, storage=storage_url)
    base_params = load_base_params(opt_args.base_params)
    save_results(study, output_root)
    save_summary(study, output_root, opt_args, base_params)
    save_best_artifacts(study, output_root, opt_args, train_args)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('[main] done\n')


if __name__ == '__main__':
    main()
