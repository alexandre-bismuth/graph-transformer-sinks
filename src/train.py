import hashlib
import json
import os
import random
import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, f1_score


def _build_real_node_mask(batch):
    mask = torch.ones(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
    _, counts = batch.batch.unique(return_counts=True)
    offset = 0
    for c in counts:
        mask[offset + c - 1] = False
        offset += c
    return mask


def build_criterion(task):
    if task == 'regression':
        return nn.L1Loss()
    elif task == 'multilabel_classification':
        return nn.BCEWithLogitsLoss()
    elif task == 'node_classification':
        return nn.CrossEntropyLoss()
    elif task == 'graph_classification':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task: {task}")


def build_scheduler(optimizer, config):
    epochs = config['training']['epochs']
    warmup = config['training']['warmup_epochs']

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup]
    )
    return scheduler


def train_epoch(model, loader, optimizer, criterion, device, task, max_grad_norm=1.0,
                entropy_coef=0.0):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, collect_diagnostics=False)

        if task == 'regression':
            target = batch.y.float()
        elif task == 'multilabel_classification':
            target = batch.y.float()
        elif task == 'node_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)
            if hasattr(batch, 'vnode_idx'):
                pred = pred[_build_real_node_mask(batch)]
        elif task == 'graph_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)

        loss = criterion(pred, target)
        if entropy_coef and hasattr(model, 'pop_attn_entropy'):
            ent = model.pop_attn_entropy()
            if ent is not None:
                # sink forcing: positive coef penalises high attention entropy,
                # pushing rows toward concentration on few nodes.
                loss = loss + entropy_coef * ent
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        n = batch.num_graphs if task not in ('node_classification',) else batch.y.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / total_samples


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, collect_diagnostics=False)

        if task == 'regression':
            target = batch.y.float()
        elif task == 'multilabel_classification':
            target = batch.y.float()
        elif task == 'node_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)
            if hasattr(batch, 'vnode_idx'):
                pred = pred[_build_real_node_mask(batch)]
        elif task == 'graph_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)

        loss = criterion(pred, target)

        n = batch.num_graphs if task not in ('node_classification',) else target.size(0)
        total_loss += loss.item() * n
        total_samples += n

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if task == 'regression':
        metric_val = avg_loss
        metric_name = 'mae'
    elif task == 'multilabel_classification':
        probs = torch.sigmoid(all_preds).numpy()
        targets_np = all_targets.numpy()
        metric_val = average_precision_score(targets_np, probs, average='macro')
        metric_name = 'ap'
    elif task == 'node_classification':
        pred_labels = all_preds.argmax(dim=-1).numpy()
        targets_np = all_targets.numpy()
        metric_val = f1_score(targets_np, pred_labels, average='macro', zero_division=0)
        metric_name = 'f1'
    elif task == 'graph_classification':
        pred_labels = all_preds.argmax(dim=-1).numpy()
        targets_np = all_targets.numpy()
        metric_val = (pred_labels == targets_np).mean()
        metric_name = 'accuracy'

    return avg_loss, metric_val, metric_name


def is_better(metric_val, best_val, task):
    if task == 'regression':
        return metric_val < best_val
    else:
        return metric_val > best_val


def _config_fingerprint(config):
    """Identity of the run, so a stale ckpt.pt from a different config is never resumed."""
    return hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _save_ckpt(save_dir, epoch, epochs, model, optimizer, scheduler, best_val_metric,
               train_losses, val_losses, fingerprint):
    """Resumable snapshot. Written tmp-then-rename so a kill mid-write cannot
    corrupt it, plus a single-file breadcrumb (never discover checkpoints by listing)."""
    path = os.path.join(save_dir, 'ckpt.pt')
    payload = {
        'epoch': epoch, 'epochs': epochs, 'fingerprint': fingerprint,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'best_val_metric': best_val_metric,
        'train_losses': train_losses, 'val_losses': val_losses,
        'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                'python': random.getstate(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    torch.save(payload, path + '.tmp')
    os.replace(path + '.tmp', path)
    with open(os.path.join(save_dir, 'latest_checkpoint.json'), 'w') as f:
        json.dump({'path': path, 'epoch': epoch, 'epochs': epochs, 'complete': False}, f)


def _atomic_save(obj, path):
    """tmp-then-rename, so a copy taken mid-write is never half a file."""
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def _write_curve(save_dir, train_losses, val_losses):
    with open(os.path.join(save_dir, 'train_curve.json'), 'w') as f:
        json.dump({'train_loss': [float(x) for x in train_losses],
                   'val_loss': [float(x) for x in val_losses]}, f)


def _check_finite(save_dir, epoch, train_loss, val_loss, train_losses, val_losses):
    """A diverged run must fail (non-zero exit), not finish with a frozen best_model.pt."""
    if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
        _write_curve(save_dir, train_losses, val_losses)
        raise FloatingPointError(f'non-finite loss at epoch {epoch}: '
                                 f'train={train_loss} val={val_loss}')


def _try_resume(save_dir, device, model, optimizer, scheduler, fingerprint, epochs):
    """Return (start_epoch, best_val_metric, train_losses, val_losses) — 1/None/[]/[] if fresh."""
    path = os.path.join(save_dir, 'ckpt.pt')
    if not os.path.exists(path):
        return 1, None, [], []
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get('fingerprint') != fingerprint:
        print(f"[resume] ckpt.pt is from a different config — ignoring it and starting fresh")
        return 1, None, [], []
    if state['epoch'] >= epochs:
        print(f"[resume] ckpt.pt already at epoch {state['epoch']}/{epochs} — starting fresh")
        return 1, None, [], []
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    try:
        torch.set_rng_state(state['rng']['torch'].cpu().to(torch.uint8))
        np.random.set_state(state['rng']['numpy'])
        random.setstate(state['rng']['python'])
        if state['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in state['rng']['cuda']])
    except Exception as e:                      # RNG is a nicety; never fail a resume over it
        print(f"[resume] could not restore RNG state ({e}) — continuing")
    print(f"[resume] continuing from epoch {state['epoch'] + 1}/{epochs} "
          f"(best so far {state['best_val_metric']:.4f})")
    return state['epoch'] + 1, state['best_val_metric'], state['train_losses'], state['val_losses']


def train_model(model, train_loader, val_loader, config, device, task, save_dir='outputs', use_wandb=True):
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    wd = config['training']['weight_decay']
    grad_clip = config['training']['grad_clip']

    criterion = build_criterion(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = build_scheduler(optimizer, config)

    entropy_coef = float(config['training'].get('attn_entropy_coef', 0.0) or 0.0)
    if entropy_coef and hasattr(model, 'layers'):
        for layer in model.layers:
            if hasattr(layer, 'mha'):
                layer.mha.entropy_track = True

    os.makedirs(save_dir, exist_ok=True)

    best_val_metric = float('inf') if task == 'regression' else 0.0
    train_losses = []
    val_losses = []

    # Snapshot every ckpt_every_epochs so a killed run resumes instead of
    # restarting. 0 disables. At ~24 s/epoch the default bounds lost work at ~20 min.
    ckpt_every = int(config['training'].get('ckpt_every_epochs', 50) or 0)
    fingerprint = _config_fingerprint(config)
    start_epoch = 1
    if ckpt_every:
        start_epoch, resumed_best, resumed_tr, resumed_val = _try_resume(
            save_dir, device, model, optimizer, scheduler, fingerprint, epochs)
        if resumed_best is not None:
            best_val_metric, train_losses, val_losses = resumed_best, resumed_tr, resumed_val

    pbar = tqdm(range(start_epoch, epochs + 1), desc='Training', initial=start_epoch - 1,
                total=epochs)

    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task,
                                 grad_clip, entropy_coef=entropy_coef)
        val_loss, val_metric, metric_name = eval_epoch(model, val_loader, criterion, device, task)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        _check_finite(save_dir, epoch, train_loss, val_loss, train_losses, val_losses)

        current_lr = optimizer.param_groups[0]['lr']

        if is_better(val_metric, best_val_metric, task):
            best_val_metric = val_metric
            _atomic_save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            f'val_{metric_name}': f'{val_metric:.4f}',
            f'best_{metric_name}': f'{best_val_metric:.4f}',
        })

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                f'val/{metric_name}': val_metric,
                'lr': current_lr,
                f'best/{metric_name}': best_val_metric,
            })

        if ckpt_every and (epoch % ckpt_every == 0):
            _save_ckpt(save_dir, epoch, epochs, model, optimizer, scheduler,
                       best_val_metric, train_losses, val_losses, fingerprint)

    _atomic_save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    _write_curve(save_dir, train_losses, val_losses)
    if ckpt_every:                      # run finished: drop the snapshot, mark the breadcrumb
        for f in ('ckpt.pt', 'ckpt.pt.tmp'):
            try:
                os.remove(os.path.join(save_dir, f))
            except OSError:
                pass
        with open(os.path.join(save_dir, 'latest_checkpoint.json'), 'w') as f:
            json.dump({'path': os.path.join(save_dir, 'final_model.pt'),
                       'epoch': epochs, 'epochs': epochs, 'complete': True}, f)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt'), weights_only=True))

    return {
        'best_val_metric': best_val_metric,
        'metric_name': metric_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
