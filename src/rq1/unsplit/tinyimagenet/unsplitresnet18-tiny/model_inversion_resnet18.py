from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

try:
    import lpips
except ImportError:
    lpips = None

import unsplit.attacks as unsplit
from data import tinyimagenet_loaders
from unsplit.models import build_split_resnet18
from unsplit.util import get_random_indices, normalize


def device_auto():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def mse_metric(x, y):
    return float(torch.mean((x.cpu().clamp(0,1) - y.cpu().clamp(0,1)) ** 2).item())


def psnr_metric(mse):
    if mse <= 0: return float('inf')
    return float(-10.0 * np.log10(mse))


def ssim_metric(x, y):
    x_np = x.cpu().clamp(0,1).permute(1,2,0).numpy()
    y_np = y.cpu().clamp(0,1).permute(1,2,0).numpy()
    return float(structural_similarity(x_np, y_np, channel_axis=2, data_range=1.0))


def write_csv(rows, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def run_attack(args):
    device = device_auto()
    torch.manual_seed(args.seed)
    lambda_tv = args.lambda_tv
    if lambda_tv is None:
        lambda_tv = 0.1 if args.cut in {'c1', 'c2', 'c3'} else 1.0
    _, testset, _, _ = tinyimagenet_loaders(root=args.data_root, batch_size=args.batch_size, download=False)
    indices = get_random_indices(testset, args.n_attack, args.seed)
    loader = DataLoader(Subset(testset, indices), batch_size=1, shuffle=False)

    victim_client, _, split_layer = build_split_resnet18(args.cut, num_classes=200)
    clone, _, _ = build_split_resnet18(args.cut, num_classes=200)
    ckpt = torch.load(Path(args.checkpoint_dir) / f'tinyimagenet_resnet18_{args.cut}' / 'best.pt', map_location=device)
    victim_client.load_state_dict(ckpt['client_state_dict'])
    victim_client.to(device).eval()
    clone.to(device).eval()
    for p in victim_client.parameters(): p.requires_grad_(False)

    # Copy all BN parameters from victim to clone — ResNet-18 has BatchNorm;
    # without this the clone and victim operate in different activation spaces.
    victim_bns = {n: m for n, m in victim_client.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}
    clone_bns  = {n: m for n, m in clone.named_modules()         if isinstance(m, torch.nn.BatchNorm2d)}
    for name in victim_bns:
        clone_bns[name].running_mean.copy_(victim_bns[name].running_mean)
        clone_bns[name].running_var.copy_(victim_bns[name].running_var)
        clone_bns[name].num_batches_tracked.copy_(victim_bns[name].num_batches_tracked)
        clone_bns[name].weight.data.copy_(victim_bns[name].weight.data)
        clone_bns[name].bias.data.copy_(victim_bns[name].bias.data)
        clone_bns[name].weight.requires_grad_(False)
        clone_bns[name].bias.requires_grad_(False)

    lpips_model = None
    if not args.no_lpips:
        if lpips is None: raise ImportError('Install lpips or pass --no-lpips')
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    out = Path(args.output_dir) / f'tinyimagenet_resnet18_{args.cut}'
    orig_dir = out / 'originals'; recon_dir = out / 'reconstructions'
    orig_dir.mkdir(parents=True, exist_ok=True); recon_dir.mkdir(parents=True, exist_ok=True)
    rows = []; origs = []; recons = []
    start = time.time()
    for i, (target, label) in enumerate(tqdm(loader, desc=f'UnSplit {args.cut}')):
        sample_idx = indices[i]
        target = target.to(device)
        with torch.no_grad():
            client_out = victim_client(target, end=split_layer)
        result = unsplit.model_inversion_stealing(clone, split_layer, client_out, target.size(), lambda_tv=lambda_tv, lambda_l2=args.lambda_l2, main_iters=args.main_iters, input_iters=args.input_iters, model_iters=args.model_iters)
        result = normalize(result) if args.normalize_output else result.clamp(0,1)
        orig = target.detach().cpu().squeeze(0).clamp(0,1)
        recon = result.detach().cpu().squeeze(0).clamp(0,1)
        origs.append(orig); recons.append(recon)
        save_image(orig, orig_dir / f'sample_{sample_idx:05d}_label_{int(label)}.png')
        save_image(recon, recon_dir / f'sample_{sample_idx:05d}_label_{int(label)}.png')
        mse = mse_metric(orig, recon)
        row = {'cut': args.cut, 'sample_index': sample_idx, 'label': int(label), 'mse': mse, 'psnr': psnr_metric(mse), 'ssim': ssim_metric(orig, recon)}
        if lpips_model is not None:
            with torch.no_grad(): row['lpips'] = float(lpips_model(target * 2 - 1, result.to(device) * 2 - 1).view(-1).item())
        rows.append(row)
    summary = {'cut': args.cut, 'n_attack': args.n_attack, 'main_iters': args.main_iters, 'input_iters': args.input_iters, 'model_iters': args.model_iters, 'lambda_tv': lambda_tv, 'lambda_l2': args.lambda_l2, 'runtime_sec': time.time() - start}
    for name in [n for n in ('mse','psnr','ssim','lpips') if n in rows[0]]:
        vals = np.array([r[name] for r in rows], dtype=float)
        summary[f'{name}_mean'] = float(vals.mean()); summary[f'{name}_std'] = float(vals.std()); summary[f'{name}_min'] = float(vals.min()); summary[f'{name}_max'] = float(vals.max())
    summary['asr_ssim_0.90'] = float(np.mean([r['ssim'] >= 0.90 for r in rows]))
    write_csv(rows, out / 'metrics.csv')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'attack_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    save_image(make_grid(torch.stack(origs), nrow=min(args.n_attack, 8)), out / 'original_grid.png')
    save_image(make_grid(torch.stack(recons), nrow=min(args.n_attack, 8)), out / 'reconstruction_grid.png')
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cut', required=True, choices=['c1','c2','c3','c4'])
    p.add_argument('--n-attack', type=int, default=20)
    p.add_argument('--main-iters', type=int, default=1000)
    p.add_argument('--input-iters', type=int, default=100)
    p.add_argument('--model-iters', type=int, default=100)
    p.add_argument('--lambda-tv', type=float, default=None)
    p.add_argument('--lambda-l2', type=float, default=0.0)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--output-dir', default='results/reconstructions')
    p.add_argument('--data-root', default='data/tiny-imagenet-200')
    p.add_argument('--normalize-output', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--no-lpips', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    run_attack(parse_args())
