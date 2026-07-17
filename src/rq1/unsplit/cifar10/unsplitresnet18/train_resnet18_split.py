from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data import cifar10_loaders
from unsplit.models import build_split_resnet18


def device_auto():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def test_score(client, server, loader, split_layer, device):
    client.eval(); server.eval()
    correct = 0; total = 0; loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for images, labels in loader:
        images = images.to(device); labels = labels.to(device)
        pred = server(client(images, end=split_layer), start=split_layer + 1)
        loss_sum += float(criterion(pred, labels).item())
        correct += int(pred.argmax(1).eq(labels).sum().item())
        total += labels.size(0)
    return loss_sum / total, correct / total


def train_cut(args):
    device = device_auto()
    torch.manual_seed(args.seed)
    _, _, trainloader, testloader = cifar10_loaders(batch_size=args.batch_size, download=True)
    client, server, split_layer = build_split_resnet18(args.cut)
    client.to(device); server.to(device)

    # Official UnSplit uses Adam lr=0.001. Batch size changed to 128.
    client_opt = torch.optim.Adam(client.parameters(), lr=args.lr, amsgrad=True)
    server_opt = torch.optim.Adam(server.parameters(), lr=args.lr, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(args.checkpoint_dir) / f'cifar10_resnet18_{args.cut}'
    log_dir = Path(args.log_dir)
    metrics_dir = Path(args.metrics_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'cifar10_resnet18_{args.cut}.log'

    best_acc = 0.0
    history = []
    start = time.time()
    with log_path.open('w', encoding='utf-8') as log:
        def logline(msg):
            print(msg)
            log.write(msg + '\n'); log.flush()

        logline(f'starting cut={args.cut} epochs={args.epochs} batch_size={args.batch_size} device={device}')
        for epoch in range(1, args.epochs + 1):
            client.train(); server.train()
            total_loss = 0.0; correct = 0; total = 0
            for images, labels in tqdm(trainloader, desc=f'train {args.cut} epoch {epoch}', leave=False):
                images = images.to(device); labels = labels.to(device)
                client_opt.zero_grad(); server_opt.zero_grad()
                pred = server(client(images, end=split_layer), start=split_layer + 1)
                loss = criterion(pred, labels)
                loss.backward()
                client_opt.step(); server_opt.step()
                total_loss += float(loss.item()) * labels.size(0)
                correct += int(pred.argmax(1).eq(labels).sum().item())
                total += labels.size(0)
            train_loss = total_loss / total
            train_acc = correct / total
            test_loss, test_acc = test_score(client, server, testloader, split_layer, device)
            best_acc = max(best_acc, test_acc)
            record = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc, 'test_loss': test_loss, 'test_accuracy': test_acc, 'best_test_accuracy': best_acc}
            history.append(record)
            logline(f'epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f} best={best_acc:.4f}')
            ckpt = {'cut': args.cut, 'split_layer': split_layer, 'epoch': epoch, 'client_state_dict': client.state_dict(), 'server_state_dict': server.state_dict(), 'metrics': record}
            torch.save(ckpt, ckpt_dir / 'latest.pt')
            if test_acc >= best_acc:
                torch.save(ckpt, ckpt_dir / 'best.pt')
        summary = {'cut': args.cut, 'epochs': args.epochs, 'batch_size': args.batch_size, 'best_test_accuracy': best_acc, 'final_test_accuracy': history[-1]['test_accuracy'], 'runtime_sec': time.time() - start, 'history': history}
        (metrics_dir / f'cifar10_resnet18_{args.cut}_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cut', required=True, choices=['c1','c2','c3','c4'])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--log-dir', default='results/logs')
    p.add_argument('--metrics-dir', default='results/metrics')
    return p.parse_args()


if __name__ == '__main__':
    train_cut(parse_args())
