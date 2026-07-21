from __future__ import annotations

import argparse
import json
import random
import time
from itertools import cycle
from pathlib import Path

import torch
from torch import autograd, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data import cifar100_attack_subset, cifar100_private_aux_loaders
from metrics import image_metrics
from models.attack_models import ClassifierHead, FeatureCritic, InversionDecoder
from models.resnet18 import build_front_resnet18, build_ushaped_resnet18


def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_primary(front, server, head, loader, test_loader, args, device, log):
    params = list(front.parameters()) + list(server.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr_model)
    criterion = nn.CrossEntropyLoss()
    best = 0.0
    history = []
    for epoch in range(1, args.main_epochs + 1):
        front.train(); server.train(); head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(loader, desc=f"main {args.cut} epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(server(front(x)))
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * y.size(0)
            correct += int(logits.argmax(1).eq(y).sum().item())
            total += y.size(0)
        test_loss, test_acc = evaluate_primary(front, server, head, test_loader, device)
        train_loss = total_loss / total
        train_acc = correct / total
        best = max(best, test_acc)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_test_accuracy": best,
        }
        history.append(row)
        log(f"main epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f} best={best:.4f}")
    return history


@torch.no_grad()
def evaluate_primary(front, server, head, loader, device):
    front.eval(); server.eval(); head.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = head(server(front(x)))
        loss_sum += float(criterion(logits, y).item())
        correct += int(logits.argmax(1).eq(y).sum().item())
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def collect_smashed(front, loader, device, max_batches: int | None):
    front.eval()
    zs = []
    for i, (x, _) in enumerate(tqdm(loader, desc="collect smashed", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        zs.append(front(x.to(device)).detach().cpu())
    return torch.cat(zs, dim=0)


def gradient_penalty(critic, real, fake, device):
    batch = min(real.size(0), fake.size(0))
    real = real[:batch]
    fake = fake[:batch]
    eps = torch.rand(batch, 1, 1, 1, device=device)
    mixed = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    score = critic(mixed)
    grad = autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grad.view(batch, -1).norm(2, dim=1) - 1.0) ** 2).mean()


def train_shadow(shadow_front, shadow_head, server, critic, aux_loader, z_loader, args, device, log):
    for p in server.parameters():
        p.requires_grad_(False)
    server.eval()
    ce = nn.CrossEntropyLoss()
    opt_shadow = torch.optim.Adam(
        list(shadow_front.parameters()) + list(shadow_head.parameters()),
        lr=args.lr_model,
        betas=(0.5, 0.999),
    )
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_discriminator, betas=(0.5, 0.999))
    z_iter = cycle(z_loader)

    for epoch in range(1, args.shadow_epochs + 1):
        shadow_front.train(); shadow_head.train(); critic.train()
        ce_sum = 0.0
        gan_sum = 0.0
        d_sum = 0.0
        total = 0
        for x_aux, y_aux in tqdm(aux_loader, desc=f"shadow {args.cut} epoch {epoch}", leave=False):
            x_aux, y_aux = x_aux.to(device), y_aux.to(device)
            z_real = next(z_iter)[0].to(device)

            with torch.no_grad():
                z_fake_detached = shadow_front(x_aux).detach()
            opt_critic.zero_grad(set_to_none=True)
            d_loss = critic(z_fake_detached).mean() - critic(z_real).mean()
            d_loss = d_loss + args.gp_weight * gradient_penalty(critic, z_real, z_fake_detached, device)
            d_loss.backward()
            opt_critic.step()

            opt_shadow.zero_grad(set_to_none=True)
            z_fake = shadow_front(x_aux)
            logits = shadow_head(server(z_fake))
            loss_ce = ce(logits, y_aux)
            loss_gan = -critic(z_fake).mean()
            loss = loss_ce + args.lambda_gan * loss_gan
            loss.backward()
            opt_shadow.step()

            ce_sum += float(loss_ce.item()) * y_aux.size(0)
            gan_sum += float(loss_gan.item()) * y_aux.size(0)
            d_sum += float(d_loss.item()) * y_aux.size(0)
            total += y_aux.size(0)
        log(f"shadow epoch={epoch} ce={ce_sum/total:.4f} gan={gan_sum/total:.4f} d={d_sum/total:.4f}")


def train_inverse(decoder, shadow_front, aux_loader, args, device, log):
    for p in shadow_front.parameters():
        p.requires_grad_(False)
    shadow_front.eval()
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr_model)
    mse = nn.MSELoss()
    for epoch in range(1, args.inverse_epochs + 1):
        decoder.train()
        loss_sum = 0.0
        total = 0
        for x_aux, _ in tqdm(aux_loader, desc=f"inverse {args.cut} epoch {epoch}", leave=False):
            x_aux = x_aux.to(device)
            with torch.no_grad():
                z_aux = shadow_front(x_aux)
            opt.zero_grad(set_to_none=True)
            recon = decoder(z_aux)
            loss = mse(recon, x_aux)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x_aux.size(0)
            total += x_aux.size(0)
        log(f"inverse epoch={epoch} mse={loss_sum/total:.6f}")


@torch.no_grad()
def reconstruct(front, decoder, attack_loader, device):
    front.eval(); decoder.eval()
    originals = []
    recons = []
    labels = []
    for x, y in attack_loader:
        x = x.to(device)
        z = front(x)
        recon = decoder(z).clamp(0, 1)
        originals.append(x.detach().cpu())
        recons.append(recon.detach().cpu())
        labels.append(y)
    return torch.cat(originals), torch.cat(recons), torch.cat(labels)


def run(args):
    set_seed(args.seed)
    device = device_auto()
    run_name = f"cifar100_resnet18_{args.cut}"
    out_dir = Path(args.output_dir) / run_name
    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    for path in (out_dir, log_dir, ckpt_dir):
        path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        def log(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        log(
            f"starting MAERA run={run_name} device={device} batch={args.batch_size} "
            f"test_aux_fraction={args.test_aux_fraction} private_aux_fraction={args.private_aux_fraction}"
        )
        _, _, private_loader, aux_loader, test_loader, prior_info = cifar100_private_aux_loaders(
            batch_size=args.batch_size,
            test_aux_fraction=args.test_aux_fraction,
            private_aux_fraction=args.private_aux_fraction,
            seed=args.seed,
            download=True,
            num_workers=args.num_workers,
        )
        log(f"minimum prior info={prior_info}")

        front, server, head, cut_def = build_ushaped_resnet18(args.cut, num_classes=100)
        front.to(device); server.to(device); head.to(device)
        start = time.time()
        history = train_primary(front, server, head, private_loader, test_loader, args, device, log)
        z_real = collect_smashed(front, private_loader, device, args.max_smashed_batches)
        z_loader = DataLoader(TensorDataset(z_real), batch_size=args.batch_size, shuffle=True, num_workers=0)
        log(f"stored smashed shape={tuple(z_real.shape)}")

        shadow_front, _ = build_front_resnet18(args.cut)
        shadow_head = ClassifierHead(512, num_classes=100)
        critic = FeatureCritic(cut_def.z_channels, cut_def.z_spatial)
        decoder = InversionDecoder(cut_def.z_channels, cut_def.z_spatial)
        shadow_front.to(device); shadow_head.to(device); critic.to(device); decoder.to(device)

        train_shadow(shadow_front, shadow_head, server, critic, aux_loader, z_loader, args, device, log)
        train_inverse(decoder, shadow_front, aux_loader, args, device, log)

        attack_set = cifar100_attack_subset(n_attack=args.n_attack, download=True)
        attack_loader = DataLoader(attack_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        originals, recons, labels = reconstruct(front, decoder, attack_loader, device)
        originals_dev = originals.to(device)
        recons_dev = recons.to(device)
        metrics = image_metrics(originals_dev, recons_dev, use_lpips=not args.no_lpips)
        metrics.update(
            {
                "run_name": run_name,
                "cut": args.cut,
                "dataset": "cifar100",
                "num_classes": 100,
                "batch_size": args.batch_size,
                "main_epochs": args.main_epochs,
                "shadow_epochs": args.shadow_epochs,
                "inverse_epochs": args.inverse_epochs,
                "lambda_gan": args.lambda_gan,
                "gp_weight": args.gp_weight,
                "lr_model": args.lr_model,
                "lr_discriminator": args.lr_discriminator,
                "test_aux_fraction": args.test_aux_fraction,
                "private_aux_fraction": args.private_aux_fraction,
                **prior_info,
                "n_attack": int(originals.size(0)),
                "runtime_sec": time.time() - start,
                "primary_best_test_accuracy": history[-1]["best_test_accuracy"],
                "primary_final_test_accuracy": history[-1]["test_accuracy"],
            }
        )

        save_image(make_grid(originals, nrow=min(16, originals.size(0))), out_dir / "original_grid.png")
        save_image(make_grid(recons, nrow=min(16, recons.size(0))), out_dir / "reconstruction_grid.png")
        for i in range(originals.size(0)):
            stem = f"sample_{i:05d}_label_{int(labels[i])}"
            save_image(originals[i], out_dir / f"{stem}_orig.png")
            save_image(recons[i], out_dir / f"{stem}_recon.png")

        torch.save(
            {
                "cut": args.cut,
                "front_state_dict": front.state_dict(),
                "server_state_dict": server.state_dict(),
                "head_state_dict": head.state_dict(),
                "shadow_front_state_dict": shadow_front.state_dict(),
                "shadow_head_state_dict": shadow_head.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "metrics": metrics,
                "history": history,
            },
            ckpt_dir / "maera.pt",
        )
        (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        log(json.dumps(metrics, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--main-epochs", type=int, default=100)
    p.add_argument("--shadow-epochs", type=int, default=200)
    p.add_argument("--inverse-epochs", type=int, default=200)
    p.add_argument("--lr-model", type=float, default=0.001)
    p.add_argument("--lr-discriminator", type=float, default=0.0005)
    p.add_argument("--lambda-gan", type=float, default=0.5)
    p.add_argument("--gp-weight", type=float, default=10.0)
    p.add_argument("--test-aux-fraction", type=float, default=0.01)
    p.add_argument("--private-aux-fraction", type=float, default=0.01)
    p.add_argument("--max-smashed-batches", type=int, default=None)
    p.add_argument("--n-attack", type=int, default=512)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="results/maera")
    p.add_argument("--log-dir", default="results/logs")
    p.add_argument("--no-lpips", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
