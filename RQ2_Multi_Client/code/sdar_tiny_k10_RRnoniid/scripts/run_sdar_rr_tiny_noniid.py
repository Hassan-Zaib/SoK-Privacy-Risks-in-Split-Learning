import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import data
from sdar.sdar_hetero import SDARHeteroAttacker
from util.util import load_config


def reconstruction_metrics(x, x_recon):
    mse_per = np.mean((x - x_recon) ** 2, axis=(1, 2, 3))
    psnr_per = np.where(mse_per <= 0, np.inf, -10.0 * np.log10(mse_per + 1e-12))
    metrics = {
        "n": int(len(x)),
        "mse_mean": float(np.mean(mse_per)),
        "mse_std": float(np.std(mse_per)),
        "mse_min": float(np.min(mse_per)),
        "mse_max": float(np.max(mse_per)),
        "psnr_mean": float(np.mean(psnr_per)),
        "psnr_std": float(np.std(psnr_per)),
        "psnr_min": float(np.min(psnr_per)),
        "psnr_max": float(np.max(psnr_per)),
    }
    try:
        from skimage.metrics import structural_similarity as ssim

        vals = [ssim(x[i], x_recon[i], channel_axis=2, data_range=1.0) for i in range(len(x))]
        vals = np.asarray(vals, dtype=np.float32)
        metrics.update(
            {
                "ssim_mean": float(np.mean(vals)),
                "ssim_std": float(np.std(vals)),
                "ssim_min": float(np.min(vals)),
                "ssim_max": float(np.max(vals)),
                "asr_ssim_0.90": float(np.mean(vals >= 0.90)),
            }
        )
    except Exception as exc:
        metrics["ssim_error"] = repr(exc)
    try:
        import lpips
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = lpips.LPIPS(net="alex").to(device).eval()
        xt = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device) * 2 - 1
        rt = torch.from_numpy(x_recon).permute(0, 3, 1, 2).float().to(device) * 2 - 1
        with torch.no_grad():
            vals = loss_fn(xt, rt).view(-1).detach().cpu().numpy()
        metrics.update(
            {
                "lpips_mean": float(np.mean(vals)),
                "lpips_std": float(np.std(vals)),
                "lpips_min": float(np.min(vals)),
                "lpips_max": float(np.max(vals)),
            }
        )
    except Exception as exc:
        metrics["lpips_error"] = repr(exc)
    return metrics


def save_grid(rows, names, out_file, max_images=20):
    import matplotlib.pyplot as plt

    n = min(max_images, min(len(row) for row in rows))
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.05, len(rows) * 1.15))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    for r, row in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(np.clip(row[c], 0, 1))
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(names[r], fontsize=9, rotation=0, labelpad=28, va="center")
    plt.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def class_map(num_classes, num_clients):
    mapping = {}
    for cid in range(num_clients):
        mapping[str(cid)] = [cls for cls in range(num_classes) if cls % num_clients == cid]
    return mapping


def collect_client_recons(attacker, client_ds_list, batch_size, eval_batches):
    per_client = {}
    all_orig, all_recon, all_labels = [], [], []
    for cid, ds in enumerate(client_ds_list):
        originals, recons, labels = [], [], []
        for x, y in ds.batch(batch_size, drop_remainder=True).take(eval_batches):
            x_recon, _ = attacker.attack(x, y)
            originals.append(x.numpy())
            recons.append(x_recon.numpy())
            labels.append(y.numpy())
        x_np = np.concatenate(originals, axis=0)
        r_np = np.concatenate(recons, axis=0)
        y_np = np.concatenate(labels, axis=0)
        per_client[str(cid)] = {
            "originals": x_np,
            "reconstructions": r_np,
            "labels": y_np,
            "metrics": reconstruction_metrics(x_np, r_np),
        }
        all_orig.append(x_np)
        all_recon.append(r_np)
        all_labels.append(y_np)
    return per_client, np.concatenate(all_orig), np.concatenate(all_recon), np.concatenate(all_labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-style SDAR Tiny ImageNet round-robin non-IID multi-client run.")
    parser.add_argument("--dataset", default="tinyimagenet", choices=["tinyimagenet"])
    parser.add_argument("--model", default="resnet18", choices=["resnet18"])
    parser.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-iter", type=int, default=20000)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--output-dir", default="results/sdar_rr_noniid")
    parser.add_argument("--print-to-stdout", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    num_classes = 200
    run_name = f"{args.dataset}_{args.model}_{args.cut}_K{args.num_clients}_noniid_rr_run{args.run}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_fp = None
    if not args.print_to_stdout:
        log_fp = open(out_dir / "run.log", "w")
        sys.stdout = log_fp

    print(f"run_name={run_name}")
    print("mode=round_robin partition=non-iid")
    print(f"class_map={class_map(num_classes, args.num_clients)}")

    client_ds_list, server_ds_list = data.load_dataset_with_hetero_clients(args.dataset, args.num_clients)
    attacker = SDARHeteroAttacker(
        client_ds_list,
        server_ds_list,
        num_class=num_classes,
        batch_size=args.batch_size,
    )
    config = load_config("sdar", args.model, args.dataset, u_shape=False)
    print(f"config={config}")

    start = time.time()
    history = attacker.run(
        level=args.cut,
        num_iter=args.num_iter,
        config=config,
        u_shape=False,
        conditional=True,
        model_type=args.model,
        verbose_freq=100,
    )
    train_attack_runtime = time.time() - start

    comm_start = time.time()
    sample_x, _ = next(iter(client_ds_list[0].batch(args.batch_size, drop_remainder=True)))
    z = attacker.f(sample_x, training=False)
    smashed_shape = [int(v) for v in z.shape]
    bytes_per_sample = int(np.prod(smashed_shape[1:]) * 4)
    bytes_per_batch = int(np.prod(smashed_shape) * 4)
    rounds_per_client = {
        str(cid): int(args.num_iter // args.num_clients + (1 if cid < args.num_iter % args.num_clients else 0))
        for cid in range(args.num_clients)
    }
    communication = {
        "dtype": "float32",
        "smashed_shape": smashed_shape,
        "bytes_per_sample": bytes_per_sample,
        "bytes_per_batch": bytes_per_batch,
        "mb_per_batch": bytes_per_batch / (1024**2),
        "total_rounds": args.num_iter,
        "rounds_per_client": rounds_per_client,
        "total_smashed_bytes": int(bytes_per_batch * args.num_iter),
        "total_smashed_mb": float(bytes_per_batch * args.num_iter / (1024**2)),
    }

    eval_start = time.time()
    per_client, originals, recons, labels = collect_client_recons(
        attacker, client_ds_list, args.batch_size, args.eval_batches
    )
    evaluation_runtime = time.time() - eval_start
    global_metrics = reconstruction_metrics(originals, recons)

    client_metrics = {cid: payload["metrics"] for cid, payload in per_client.items()}
    ssim_means = np.asarray([m.get("ssim_mean", np.nan) for m in client_metrics.values()], dtype=np.float32)
    mse_means = np.asarray([m["mse_mean"] for m in client_metrics.values()], dtype=np.float32)
    lpips_means = np.asarray([m.get("lpips_mean", np.nan) for m in client_metrics.values()], dtype=np.float32)
    timing = {
        "train_attack_runtime_sec": float(train_attack_runtime),
        "evaluation_runtime_sec": float(evaluation_runtime),
        "communication_summary_runtime_sec": float(time.time() - comm_start),
        "total_runtime_sec": float(train_attack_runtime + evaluation_runtime),
        "avg_round_time_sec": float(train_attack_runtime / args.num_iter),
    }
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "cut": args.cut,
        "attack": "sdar",
        "num_clients": args.num_clients,
        "partition": "noniid_label_modulo",
        "orchestration": "round_robin",
        "classes_per_client": num_classes // args.num_clients,
        "class_map": class_map(num_classes, args.num_clients),
        "batch_size": args.batch_size,
        "num_iter": args.num_iter,
        "eval_batches_per_client": args.eval_batches,
        "n_eval_per_client": args.batch_size * args.eval_batches,
        "config": config,
        "fg_acc_last100": float(np.mean(history["fg_acc"][-100:])),
        "fg_loss_last100": float(np.mean(history["fg_loss"][-100:])),
        **global_metrics,
        "best_client_ssim": float(np.nanmax(ssim_means)),
        "worst_client_ssim": float(np.nanmin(ssim_means)),
        "ssim_std_across_clients": float(np.nanstd(ssim_means)),
        "mse_std_across_clients": float(np.nanstd(mse_means)),
        "lpips_std_across_clients": float(np.nanstd(lpips_means)),
        **{f"comm_{k}": v for k, v in communication.items() if k not in {"rounds_per_client"}},
        **timing,
    }

    np.save(out_dir / "history.npy", history)
    np.save(out_dir / "originals.npy", originals)
    np.save(out_dir / "reconstructions.npy", recons)
    np.save(out_dir / "labels.npy", labels)
    for cid, payload in per_client.items():
        np.save(out_dir / f"client_{cid}_originals.npy", payload["originals"])
        np.save(out_dir / f"client_{cid}_reconstructions.npy", payload["reconstructions"])
        save_grid(
            [payload["originals"], payload["reconstructions"]],
            [f"client {cid} orig", f"client {cid} recon"],
            out_dir / f"client_{cid}_grid.png",
        )
    save_grid([originals, recons], ["Original", "Recon"], out_dir / "original_recon_grid.png")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_client_metrics.json", "w") as f:
        json.dump(client_metrics, f, indent=2)
    with open(out_dir / "communication.json", "w") as f:
        json.dump(communication, f, indent=2)
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print("saved:", out_dir)
    print(json.dumps(summary, indent=2))
    if log_fp is not None:
        log_fp.close()


if __name__ == "__main__":
    main()
