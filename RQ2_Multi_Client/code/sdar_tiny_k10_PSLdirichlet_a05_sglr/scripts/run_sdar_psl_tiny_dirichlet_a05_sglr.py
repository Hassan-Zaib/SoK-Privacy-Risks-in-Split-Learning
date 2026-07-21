import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import data
from sdar.sdar_psl import SDARPSLHeteroAttacker
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


def make_tf_dataset(x, y):
    x_ds = tf.data.Dataset.from_tensor_slices(x.astype(np.float32)).map(
        lambda v: tf.clip_by_value(v / 255.0, 0.0, 1.0)
    )
    y_ds = tf.data.Dataset.from_tensor_slices(y.astype(np.int64))
    return tf.data.Dataset.zip((x_ds, y_ds)).shuffle(1000, seed=42)


def collect_client_recons(attacker, client_ds_list, batch_size, eval_batches):
    per_client = {}
    all_orig, all_recon, all_labels = [], [], []
    for cid, ds in enumerate(client_ds_list):
        originals, recons, labels = [], [], []
        for x, y in ds.batch(batch_size, drop_remainder=True).take(eval_batches):
            x_recon, _ = attacker.attack(x, y, client_id=cid)
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


def dirichlet_partition_indices(labels, num_clients, alpha, seed, min_size=1):
    labels = labels.flatten()
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(cls_indices)).astype(int)
            for cid, split in enumerate(np.split(cls_indices, split_points)):
                client_indices[cid].extend(split.tolist())
        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size:
            break
    return [np.asarray(indices, dtype=np.int64) for indices in client_indices]


def class_counts_for_splits(labels, splits):
    labels = labels.flatten()
    class_counts = {}
    for cid, idx in enumerate(splits):
        vals, counts = np.unique(labels[idx], return_counts=True)
        class_counts[str(cid)] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
    return class_counts


def load_dirichlet_client_datasets(dataset, num_clients, alpha, seed, min_size):
    if dataset != "tinyimagenet":
        raise NotImplementedError("This runner is scoped to Tiny ImageNet.")

    (x_train, y_train), (x_test, y_test) = data.load_tiny_imagenet()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test))
    x_client, x_server, y_client, y_server = train_test_split(
        x,
        y,
        train_size=0.5,
        random_state=42,
        stratify=y.flatten(),
    )

    client_splits = dirichlet_partition_indices(y_client, num_clients, alpha, seed, min_size)
    server_splits = dirichlet_partition_indices(y_server, num_clients, alpha, seed + 1000, min_size)

    client_ds_list = []
    server_ds_list = []
    for cid, (client_idx, server_idx) in enumerate(zip(client_splits, server_splits)):
        client_ds_list.append(
            data.make_dataset(
                x_client[client_idx],
                y_client[client_idx],
                lambda x: tf.clip_by_value(x / 255.0, 0.0, 1.0),
                seed=42 + cid,
            )
        )
        server_ds_list.append(
            data.make_dataset(
                x_server[server_idx],
                y_server[server_idx],
                lambda x: tf.clip_by_value(x / 255.0, 0.0, 1.0),
                seed=420 + cid,
            )
        )

    return client_ds_list, server_ds_list, class_counts_for_splits(y_client, client_splits)


def tinyimagenet_dirichlet_local_test_client_datasets(num_clients, alpha, seed, min_size=1):
    """Local test: each client is evaluated on a deterministic Dirichlet validation subset."""
    (_, _), (x_test, y_test) = data.load_tiny_imagenet()
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int64)
    splits = dirichlet_partition_indices(y_test, num_clients, alpha, seed, min_size)
    return [make_tf_dataset(x_test[idx], y_test[idx]) for idx in splits]


def tinyimagenet_global_test_client_datasets(num_clients):
    """Global test: each client front is evaluated on the full 200-class test set."""
    (_, _), (x_test, y_test) = data.load_tiny_imagenet()
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int64)
    full_ds = make_tf_dataset(x_test, y_test)
    return [full_ds for _ in range(num_clients)]


def evaluate_split_test_accuracy(attacker, eval_ds_list, batch_size):
    per_client = {}
    total_correct = 0
    total_seen = 0
    for cid, ds in enumerate(eval_ds_list):
        correct = 0
        seen = 0
        for x, y in ds.batch(batch_size, drop_remainder=False):
            z = attacker.fronts[cid](x, training=False)
            logits = attacker.g(z, training=False)
            pred = tf.argmax(logits, axis=1, output_type=tf.int64)
            y_flat = tf.reshape(tf.cast(y, tf.int64), [-1])
            batch_correct = int(tf.reduce_sum(tf.cast(pred == y_flat, tf.int32)).numpy())
            correct += batch_correct
            seen += int(y_flat.shape[0])
        acc = correct / seen if seen else float("nan")
        per_client[str(cid)] = {"correct": correct, "n": seen, "accuracy": float(acc)}
        total_correct += correct
        total_seen += seen
    acc_values = np.asarray([m["accuracy"] for m in per_client.values()], dtype=np.float32)
    return {
        "test_accuracy": float(total_correct / total_seen),
        "test_correct": int(total_correct),
        "test_n": int(total_seen),
        "per_client_test_accuracy": per_client,
        "best_client_test_accuracy": float(np.nanmax(acc_values)),
        "worst_client_test_accuracy": float(np.nanmin(acc_values)),
        "test_accuracy_std_across_clients": float(np.nanstd(acc_values)),
    }


def fixed_tinyimagenet_dirichlet_probes(num_clients, alpha, seed, images_per_client=2):
    (_, _), (x_test, y_test) = data.load_tiny_imagenet()
    x_test = np.clip(x_test.astype(np.float32) / 255.0, 0.0, 1.0)
    y_test = y_test.astype(np.int64).reshape(-1)
    splits = dirichlet_partition_indices(
        y_test,
        num_clients,
        alpha,
        seed,
        min_size=images_per_client,
    )

    probe_x, probe_y, probe_client_ids, probe_indices = [], [], [], []
    for cid, split in enumerate(splits):
        idx = split[:images_per_client]
        probe_x.append(x_test[idx])
        probe_y.append(y_test[idx])
        probe_client_ids.extend([cid] * len(idx))
        probe_indices.extend(idx.tolist())

    return (
        np.concatenate(probe_x, axis=0),
        np.concatenate(probe_y, axis=0).reshape(-1, 1),
        np.asarray(probe_client_ids, dtype=np.int64),
        np.asarray(probe_indices, dtype=np.int64),
    )


def reconstruct_fixed_probes(attacker, probe_x, probe_y, probe_client_ids):
    recons = []
    for x, y, cid in zip(probe_x, probe_y, probe_client_ids):
        xb = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
        yb = tf.convert_to_tensor(y.reshape(1, 1), dtype=tf.int64)
        x_recon, _ = attacker.attack(xb, yb, client_id=int(cid))
        recons.append(x_recon.numpy()[0])
    return np.asarray(recons, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="SDAR Tiny ImageNet SGLR-style PSL Dirichlet non-IID run with separate client fronts and one shared server.")
    parser.add_argument("--dataset", default="tinyimagenet", choices=["tinyimagenet"])
    parser.add_argument("--model", default="resnet18", choices=["resnet18"])
    parser.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--client-batch-size", type=int, default=32)
    parser.add_argument("--num-iter", type=int, default=20000)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--fixed-probe-per-client", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splitlr-alpha", type=float, default=0.5)
    parser.add_argument("--splitavg-phi", type=float, default=0.5)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--output-dir", default="results/sdar_psl_dirichlet_a05_sglr")
    parser.add_argument("--print-to-stdout", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    num_classes = 200
    alpha_tag = str(args.alpha).replace(".", "p")
    run_name = f"{args.dataset}_{args.model}_{args.cut}_K{args.num_clients}_dirichlet_a{alpha_tag}_psl_sglr_run{args.run}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_fp = None
    if not args.print_to_stdout:
        log_fp = open(out_dir / "run.log", "w")
        sys.stdout = log_fp

    print(f"run_name={run_name}")
    print(f"mode=parallel_split_learning_sglr partition=dirichlet alpha={args.alpha} client_fronts=separate server=shared")
    print(f"client_batch_size={args.client_batch_size} effective_server_batch_size={args.client_batch_size * args.num_clients}")
    print(f"splitlr_alpha={args.splitlr_alpha} splitavg_phi={args.splitavg_phi}")

    min_size = args.client_batch_size * args.eval_batches
    client_ds_list, server_ds_list, class_counts = load_dirichlet_client_datasets(
        args.dataset,
        args.num_clients,
        args.alpha,
        args.seed,
        min_size,
    )
    print(f"class_counts_per_client={class_counts}")
    attacker = SDARPSLHeteroAttacker(
        client_ds_list,
        server_ds_list,
        num_class=num_classes,
        batch_size=args.client_batch_size,
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
        splitlr_alpha=args.splitlr_alpha,
        splitavg_phi=args.splitavg_phi,
        verbose_freq=100,
    )
    train_attack_runtime = time.time() - start

    comm_start = time.time()
    sample_x, _ = next(iter(client_ds_list[0].batch(args.client_batch_size, drop_remainder=True)))
    z = attacker.fronts[0](sample_x, training=False)
    per_client_smashed_shape = [int(v) for v in z.shape]
    effective_smashed_shape = [args.client_batch_size * args.num_clients] + per_client_smashed_shape[1:]
    bytes_per_sample = int(np.prod(per_client_smashed_shape[1:]) * 4)
    bytes_per_client_batch = int(np.prod(per_client_smashed_shape) * 4)
    bytes_per_global_round = int(bytes_per_client_batch * args.num_clients)
    communication = {
        "dtype": "float32",
        "per_client_smashed_shape": per_client_smashed_shape,
        "effective_smashed_shape": effective_smashed_shape,
        "bytes_per_sample": bytes_per_sample,
        "bytes_per_client_batch": bytes_per_client_batch,
        "bytes_per_global_round": bytes_per_global_round,
        "mb_per_client_batch": bytes_per_client_batch / (1024**2),
        "mb_per_global_round": bytes_per_global_round / (1024**2),
        "global_rounds": args.num_iter,
        "clients_per_round": args.num_clients,
        "client_batch_size": args.client_batch_size,
        "effective_server_batch_size": args.client_batch_size * args.num_clients,
        "total_smashed_bytes": int(bytes_per_global_round * args.num_iter),
        "total_smashed_mb": float(bytes_per_global_round * args.num_iter / (1024**2)),
    }

    eval_start = time.time()
    per_client, originals, recons, labels = collect_client_recons(
        attacker, client_ds_list, args.client_batch_size, args.eval_batches
    )
    evaluation_runtime = time.time() - eval_start
    global_metrics = reconstruction_metrics(originals, recons)
    fixed_probe_x, fixed_probe_y, fixed_probe_client_ids, fixed_probe_indices = fixed_tinyimagenet_dirichlet_probes(
        args.num_clients,
        args.alpha,
        args.seed + 2000,
        images_per_client=args.fixed_probe_per_client,
    )
    fixed_probe_recons = reconstruct_fixed_probes(
        attacker,
        fixed_probe_x,
        fixed_probe_y,
        fixed_probe_client_ids,
    )
    fixed_probe_metrics = reconstruction_metrics(fixed_probe_x, fixed_probe_recons)
    local_test_accuracy = evaluate_split_test_accuracy(
        attacker,
        tinyimagenet_dirichlet_local_test_client_datasets(
            args.num_clients,
            args.alpha,
            args.seed + 2000,
            min_size=1,
        ),
        args.client_batch_size,
    )
    global_test_accuracy = evaluate_split_test_accuracy(
        attacker,
        tinyimagenet_global_test_client_datasets(args.num_clients),
        args.client_batch_size,
    )

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
        "client_fronts": "separate",
        "server_model": "shared",
        "attacker_simulator": "shared",
        "attacker_decoder": "shared",
        "partition": "dirichlet",
        "dirichlet_alpha": args.alpha,
        "dirichlet_seed": args.seed,
        "class_counts_per_client": class_counts,
        "orchestration": "parallel_split_learning_sglr",
        "psl_mode": "sglr_splitlr_splitavg",
        "splitlr_alpha": args.splitlr_alpha,
        "splitavg_phi": args.splitavg_phi,
        "active_clients_per_round": attacker.splitavg_active_clients_per_round,
        "splitavg_active_clients_per_round": attacker.splitavg_active_clients_per_round,
        "client_front_lr": attacker.fg_client_lr,
        "server_lr": attacker.fg_server_lr,
        "client_batch_size": args.client_batch_size,
        "effective_server_batch_size": args.client_batch_size * args.num_clients,
        "num_iter": args.num_iter,
        "eval_batches_per_client": args.eval_batches,
        "n_eval_per_client": args.client_batch_size * args.eval_batches,
        "fixed_probe_per_client": args.fixed_probe_per_client,
        "fixed_probe_n": int(len(fixed_probe_x)),
        "fixed_probe_indices": fixed_probe_indices.tolist(),
        "fixed_probe_client_ids": fixed_probe_client_ids.tolist(),
        "config": config,
        "fg_acc_last100": float(np.mean(history["fg_acc"][-100:])),
        "fg_loss_last100": float(np.mean(history["fg_loss"][-100:])),
        "local_test_note": "Each client is evaluated on its deterministic Dirichlet alpha validation subset.",
        **{f"local_{k}": v for k, v in local_test_accuracy.items()},
        "global_test_note": "Each client front is evaluated on the full 200-class Tiny ImageNet validation set.",
        **{f"global_{k}": v for k, v in global_test_accuracy.items()},
        **global_metrics,
        **{f"fixed_probe_{k}": v for k, v in fixed_probe_metrics.items()},
        "best_client_ssim": float(np.nanmax(ssim_means)),
        "worst_client_ssim": float(np.nanmin(ssim_means)),
        "ssim_std_across_clients": float(np.nanstd(ssim_means)),
        "mse_std_across_clients": float(np.nanstd(mse_means)),
        "lpips_std_across_clients": float(np.nanstd(lpips_means)),
        **{f"comm_{k}": v for k, v in communication.items()},
        **timing,
    }

    np.save(out_dir / "history.npy", history)
    np.save(out_dir / "originals.npy", originals)
    np.save(out_dir / "reconstructions.npy", recons)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "fixed_probe_originals.npy", fixed_probe_x)
    np.save(out_dir / "fixed_probe_reconstructions.npy", fixed_probe_recons)
    np.save(out_dir / "fixed_probe_labels.npy", fixed_probe_y)
    np.save(out_dir / "fixed_probe_client_ids.npy", fixed_probe_client_ids)
    np.save(out_dir / "fixed_probe_indices.npy", fixed_probe_indices)
    for cid, payload in per_client.items():
        np.save(out_dir / f"client_{cid}_originals.npy", payload["originals"])
        np.save(out_dir / f"client_{cid}_reconstructions.npy", payload["reconstructions"])
        save_grid(
            [payload["originals"], payload["reconstructions"]],
            [f"client {cid} orig", f"client {cid} recon"],
            out_dir / f"client_{cid}_grid.png",
        )
    save_grid([originals, recons], ["Original", "Recon"], out_dir / "original_recon_grid.png")
    save_grid([fixed_probe_x, fixed_probe_recons], ["Fixed Original", "Fixed Recon"], out_dir / "fixed_probe_grid.png")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_client_metrics.json", "w") as f:
        json.dump(client_metrics, f, indent=2)
    with open(out_dir / "communication.json", "w") as f:
        json.dump(communication, f, indent=2)
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    with open(out_dir / "local_test_accuracy.json", "w") as f:
        json.dump(local_test_accuracy, f, indent=2)
    with open(out_dir / "global_test_accuracy.json", "w") as f:
        json.dump(global_test_accuracy, f, indent=2)
    with open(out_dir / "fixed_probe_metrics.json", "w") as f:
        json.dump(fixed_probe_metrics, f, indent=2)

    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    for cid, front in enumerate(attacker.fronts):
        front.save_weights(str(weights_dir / f"front_{cid}.weights.h5"))
    attacker.g.save_weights(str(weights_dir / "server_g.weights.h5"))

    print("saved:", out_dir)
    print(json.dumps(summary, indent=2))
    if log_fp is not None:
        log_fp.close()


if __name__ == "__main__":
    main()
