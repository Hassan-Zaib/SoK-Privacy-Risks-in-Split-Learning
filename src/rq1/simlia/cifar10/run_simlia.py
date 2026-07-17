from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from torch import nn
from tqdm import tqdm

from data import cifar10_loaders
from models.resnet18 import build_split_resnet18


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_smashed(client: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    client.eval()
    features = []
    labels_all = []
    indices_all = []
    offset = 0
    for images, labels in tqdm(loader, desc="collect smashed", leave=False):
        images = images.to(device)
        z = client(images).detach().cpu().numpy().astype("float32")
        features.append(z.reshape(z.shape[0], -1))
        labels_np = labels.numpy()
        labels_all.append(labels_np)
        indices_all.append(np.arange(offset, offset + len(labels_np)))
        offset += len(labels_np)
    return np.concatenate(features), np.concatenate(labels_all), np.concatenate(indices_all)


def official_style_features(train_x: np.ndarray, test_x: np.ndarray, pca_dims: int) -> tuple[np.ndarray, np.ndarray, bool]:
    used_pca = train_x.shape[1] > pca_dims
    if used_pca:
        pca = IncrementalPCA(n_components=pca_dims, batch_size=max(1024, pca_dims * 2))
        pca.fit(train_x)
        train_x = pca.transform(train_x).astype("float32")
        test_x = pca.transform(test_x).astype("float32")
    train_x = normalize(train_x).astype("float32")
    test_x = normalize(test_x).astype("float32")
    return train_x, test_x, used_pca


def one_known_per_class(labels: np.ndarray, seed: int, num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    known_indices = []
    for cls in range(num_classes):
        candidates = np.flatnonzero(labels == cls)
        if len(candidates) == 0:
            raise RuntimeError(f"no reference sample found for class {cls}")
        known_indices.append(int(rng.choice(candidates)))
    return np.array(known_indices, dtype=np.int64)


def run_attack(args: argparse.Namespace) -> None:
    start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = device_auto()

    trainset, testset, trainloader, testloader = cifar10_loaders(
        batch_size=args.batch_size,
        download=True,
        num_workers=args.num_workers,
    )

    client, server = build_split_resnet18(args.cut)
    ckpt_path = Path(args.checkpoint_dir) / f"cifar10_resnet18_{args.cut}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    client.load_state_dict(ckpt["client_state_dict"])
    server.load_state_dict(ckpt["server_state_dict"])
    client.to(device)

    train_x, train_y, train_indices = collect_smashed(client, trainloader, device)
    test_x, test_y, test_indices = collect_smashed(client, testloader, device)
    train_x, test_x, used_pca = official_style_features(train_x, test_x, args.pca_dims)

    known_indices = one_known_per_class(train_y, args.seed, args.num_classes)
    known_x = train_x[known_indices]
    known_y = train_y[known_indices]

    if args.n_attack > 0 and args.n_attack < len(test_y):
        rng = np.random.default_rng(args.seed)
        attack_indices = np.sort(rng.choice(np.arange(len(test_y)), size=args.n_attack, replace=False))
        attack_x = test_x[attack_indices]
        attack_y = test_y[attack_indices]
        attack_sample_indices = test_indices[attack_indices]
    else:
        attack_indices = np.arange(len(test_y))
        attack_x = test_x
        attack_y = test_y
        attack_sample_indices = test_indices

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(known_x, known_y)
    pred_y = knn.predict(attack_x)

    out_dir = Path(args.output_dir) / f"cifar10_resnet18_{args.cut}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(attack_y, pred_y, labels=list(range(args.num_classes)))
    np.save(out_dir / "confusion_matrix.npy", cm)

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_index", "true_label", "pred_label"])
        writer.writeheader()
        for idx, true_label, pred_label in zip(attack_sample_indices, attack_y, pred_y):
            writer.writerow({"test_index": int(idx), "true_label": int(true_label), "pred_label": int(pred_label)})

    known_rows = []
    for local_index in known_indices:
        known_rows.append({
            "train_index": int(train_indices[local_index]),
            "label": int(train_y[local_index]),
        })
    (out_dir / "known_reference_samples.json").write_text(json.dumps(known_rows, indent=2) + "\n")

    summary = {
        "attack": "simlia_smashed_knn",
        "source": "official Sim-LIA repo logic: flattened smashed data + optional PCA + L2 normalization + 1-NN",
        "dataset": "cifar10",
        "cut": args.cut,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size,
        "known_per_class": 1,
        "pca_dims": args.pca_dims,
        "used_pca": used_pca,
        "n_attack": int(len(attack_y)),
        "accuracy": float(accuracy_score(attack_y, pred_y)),
        "macro_f1": float(f1_score(attack_y, pred_y, average="macro")),
        "weighted_f1": float(f1_score(attack_y, pred_y, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(attack_y, pred_y)),
        "checkpoint": str(ckpt_path),
        "primary_best_test_accuracy": float(ckpt["metrics"]["best_test_accuracy"]),
        "primary_final_test_accuracy": float(ckpt["metrics"]["test_accuracy"]),
        "runtime_sec": time.time() - start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--n-attack", type=int, default=10000, help="Use <=0 for all test samples.")
    p.add_argument("--pca-dims", type=int, default=512)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="results/simlia")
    return p.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
