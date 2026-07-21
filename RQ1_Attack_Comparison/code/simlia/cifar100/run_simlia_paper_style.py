import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import data as data_mod
from models.resnet18 import build_split_resnet18
from run_simlia import collect_smashed, one_known_per_class, device_auto

def get_dataset():
    if hasattr(data_mod, "cifar10_loaders"):
        return "cifar10", data_mod.cifar10_loaders, 10
    if hasattr(data_mod, "cifar100_loaders"):
        return "cifar100", data_mod.cifar100_loaders, 100
    return "tinyimagenet", data_mod.tinyimagenet_loaders, 200

def reduce_features(train_x, test_x, pca_dims):
    used_pca = train_x.shape[1] > pca_dims
    if used_pca:
        pca = IncrementalPCA(n_components=pca_dims, batch_size=max(1024, pca_dims * 2))
        pca.fit(train_x)
        train_x = pca.transform(train_x).astype("float32")
        test_x = pca.transform(test_x).astype("float32")
    return normalize(train_x).astype("float32"), normalize(test_x).astype("float32"), used_pca

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--pca-dims", type=int, default=512)
    ap.add_argument("--n-attack", type=int, default=10000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[123,124,125,126,127])
    ap.add_argument("--output-dir", default="results/simlia_paper")
    args = ap.parse_args()

    start = time.time()
    dataset, loader_fn, num_classes = get_dataset()
    device = device_auto()

    _, _, trainloader, testloader = loader_fn(batch_size=args.batch_size, download=True, num_workers=args.num_workers)
    client, server = build_split_resnet18(args.cut, num_classes=num_classes)

    ckpt_path = Path("checkpoints") / f"{dataset}_resnet18_{args.cut}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    client.load_state_dict(ckpt["client_state_dict"])
    client.to(device)

    train_x, train_y, _ = collect_smashed(client, trainloader, device)
    test_x, test_y, _ = collect_smashed(client, testloader, device)
    train_x, test_x, used_pca = reduce_features(train_x, test_x, args.pca_dims)

    attack_x, attack_y = test_x, test_y
    if args.n_attack > 0 and args.n_attack < len(test_y):
        rng = np.random.default_rng(999)
        idx = np.sort(rng.choice(np.arange(len(test_y)), size=args.n_attack, replace=False))
        attack_x, attack_y = test_x[idx], test_y[idx]

    rows = []
    for seed in args.seeds:
        known = one_known_per_class(train_y, seed, num_classes)
        clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        clf.fit(train_x[known], train_y[known])
        pred = clf.predict(attack_x)
        rows.append({
            "seed": seed,
            "accuracy": float(accuracy_score(attack_y, pred)),
            "macro_f1": float(f1_score(attack_y, pred, average="macro")),
            "weighted_f1": float(f1_score(attack_y, pred, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(attack_y, pred)),
        })

    out = Path(args.output_dir) / f"{dataset}_resnet18_{args.cut}"
    out.mkdir(parents=True, exist_ok=True)

    with (out / "paper_trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    acc = [r["accuracy"] for r in rows]
    mf1 = [r["macro_f1"] for r in rows]
    summary = {
        "attack": "simlia_euclidean_nearest_reference",
        "dataset": dataset,
        "cut": args.cut,
        "trials": len(rows),
        "seeds": args.seeds,
        "known_per_class": 1,
        "avg_attack_accuracy": float(np.mean(acc)),
        "max_attack_accuracy": float(np.max(acc)),
        "avg_macro_f1": float(np.mean(mf1)),
        "max_macro_f1": float(np.max(mf1)),
        "best_model_accuracy": float(ckpt["metrics"]["best_test_accuracy"]),
        "used_pca": used_pca,
        "runtime_sec": time.time() - start,
    }
    (out / "paper_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
