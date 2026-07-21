import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import data
from sdar.sdar import SDARAttacker
from sdar.sdar_hetero import SDARHeteroAttacker
from util.util import plot_attack_results, load_config

def reconstruction_metrics(x, x_recon):
    import numpy as _np
    mse_per = _np.mean((x - x_recon) ** 2, axis=(1, 2, 3))
    mse = float(_np.mean(mse_per))
    psnr_per = _np.where(mse_per <= 0, _np.inf, -10.0 * _np.log10(mse_per + 1e-12))
    metrics = {
        "mse_mean": mse,
        "mse_std": float(_np.std(mse_per)),
        "mse_min": float(_np.min(mse_per)),
        "mse_max": float(_np.max(mse_per)),
        "psnr_mean": float(_np.mean(psnr_per)),
        "psnr_std": float(_np.std(psnr_per)),
        "psnr_min": float(_np.min(psnr_per)),
        "psnr_max": float(_np.max(psnr_per)),
    }
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_vals = [_ssim(x[i], x_recon[i], channel_axis=2, data_range=1.0) for i in range(len(x))]
        metrics.update({
            "ssim_mean": float(_np.mean(ssim_vals)),
            "ssim_std": float(_np.std(ssim_vals)),
            "ssim_min": float(_np.min(ssim_vals)),
            "ssim_max": float(_np.max(ssim_vals)),
            "asr_ssim_0.90": float(_np.mean(_np.asarray(ssim_vals) >= 0.90)),
        })
    except Exception as e:
        metrics["ssim_error"] = repr(e)
    try:
        import lpips as _lpips
        import torch as _torch
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        loss_fn = _lpips.LPIPS(net="alex").to(device).eval()
        xt = _torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device) * 2 - 1
        rt = _torch.from_numpy(x_recon).permute(0, 3, 1, 2).float().to(device) * 2 - 1
        with _torch.no_grad():
            vals = loss_fn(xt, rt).view(-1).detach().cpu().numpy()
        metrics.update({
            "lpips_mean": float(_np.mean(vals)),
            "lpips_std": float(_np.std(vals)),
            "lpips_min": float(_np.min(vals)),
            "lpips_max": float(_np.max(vals)),
        })
    except Exception as e:
        metrics["lpips_error"] = repr(e)
    return metrics

parser = argparse.ArgumentParser(description='Run SDAR experiment with specified configurations.')
parser.add_argument("setting", type=str, choices=["vsl", "usl"])
parser.add_argument("model", type=str, choices=["resnet", "plainnet", "resnet18"])
parser.add_argument("dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])
parser.add_argument("level", type=str, choices=["4","5","6","7","8","9","c1","c2","c3","c4"])
parser.add_argument("run", type=int, help="Run number.")

parser.add_argument("--width", type=str, default="standard", choices=["standard", "wide", "narrow"])
parser.add_argument("--aux_data_frac", type=float, default=1.0)
parser.add_argument("--num_class_to_remove", type=int, default=0)
parser.add_argument("--diff_simulator", action="store_true")

parser.add_argument("--ablation", type=str, choices=["no_e_dis", "no_d_dis", "no_dis", "no_cond", "no_label_flip", "naive_sda"], default=None)

# number of heterogenous distributions of the client data
parser.add_argument("--num_hetero_client", type=int, default=1)

# the following arguments are used to specify the defense method
parser.add_argument("--l1", type=float, default=0.0)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=0.0)

parser.add_argument("--print_to_stdout", action="store_true")

args = parser.parse_args()

numeric_level = int(args.level) if str(args.level).isdigit() else None
if args.model == "resnet18" and args.setting != "vsl":
    raise ValueError("resnet18 adaptation currently supports vanilla SL only.")
if args.model == "resnet18" and args.dataset != "cifar10":
    raise ValueError("resnet18 adaptation currently targets CIFAR-10 first.")
if sum([numeric_level is not None and numeric_level > 7, args.l1 != 0.0, args.l2 != 0.0, args.dropout != 0.0, args.alpha != 0.0]) > 1:
        raise ValueError("At most one of l1, l2, dropout, alpha can be not default.")

defense = sum([numeric_level is not None and numeric_level > 7, args.l1 != 0.0, args.l2 != 0.0, args.dropout != 0.0, args.alpha != 0.0]) > 0

# at most one of width and aux_data_frac can be not default
if sum([args.width != "standard", args.aux_data_frac != 1.0, args.diff_simulator, args.num_class_to_remove != 0, args.ablation != None, args.num_hetero_client > 1, defense]) > 1:
    raise ValueError("Only one of width, aux_data_frac, diff_simulator, num_class_to_remove, ablation, num_hetero_client, defense can be not default.")

# get the directory of this file
curr_dir = os.path.join(os.path.dirname(__file__))

if args.width != "standard":
    rela_dir = f"model_width_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_{args.width}_l{args.level}_run{args.run}"
elif args.aux_data_frac != 1.0:
    rela_dir = f"data_frac_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_{args.aux_data_frac}_l{args.level}_run{args.run}"
elif args.diff_simulator:
    rela_dir = f"diff_simulator_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"
elif args.num_class_to_remove != 0:
    rela_dir = f"remove_aux_class_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_aux_wo_{args.num_class_to_remove}_l{args.level}_run{args.run}"
elif args.ablation != None:
    rela_dir = f"ablation_results/{args.ablation}/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"
elif args.num_hetero_client > 1:
    rela_dir = f"hetero_clients_results/{args.setting}/{args.dataset}"
    file_name = f"{args.num_hetero_client}_{args.model}_l{args.level}_run{args.run}"
elif defense:
    if args.l1 != 0.0:
        rela_dir = f"defense_results/l1/{args.setting}/{args.dataset}"
        file_name = f"{args.l1}_{args.model}_l{args.level}_run{args.run}"
    elif args.l2 != 0.0:
        rela_dir = f"defense_results/l2/{args.setting}/{args.dataset}"
        file_name = f"{args.l2}_{args.model}_l{args.level}_run{args.run}"
    elif args.dropout != 0.0:
        rela_dir = f"defense_results/dropout/{args.setting}/{args.dataset}"
        file_name = f"{args.dropout}_{args.model}_l{args.level}_run{args.run}"
    elif args.alpha != 0.0:
        rela_dir = f"defense_results/decorrelation/{args.setting}/{args.dataset}"
        file_name = f"{args.alpha}_{args.model}_l{args.level}_run{args.run}"
    elif numeric_level is not None and numeric_level > 7:
        rela_dir = f"defense_results/deeper_f/{args.setting}/{args.dataset}"
        file_name = f"{args.model}_l{args.level}_run{args.run}"
else:
    rela_dir = f"sdar_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"

base_dir = os.path.join(curr_dir, rela_dir)
fig_dir = os.path.join(base_dir, "fig")
output_dir = os.path.join(base_dir, "output")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not args.print_to_stdout:
    sys.stdout = open(os.path.join(base_dir, file_name + ".txt"),'wt')

u_shape = (args.setting == "usl")
conditional = not u_shape
num_class = 100 if args.dataset == "cifar100" else 200 if args.dataset == "tinyimagenet" else 10
batch_size = 32 if args.dataset == "stl10" else 128 # since stl10 contains only 13000 images
num_iter = 40000 if (numeric_level is not None and numeric_level > 7) else 20000 if args.dataset in ["cifar10", "cifar100", "tinyimagenet", "stl10"] else 10000
config = load_config("sdar", args.model, args.dataset, u_shape)

if args.ablation == "no_e_dis":
    config["lambda1"] = 0.0
elif args.ablation == "no_d_dis":
    config["lambda2"] = 0.0
elif args.ablation == "no_dis":
    config["lambda1"] = 0.0
    config["lambda2"] = 0.0
elif args.ablation == "no_cond":
    conditional = False
elif args.ablation == "no_label_flip":
    config["flip_rate"] = 0.0
elif args.ablation == "naive_sda":
    config["lambda1"] = 0.0
    config["lambda2"] = 0.0
    config["flip_rate"] = 0.0
    conditional = False

if args.num_hetero_client == 1:
    client_ds, server_ds = data.load_dataset(args.dataset, frac=args.aux_data_frac, num_class_to_remove=args.num_class_to_remove)
else:
    client_ds_list, server_ds_list = data.load_dataset_with_hetero_clients(args.dataset, args.num_hetero_client)

print(f"Start {args.setting}/{args.model} experiments on {args.dataset} at l{args.level} with {config}")

if args.num_hetero_client == 1:
    sdar_attacker = SDARAttacker(client_ds, server_ds, num_class=num_class, batch_size=batch_size)
else:
    sdar_attacker = SDARHeteroAttacker(client_ds_list, server_ds_list, num_class=num_class, batch_size=batch_size)

start_time = time.time()
history = sdar_attacker.run(level=args.level, num_iter=num_iter, config=config, u_shape=u_shape, conditional=conditional, model_type=args.model, width=args.width, diff_simulator=args.diff_simulator, l1=args.l1, l2=args.l2, dropout=args.dropout, alpha=args.alpha, verbose_freq=100)
end_time = time.time()

print(f"Experiment took {end_time - start_time} seconds, on average {(end_time - start_time)/num_iter} seconds/iteration.")
print("Experiment done. Saving history...")

if args.num_hetero_client == 1:
    _ = sdar_attacker.evaluate(verbose=True)
else:
    eval_ds, _ = data.load_dataset(args.dataset, frac=args.aux_data_frac, num_class_to_remove=args.num_class_to_remove)
    _ = sdar_attacker.evaluate(eval_ds, verbose=True)

eval_batch = data.load_single_batch(args.dataset, batch_size=batch_size)
x,y = [(x,y) for (x,y) in iter(eval_batch)][0]

x_recon, mse = sdar_attacker.attack(x, y)
t = 10
plot_attack_results(x_recon[t:t+20], os.path.join(fig_dir, file_name + ".png"))
example_mse = np.mean((x[t:t+20] - x_recon[t:t+20])**2)
print(f"Attack MSE on batch: {mse}, on these examples: {example_mse}")

extra_metrics = reconstruction_metrics(x.numpy() if hasattr(x, "numpy") else x, x_recon.numpy() if hasattr(x_recon, "numpy") else x_recon)
print("Reconstruction metrics:")
for k in sorted(extra_metrics):
    print(f"  {k}: {extra_metrics[k]}")

np.save(os.path.join(output_dir, file_name + ".npy"), history)
np.save(os.path.join(output_dir, file_name + "_recon.npy"), x_recon)
np.save(os.path.join(output_dir, file_name + "_orig.npy"), x)
with open(os.path.join(output_dir, file_name + "_metrics.json"), "w") as f:
    import json
    json.dump(extra_metrics, f, indent=2)
print(f"History saved to {os.path.join(rela_dir, file_name + '.npy')}")
print(f"Metrics saved to {os.path.join(rela_dir, 'output', file_name + '_metrics.json')}")
