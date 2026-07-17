from __future__ import annotations

import math

import numpy as np
import torch


def image_metrics(x: torch.Tensor, recon: torch.Tensor, use_lpips: bool = True):
    x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    r_np = recon.detach().cpu().permute(0, 2, 3, 1).numpy()
    mse_per = np.mean((x_np - r_np) ** 2, axis=(1, 2, 3))
    psnr_per = np.array([math.inf if m <= 0 else -10.0 * math.log10(float(m) + 1e-12) for m in mse_per])
    out = {
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
        from skimage.metrics import structural_similarity

        ssim_vals = [
            structural_similarity(x_np[i], r_np[i], channel_axis=2, data_range=1.0)
            for i in range(len(x_np))
        ]
        out.update(
            {
                "ssim_mean": float(np.mean(ssim_vals)),
                "ssim_std": float(np.std(ssim_vals)),
                "ssim_min": float(np.min(ssim_vals)),
                "ssim_max": float(np.max(ssim_vals)),
                "asr_ssim_0.90": float(np.mean(np.asarray(ssim_vals) >= 0.90)),
            }
        )
    except Exception as exc:
        out["ssim_error"] = repr(exc)

    if use_lpips:
        try:
            import lpips

            device = x.device
            loss_fn = lpips.LPIPS(net="alex").to(device).eval()
            with torch.no_grad():
                vals = loss_fn(x * 2 - 1, recon * 2 - 1).view(-1).detach().cpu().numpy()
            out.update(
                {
                    "lpips_mean": float(np.mean(vals)),
                    "lpips_std": float(np.std(vals)),
                    "lpips_min": float(np.min(vals)),
                    "lpips_max": float(np.max(vals)),
                }
            )
        except Exception as exc:
            out["lpips_error"] = repr(exc)
    return out
