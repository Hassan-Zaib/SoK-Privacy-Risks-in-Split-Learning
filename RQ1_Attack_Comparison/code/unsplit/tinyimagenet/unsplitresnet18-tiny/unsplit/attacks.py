from __future__ import annotations

import torch

from .util import TV, l2loss


# Same official UnSplit logic, with ResNet client wrappers accepting end=split_layer.
def model_inversion_stealing(
    clone_model,
    split_layer,
    target,
    input_size,
    lambda_tv=0.1,
    lambda_l2=0.0,
    main_iters=1000,
    input_iters=100,
    model_iters=100,
):
    x_pred = torch.empty(input_size, device=target.device).fill_(0.5).requires_grad_(True)
    input_opt = torch.optim.Adam([x_pred], lr=0.001, amsgrad=True)
    model_opt = torch.optim.Adam(clone_model.parameters(), lr=0.001, amsgrad=True)
    mse = torch.nn.MSELoss()

    for _ in range(main_iters):
        for _ in range(input_iters):
            input_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = mse(pred, target) + lambda_tv * TV(x_pred) + lambda_l2 * l2loss(x_pred)
            loss.backward(retain_graph=True)
            input_opt.step()
        for _ in range(model_iters):
            model_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = mse(pred, target)
            loss.backward(retain_graph=True)
            model_opt.step()

    return x_pred.detach()
