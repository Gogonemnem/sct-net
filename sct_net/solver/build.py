import torch

from detectron2.config import CfgNode
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    # Handle freezing of RPN parameters
    if cfg.MODEL.RPN.FREEZE_RPN:
        for name, param in model.named_parameters():
            if "proposal_generator" in name:
                param.requires_grad = False

    # Adjust learning rate for 'box_predictor' layers
    def lr_factor_func(param_name):
        if 'box_predictor' in param_name:
            return cfg.SOLVER.HEAD_LR_FACTOR
        return 1.0

    # Get parameter groups with appropriate hyperparameters
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        lr_factor_func=lr_factor_func,
    )

    optimizer_type = cfg.SOLVER.SOLVER_TYPE.lower()

    optimizer_kwargs = {
        "params": params,
        "lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }

    if optimizer_type == "sgd":
        optimizer_cls = torch.optim.SGD
        optimizer_kwargs.update({
            "momentum": cfg.SOLVER.MOMENTUM,
            "nesterov": cfg.SOLVER.NESTEROV,
        })
        print("============ Using the SGD optimizer ============")
    elif optimizer_type == "adamw":
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs.update({
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
            "eps": cfg.SOLVER.EPS,
        })
        print("============ Using the AdamW optimizer ============")
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.SOLVER.SOLVER_TYPE}")

    optimizer = optimizer_cls(**optimizer_kwargs)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer
