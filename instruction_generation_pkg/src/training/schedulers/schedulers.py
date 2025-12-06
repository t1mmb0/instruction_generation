import torch.optim.lr_scheduler as lr_scheduler


# -----------------------
# EPOCH-BASED SCHEDULER
# -----------------------

def cos_lr(optimizer, T_max, eta_min=0.0, last_epoch=-1):
    return lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=T_max,
        eta_min=eta_min,
        last_epoch=last_epoch
    )


def step_lr(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch
    )


def multi_step_lr(optimizer, milestones, gamma=0.1, last_epoch=-1):
    return lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=milestones,
        gamma=gamma,
        last_epoch=last_epoch
    )


def exp_lr(optimizer, gamma=0.95, last_epoch=-1):
    return lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=gamma,
        last_epoch=last_epoch
    )


# -----------------------
# METRIC-BASED SCHEDULER
# -----------------------

def plateau_lr(
    optimizer,
    mode="min",
    factor=0.1,
    patience=10,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-8
):
    """
    Muss mit scheduler.step(val_loss) aufgerufen werden.
    """
    return lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=mode,                # "min" => Loss, "max" => Score
        factor=factor,            # LR-Reduktion: LR *= factor
        patience=patience,        # Epochen ohne Verbesserung
        threshold=threshold,      # Minimal-Verbesserung
        threshold_mode=threshold_mode,
        cooldown=cooldown,        # Wartezeit nach Reduktion
        min_lr=min_lr,
        eps=eps
    )
