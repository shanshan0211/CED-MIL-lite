from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .data.dataset import SlideFeatureBatch
from .model.contrastive import SupConLoss, RDropLoss


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    auc: float
    sample_count: int


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss with optional label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if self.label_smoothing > 0 and num_classes > 1:
            smooth = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = F.one_hot(targets, num_classes).float()
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * smooth * log_probs
        loss = loss.sum(dim=-1)
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for long-tail classification (Ben-Baruch et al., 2020).

    Applies different focusing parameters for positive and negative samples,
    with probability shifting to eliminate easy negatives. Particularly
    effective when some classes have very few samples (1-3 per class).
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        probs = torch.sigmoid(logits)

        one_hot = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        probs_pos = probs * one_hot
        probs_neg = probs * (1 - one_hot)

        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(probs_pos.clamp(min=1e-8))
        log_neg = torch.log(1 - probs_neg.clamp(max=1 - 1e-8))

        loss_pos = -log_pos * one_hot * ((1 - probs_pos) ** self.gamma_pos)
        loss_neg = -log_neg * (1 - one_hot) * (probs_neg ** self.gamma_neg)

        loss = loss_pos + loss_neg
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets].unsqueeze(-1)
        return loss.sum(dim=-1).mean()


def slide_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    coords: torch.Tensor | None,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, float]:
    """Feature-level MixUp across slides (Zhang et al., 2018, adapted for MIL).

    Interpolates features between random slide pairs. Returns mixed features,
    the two label sets, and lambda for loss mixing.
    """
    B = features.size(0)
    if B < 2:
        lam = 1.0
        return features, labels, mask, coords, labels, lam

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item()) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)

    perm = torch.randperm(B, device=features.device)
    min_len = min(features.size(1), features[perm].size(1))

    mixed_features = lam * features[:, :min_len] + (1 - lam) * features[perm][:, :min_len]
    mixed_mask = mask[:, :min_len] & mask[perm][:, :min_len]
    mixed_coords = None
    if coords is not None:
        mixed_coords = lam * coords[:, :min_len] + (1 - lam) * coords[perm][:, :min_len]

    labels_b = labels[perm]
    return mixed_features, labels, mixed_mask, mixed_coords, labels_b, lam


class GradTailReweighter:
    """GradTail: amplify gradients for underrepresented classes."""

    def __init__(self, num_classes: int, momentum: float = 0.9, min_weight: float = 0.5, max_weight: float = 3.0) -> None:
        self.num_classes = num_classes
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.counts = torch.ones(num_classes, dtype=torch.float32)

    def update(self, targets: torch.Tensor) -> None:
        for cls in range(self.num_classes):
            count = (targets == cls).sum().float().item()
            self.counts[cls] = self.momentum * self.counts[cls] + (1 - self.momentum) * count

    def get_weights(self, device: torch.device) -> torch.Tensor:
        total = self.counts.sum().clamp(min=1.0)
        freq = self.counts / total
        inv_freq = 1.0 / freq.clamp(min=1e-6)
        weights = inv_freq / inv_freq.mean()
        return weights.clamp(self.min_weight, self.max_weight).to(device)


# ---------------------------------------------------------------------------
# Schedulers & weight averaging
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6, last_epoch: int = -1) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow.state_dict())


class SWAModel:
    """Stochastic Weight Averaging: accumulates and averages state dicts."""

    def __init__(self, model: nn.Module) -> None:
        self.avg_state: dict[str, torch.Tensor] = {
            k: v.clone().float() for k, v in model.state_dict().items()
        }
        self.num_averaged = 1

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, val in model.state_dict().items():
            self.avg_state[key] += val.float()
        self.num_averaged += 1

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: (v / self.num_averaged).to(v.dtype) for k, v in self.avg_state.items()}

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.state_dict())


# ---------------------------------------------------------------------------
# SAM optimizer
# ---------------------------------------------------------------------------

class SAM:
    """Sharpness-Aware Minimization (Foret et al., 2021) wrapper.

    Wraps any base optimizer. Each step: (1) compute gradient and perturb
    weights toward steepest ascent, (2) compute gradient at perturbed point,
    (3) restore weights and apply update.
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs) -> None:
        self.rho = rho
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self._sam_state: dict[int, dict] = {}

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self._sam_state[id(p)] = {"e_w": e_w}

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._sam_state.get(id(p), {})
                if "e_w" in state:
                    p.sub_(state["e_w"])
        self.base_optimizer.step()
        self._sam_state.clear()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> torch.Tensor:
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.detach().norm(2))
        if not norms:
            return torch.tensor(0.0)
        return torch.stack(norms).norm(2)

    def parameters(self):
        for group in self.param_groups:
            yield from group["params"]


class GradientCentralization:
    """Gradient Centralization (Yong et al., 2020) wrapper.

    Centralizes gradients by subtracting their mean before optimizer step.
    Proven to improve generalization with Adam-family optimizers by +0.5-1%.
    Applied only to weight matrices (ndim >= 2), not biases or norms.
    """

    @staticmethod
    @torch.no_grad()
    def centralize_gradients(optimizer) -> None:
        for group in optimizer.param_groups if hasattr(optimizer, 'param_groups') else []:
            for p in group["params"]:
                if p.grad is None or p.grad.ndim < 2:
                    continue
                p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.ndim)), keepdim=True))


class SelfDistillationLoss(nn.Module):
    """Self-Distillation from EMA teacher (Xu et al., 2023).

    The EMA model produces soft targets that guide the student model,
    providing a richer learning signal than hard labels alone. Free
    improvement since EMA is already maintained.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_loss: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach() / T, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
        return self.alpha * kd_loss + (1 - self.alpha) * hard_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    if preds.numel() == 0:
        return 0.0
    preds, targets = preds.view(-1), targets.view(-1)
    f1s: list[float] = []
    for cls in range(num_classes):
        tp = int(((preds == cls) & (targets == cls)).sum().item())
        fp = int(((preds == cls) & (targets != cls)).sum().item())
        fn = int(((preds != cls) & (targets == cls)).sum().item())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / max(len(f1s), 1))


def _balanced_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    if preds.numel() == 0:
        return 0.0
    preds, targets = preds.view(-1), targets.view(-1)
    accs: list[float] = []
    for cls in range(num_classes):
        m = targets == cls
        if m.sum() == 0:
            continue
        accs.append(((preds == cls) & m).sum().float().item() / m.sum().float().item())
    return float(sum(accs) / max(len(accs), 1))


def _multiclass_auc(logits_list: list[torch.Tensor], targets_list: list[torch.Tensor], num_classes: int) -> float:
    if not logits_list:
        return 0.0
    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    probs = F.softmax(all_logits, dim=-1)
    aucs: list[float] = []
    for cls in range(num_classes):
        scores = probs[:, cls]
        labels = (all_targets == cls).float()
        if labels.sum() == 0 or labels.sum() == labels.numel():
            continue
        idx = scores.argsort(descending=True)
        sl = labels[idx]
        tps = sl.cumsum(0)
        fps = (1.0 - sl).cumsum(0)
        tpr = tps / labels.sum().clamp(min=1)
        fpr = fps / (labels.numel() - labels.sum()).clamp(min=1)
        fpr_diff = torch.cat([fpr[:1], fpr[1:] - fpr[:-1]])
        aucs.append((fpr_diff * tpr).sum().item())
    return float(sum(aucs) / max(len(aucs), 1))


# ---------------------------------------------------------------------------
# Train config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrainConfig:
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    gradient_clip_max_norm: float = 0.0
    use_amp: bool = False
    gradient_accumulation_steps: int = 1
    use_gradtail: bool = False
    num_classes: int = 2
    moe_aux_weight: float = 0.01
    # Beyond-SOTA
    use_sam: bool = False
    sam_rho: float = 0.05
    use_rdrop: bool = False
    rdrop_alpha: float = 1.0
    use_supcon: bool = False
    supcon_weight: float = 0.1
    supcon_temperature: float = 0.07
    # Beyond-SOTA v2
    use_asymmetric_loss: bool = False
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    use_mixup: bool = False
    mixup_alpha: float = 0.4
    # Beyond-SOTA v3
    use_self_distillation: bool = False
    distill_temperature: float = 4.0
    distill_alpha: float = 0.5
    use_gradient_centralization: bool = False


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def _build_criterion(config: TrainConfig, class_weights: torch.Tensor | None, device: torch.device) -> nn.Module:
    if config.use_asymmetric_loss:
        return AsymmetricLoss(
            gamma_pos=config.asl_gamma_pos, gamma_neg=config.asl_gamma_neg,
            clip=config.asl_clip, label_smoothing=config.label_smoothing,
            class_weights=class_weights.to(device) if class_weights is not None else None,
        ).to(device)
    if config.use_focal_loss:
        return FocalLoss(
            gamma=config.focal_gamma, label_smoothing=config.label_smoothing,
            class_weights=class_weights.to(device) if class_weights is not None else None,
        ).to(device)
    if config.label_smoothing > 0:
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, weight=class_weights.to(device) if class_weights is not None else None)
    return nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)


def _forward_model(model, features, mask, coords, labels=None):
    from .model.ced_mil import CEDMIL as _CEDMIL
    if isinstance(model, _CEDMIL):
        return model(features, patch_mask=mask, labels=labels)
    if getattr(model, "use_ced_head", False):
        return model(features, patch_mask=mask, coords=coords, labels=labels)
    return model(features, patch_mask=mask, coords=coords)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[SlideFeatureBatch],
    optimizer,
    device: str,
    scheduler=None,
    ema: EMAModel | None = None,
    config: TrainConfig | None = None,
    gradtail: GradTailReweighter | None = None,
) -> EpochMetrics:
    if config is None:
        config = TrainConfig()

    target_device = _resolve_device(device)
    class_weights = gradtail.get_weights(target_device) if gradtail is not None else None
    criterion = _build_criterion(config, class_weights, target_device)
    is_sam = isinstance(optimizer, SAM)

    use_amp_flag = config.use_amp and target_device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp_flag)

    rdrop_loss_fn = RDropLoss(alpha=config.rdrop_alpha) if config.use_rdrop else None
    supcon_loss_fn = SupConLoss(temperature=config.supcon_temperature) if config.use_supcon else None
    distill_fn = SelfDistillationLoss(
        temperature=config.distill_temperature, alpha=config.distill_alpha,
    ) if config.use_self_distillation and ema is not None else None

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    accum_steps = max(1, config.gradient_accumulation_steps)

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        features = batch.features.to(target_device)
        mask = batch.mask.to(target_device)
        labels = batch.labels.to(target_device)
        coords = batch.coords.to(target_device) if hasattr(batch, "coords") and batch.coords is not None else None

        mixup_labels_b = None
        mixup_lam = 1.0
        if config.use_mixup and config.mixup_alpha > 0 and features.size(0) > 1:
            features, labels, mask, coords, mixup_labels_b, mixup_lam = slide_mixup(
                features, labels, mask, coords, alpha=config.mixup_alpha,
            )

        def _compute_loss():
            with autocast("cuda", enabled=use_amp_flag):
                output = _forward_model(model, features, mask, coords, labels=labels)
                if mixup_labels_b is not None and mixup_lam < 1.0:
                    hard_loss = mixup_lam * criterion(output.logits, labels) + (1 - mixup_lam) * criterion(output.logits, mixup_labels_b)
                else:
                    hard_loss = criterion(output.logits, labels)
                if distill_fn is not None:
                    with torch.no_grad():
                        teacher_output = _forward_model(ema.shadow, features, mask, coords)
                    loss = distill_fn(output.logits, teacher_output.logits, hard_loss)
                else:
                    loss = hard_loss
                aux = getattr(output, "aux_loss", None)
                if aux is not None and aux.requires_grad:
                    from .model.ced_mil import CEDMIL as _CEDMIL
                    if isinstance(model, _CEDMIL):
                        loss = loss + aux
                    else:
                        loss = loss + aux * config.moe_aux_weight
                if config.use_rdrop and rdrop_loss_fn is not None:
                    output2 = _forward_model(model, features, mask, coords)
                    loss = loss + rdrop_loss_fn(output.logits, output2.logits)
                    if mixup_labels_b is not None and mixup_lam < 1.0:
                        loss = loss + mixup_lam * criterion(output2.logits, labels) + (1 - mixup_lam) * criterion(output2.logits, mixup_labels_b)
                    else:
                        loss = loss + criterion(output2.logits, labels)
                    loss = loss / 2.0
                if config.use_supcon and supcon_loss_fn is not None and getattr(output, "projection", None) is not None:
                    loss = loss + config.supcon_weight * supcon_loss_fn(output.projection, labels)
                return loss / accum_steps, output
            
        if is_sam:
            loss, output = _compute_loss()
            loss.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if config.gradient_clip_max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_max_norm)
                if config.use_gradient_centralization:
                    GradientCentralization.centralize_gradients(optimizer)
                optimizer.first_step()
                optimizer.zero_grad(set_to_none=True)
                loss2, _ = _compute_loss()
                loss2.backward()
                if config.gradient_clip_max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_max_norm)
                optimizer.second_step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if scheduler is not None:
                    scheduler.step()
        else:
            loss, output = _compute_loss()
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if config.gradient_clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_max_norm)
                if config.use_gradient_centralization:
                    GradientCentralization.centralize_gradients(optimizer)
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                # Only step LR schedule when the optimizer actually updated (GradScaler
                # skips optimizer.step on inf/nan grads; scheduler must not advance then).
                if scheduler is not None and float(scaler.get_scale()) >= scale_before * 0.999:
                    scheduler.step()

        if gradtail is not None:
            gradtail.update(labels.detach().cpu())

        batch_loss = float(loss.item()) * accum_steps
        total_loss += batch_loss * labels.size(0)
        preds = output.logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_count += labels.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())
        all_logits.append(output.logits.detach().cpu())

    preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets_cat = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    nc = config.num_classes or (int(max(preds_cat.max().item(), targets_cat.max().item()) + 1) if preds_cat.numel() else 0)

    return EpochMetrics(
        loss=total_loss / max(total_count, 1),
        accuracy=total_correct / max(total_count, 1),
        macro_f1=_macro_f1(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        balanced_accuracy=_balanced_accuracy(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        auc=_multiclass_auc(all_logits, all_targets, nc) if nc > 0 else 0.0,
        sample_count=total_count,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[SlideFeatureBatch],
    device: str,
    config: TrainConfig | None = None,
) -> EpochMetrics:
    if config is None:
        config = TrainConfig()
    target_device = _resolve_device(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    for batch in loader:
        features = batch.features.to(target_device)
        mask = batch.mask.to(target_device)
        labels = batch.labels.to(target_device)
        coords = batch.coords.to(target_device) if hasattr(batch, "coords") and batch.coords is not None else None
        output = _forward_model(model, features, mask, coords)
        loss = criterion(output.logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = output.logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_count += labels.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())
        all_logits.append(output.logits.detach().cpu())

    preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets_cat = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    nc = config.num_classes or (int(max(preds_cat.max().item(), targets_cat.max().item()) + 1) if preds_cat.numel() else 0)

    return EpochMetrics(
        loss=total_loss / max(total_count, 1),
        accuracy=total_correct / max(total_count, 1),
        macro_f1=_macro_f1(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        balanced_accuracy=_balanced_accuracy(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        auc=_multiclass_auc(all_logits, all_targets, nc) if nc > 0 else 0.0,
        sample_count=total_count,
    )


def evaluate_with_tta(
    model: nn.Module,
    loader: DataLoader[SlideFeatureBatch],
    device: str,
    config: TrainConfig | None = None,
    tta_passes: int = 5,
) -> EpochMetrics:
    """Test-Time Augmentation: multiple forward passes with different dropout/
    feature masks, average softmax probabilities for more robust predictions."""
    if config is None:
        config = TrainConfig()
    target_device = _resolve_device(device)
    criterion = nn.CrossEntropyLoss()

    has_aug = hasattr(model, "feature_aug") and model.feature_aug is not None

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    for batch in loader:
        features = batch.features.to(target_device)
        mask = batch.mask.to(target_device)
        labels = batch.labels.to(target_device)
        coords = batch.coords.to(target_device) if hasattr(batch, "coords") and batch.coords is not None else None

        accum_probs = None
        for t in range(tta_passes):
            if t == 0:
                model.eval()
                with torch.no_grad():
                    output = _forward_model(model, features, mask, coords)
                    probs = F.softmax(output.logits, dim=-1)
            else:
                if has_aug:
                    model.feature_aug.training = True
                model.eval()
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.training = True
                with torch.no_grad():
                    output = _forward_model(model, features, mask, coords)
                    probs = F.softmax(output.logits, dim=-1)
            accum_probs = probs if accum_probs is None else accum_probs + probs

        if has_aug:
            model.feature_aug.training = False
        model.eval()

        avg_probs = accum_probs / tta_passes
        avg_logits = torch.log(avg_probs + 1e-8)
        loss = criterion(avg_logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = avg_probs.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_count += labels.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())
        all_logits.append(avg_logits.detach().cpu())

    preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets_cat = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    nc = config.num_classes or (int(max(preds_cat.max().item(), targets_cat.max().item()) + 1) if preds_cat.numel() else 0)

    return EpochMetrics(
        loss=total_loss / max(total_count, 1),
        accuracy=total_correct / max(total_count, 1),
        macro_f1=_macro_f1(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        balanced_accuracy=_balanced_accuracy(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        auc=_multiclass_auc(all_logits, all_targets, nc) if nc > 0 else 0.0,
        sample_count=total_count,
    )


def predict_probs(
    model: nn.Module,
    loader: DataLoader[SlideFeatureBatch],
    device: str,
    config: TrainConfig | None = None,
    tta_passes: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (probs, labels) tensors for every sample in loader."""
    if config is None:
        config = TrainConfig()
    target_device = _resolve_device(device)
    has_aug = hasattr(model, "feature_aug") and model.feature_aug is not None
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        features = batch.features.to(target_device)
        mask = batch.mask.to(target_device)
        labels = batch.labels.to(target_device)
        coords = batch.coords.to(target_device) if hasattr(batch, "coords") and batch.coords is not None else None

        accum_probs = None
        for t in range(tta_passes):
            if t == 0:
                model.eval()
            else:
                if has_aug:
                    model.feature_aug.training = True
                model.eval()
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.training = True
            with torch.no_grad():
                output = _forward_model(model, features, mask, coords)
                probs = F.softmax(output.logits, dim=-1)
            accum_probs = probs if accum_probs is None else accum_probs + probs

        if has_aug:
            model.feature_aug.training = False
        model.eval()

        avg_probs = accum_probs / tta_passes
        all_probs.append(avg_probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    return torch.cat(all_probs), torch.cat(all_labels)


def model_soup(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Model Soup (Wortsman et al., 2022): uniform average of multiple
    checkpoint state dicts. Produces a single model that captures
    the strengths of all fold-specific models."""
    if not state_dicts:
        raise ValueError("Need at least one state dict")
    if len(state_dicts) == 1:
        return state_dicts[0]
    avg: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return avg


@torch.no_grad()
def evaluate_ensemble(
    models: list[nn.Module],
    loader: DataLoader[SlideFeatureBatch],
    device: str,
    config: TrainConfig | None = None,
) -> EpochMetrics:
    """Cross-fold ensemble: average softmax probabilities from multiple
    independently trained models for more robust predictions."""
    if config is None:
        config = TrainConfig()
    target_device = _resolve_device(device)
    criterion = nn.CrossEntropyLoss()

    for m in models:
        m.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    for batch in loader:
        features = batch.features.to(target_device)
        mask = batch.mask.to(target_device)
        labels = batch.labels.to(target_device)
        coords = batch.coords.to(target_device) if hasattr(batch, "coords") and batch.coords is not None else None

        avg_probs = None
        for m in models:
            output = _forward_model(m, features, mask, coords)
            probs = F.softmax(output.logits, dim=-1)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        avg_probs = avg_probs / len(models)

        ens_logits = torch.log(avg_probs + 1e-8)
        loss = criterion(ens_logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = avg_probs.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_count += labels.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())
        all_logits.append(ens_logits.detach().cpu())

    preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets_cat = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    nc = config.num_classes or (int(max(preds_cat.max().item(), targets_cat.max().item()) + 1) if preds_cat.numel() else 0)

    return EpochMetrics(
        loss=total_loss / max(total_count, 1),
        accuracy=total_correct / max(total_count, 1),
        macro_f1=_macro_f1(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        balanced_accuracy=_balanced_accuracy(preds_cat, targets_cat, nc) if nc > 0 else 0.0,
        auc=_multiclass_auc(all_logits, all_targets, nc) if nc > 0 else 0.0,
        sample_count=total_count,
    )
