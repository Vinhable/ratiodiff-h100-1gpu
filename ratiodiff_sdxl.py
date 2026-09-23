"""RatioDiff specification (2026-09-22), independent of legacy Q3 losses.

Sum over latent dimensions, mean over pairs. SDXL disables winner auxiliary;
SD1.5 ablations retain it. No log-ratio clamp, variance-floor endpoint, or
gradient through confidence weights/budgets.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


VARIANTS = ("full", "no_ratio_correction", "fixed_regularization")


@dataclass(frozen=True)
class LossConfig:
    # Starting values ONLY: validate/tune on training pilot, not test scores.
    beta_R: float = 0.001
    gamma_R: float = 0.01
    lambda_B: float = 0.5
    rho_ref: float = 0.1
    a_max: float = 100.0
    eps_g: float = 1e-8
    w_min: float = 0.05
    label_smoothing: float = 0.05
    lambda_conf: float = 1.0
    winner_enabled: bool = False
    rho_win: float = 0.0
    beta_D: float = 0.01
    T_D: float = 1.0
    mu: float = 0.25

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Non-finite loss configuration")
        if min(self.beta_R, self.gamma_R, self.lambda_B, self.a_max,
               self.eps_g, self.lambda_conf) <= 0:
            raise ValueError("Loss scales must be positive (lambda_B must NOT be zero)")
        if not (0 <= self.rho_ref < 1 and 0 < self.w_min < 1
                and 0 <= self.label_smoothing < 1):
            raise ValueError("Invalid budget/weight/smoothing")
        if self.rho_win < 0 or not 0 <= self.mu < 1 or min(self.beta_D, self.T_D) <= 0:
            raise ValueError("Invalid winner auxiliary configuration")
        if not self.winner_enabled and self.rho_win != 0:
            raise ValueError("Disabled winner auxiliary requires rho_win=0")


def sum_sq(x):
    return x.square().flatten(1).sum(1)


def sum_dot(x, y):
    return (x * y).flatten(1).sum(1)


def vp_omega(alphas_cumprod):
    """Training index zero is degenerate and excluded, never epsilon-repaired."""
    ab = torch.as_tensor(alphas_cumprod, dtype=torch.float64)
    prev = torch.cat([torch.ones_like(ab[:1]), ab[:-1]])
    a = ab / prev
    beta = 1 - a
    variance = beta * (1 - prev) / (1 - ab)
    omega = beta / (2 * a * (1 - prev))
    valid = ((variance > 0) & (ab > 0) & (ab < 1) & (a > 0)
             & torch.isfinite(omega) & (omega > 0))
    indices = valid.nonzero().flatten()
    if not len(indices):
        raise ValueError("No nondegenerate training timesteps")
    return omega.float(), indices


def as_epsilon(pred, noisy, timesteps, alphas_cumprod, prediction_type):
    pred, noisy = pred.float(), noisy.float()
    ab = alphas_cumprod.to(pred.device)[timesteps].float()
    shape = (-1,) + (1,) * (pred.ndim - 1)
    alpha, sigma = ab.sqrt().reshape(shape), (1 - ab).sqrt().reshape(shape)
    if prediction_type == "epsilon":
        return pred
    if prediction_type == "v_prediction":
        return alpha * pred + sigma * noisy
    if prediction_type == "sample":
        return (noisy - alpha * pred) / sigma
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


class ConfidenceHead(nn.Module):
    def __init__(self, channels=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16 * channels + 1, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.net(features.detach().float()).flatten()


@torch.no_grad()
def confidence_features(noisy_w, noisy_l, ref_w, ref_l, t, num_timesteps):
    def summary(x, ref):
        f = torch.cat([x.detach().float(), ref.detach().float()], 1)
        return torch.cat([f.mean((-2, -1)), f.std((-2, -1), unbiased=False)], 1)
    uw, ul = summary(noisy_w, ref_w), summary(noisy_l, ref_l)
    tau = t.float().reshape(-1, 1) / (num_timesteps - 1)
    return torch.cat([uw, ul, uw - ul, (uw - ul).abs(), tau], 1)


@torch.no_grad()
def normalized_weights(logits, w_min, global_mean=True):
    raw = w_min + (1 - w_min) * logits.detach().float().sigmoid()
    totals = torch.stack([raw.sum(), raw.new_tensor(raw.numel())])
    if global_mean and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
    return raw / (totals[0] / totals[1])


def denoiser_loss(pw, pl, rw, rl, nw, nl, omega, weights, config,
                  variant="full", fixed_ref=None, fixed_win=None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    c = config
    pw, pl = pw.float(), pl.float()
    rw, rl = rw.detach().float(), rl.detach().float()
    ew, el = nw.detach().float() - rw, nl.detach().float() - rl
    dw, dl = pw - rw, pl - rl
    om, w = omega.detach().float().flatten(), weights.detach().float().flatten()
    if om.shape != w.shape or w.shape != (pw.shape[0],):
        raise ValueError("One omega and weight required per pair")
    if not (torch.isfinite(om).all() and torch.isfinite(w).all()
            and (om > 0).all() and (w > 0).all()):
        raise ValueError("Invalid timestep coefficient or weight")
    full_margin = 2 * om * (sum_dot(el, dl) - sum_dot(ew, dw))
    no_corr_margin = full_margin + om * (sum_sq(dw) - sum_sq(dl))
    margin = no_corr_margin if variant == "no_ratio_correction" else full_margin
    norm2 = (sum_sq(dw - ew) + sum_sq(dl - el) if variant == "no_ratio_correction"
             else sum_sq(ew) + sum_sq(el))
    z = c.beta_R * margin
    pos, neg = (1 + c.lambda_B) * z, -c.lambda_B * z
    pref = w * c.gamma_R / (2 * c.beta_R) * (
        torch.expm1(pos) / (1 + c.lambda_B) - torch.expm1(neg) / c.lambda_B)
    displacement2 = sum_sq(dw) + sum_sq(dl)
    ref_loss = om * displacement2
    if c.winner_enabled:
        with torch.no_grad():
            gap_theta = sum_sq(ew-dw) - sum_sq(el-dl)
            gap_ref = sum_sq(ew) - sum_sq(el)
            p_D = torch.sigmoid(-c.beta_D / (2*c.T_D) * (gap_theta-gap_ref))
            kappa = c.mu * (1-p_D)
        shape = (-1,) + (1,) * (dw.ndim-1)
        win_loss = sum_sq((1-kappa).reshape(shape)*dw-ew)
    else:
        if fixed_win not in (None, 0):
            raise ValueError("Disabled winner auxiliary requires fixed_win=0")
        kappa, win_loss = torch.zeros_like(om), torch.zeros_like(om)
    with torch.no_grad():
        gp = w * c.gamma_R * om * (pos.detach().exp() + neg.detach().exp()) * norm2.detach().sqrt()
        gr = 2 * om * displacement2.detach().sqrt()
        gw = 2 * (1-kappa) * win_loss.detach().sqrt()
        if variant == "fixed_regularization":
            if fixed_ref is None or not math.isfinite(fixed_ref) or fixed_ref < 0:
                raise ValueError("Provide a calibrated finite fixed_ref")
            ar = torch.full_like(om, fixed_ref)
            if c.winner_enabled:
                if fixed_win is None or not math.isfinite(fixed_win) or fixed_win < 0:
                    raise ValueError("Provide a calibrated finite fixed_win")
                aw = torch.full_like(om, fixed_win)
            else:
                aw = torch.zeros_like(om)
        else:
            ar = (c.rho_ref * gp / gr.clamp_min(c.eps_g)).clamp_max(c.a_max)
            aw = ((c.rho_win * gp / gw.clamp_min(c.eps_g)).clamp_max(c.a_max)
                  if c.winner_enabled else torch.zeros_like(om))
    reg = ar * ref_loss + aw * win_loss
    total = pref + reg
    if not all(torch.isfinite(v).all().item() for v in (total, gp, gr, ar, gw, aw)):
        raise FloatingPointError("Non-finite RatioDiff objective/budget: inspect log_ratio and scales; no clamp applied")
    stats = dict(margin=margin, margin_full=full_margin, margin_no_correction=no_corr_margin,
                 log_ratio=z, exp_arg_pos=pos, exp_arg_neg=neg, pref=pref, ref_loss=ref_loss,
                 G_pref=gp, G_ref=gr, a_ref=ar, ref_budget_used=ar * gr,
                 G_win=gw, a_win=aw, win_loss=win_loss, kappa=kappa,
                 win_budget_used=aw*gw,
                 win_budget_violation=(aw*gw > c.rho_win*gp+1e-6).float(),
                 displacement2=displacement2, weight=w,
                 winner_error_delta=sum_sq(ew - dw) - sum_sq(ew),
                 loser_error_delta=sum_sq(el - dl) - sum_sq(el),
                 budget_fraction=ar * gr / gp.clamp_min(c.eps_g),
                 budget_violation=(ar * gr > c.rho_ref * gp + 1e-6).float(),
                 cap_hit=(ar >= c.a_max).float())
    return total.mean(), {k: v.detach() for k, v in stats.items()}


def confidence_loss(logits, ref_w, ref_l, noise_w, noise_l, config):
    with torch.no_grad():
        ew = sum_sq(noise_w.detach().float() - ref_w.detach().float())
        el = sum_sq(noise_l.detach().float() - ref_l.detach().float())
        y = (ew < el).float() + 0.5 * (ew == el).float()
        target = (1 - config.label_smoothing) * y + config.label_smoothing / 2
    return F.binary_cross_entropy_with_logits(logits.float(), target)


class Calibration:
    """FP64 sum(a_i G_i)/sum(G_i); register with Accelerate for resume."""
    def __init__(self):
        self.numerator = 0.0
        self.denominator = 0.0
        self.count = 0

    def update(self, stats, term="ref"):
        a, g = stats[f"a_{term}"].double(), stats[f"G_{term}"].double()
        values = torch.stack([(a * g).sum(), g.sum(), g.new_tensor(g.numel())])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        n, d, count = values.cpu().tolist()
        self.numerator += n
        self.denominator += d
        self.count += int(count)

    def coefficient(self):
        if self.denominator <= 0 or not math.isfinite(self.numerator + self.denominator):
            raise ValueError("Calibration has no finite nonzero auxiliary gradient")
        return self.numerator / self.denominator

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        self.__dict__.update(state)
