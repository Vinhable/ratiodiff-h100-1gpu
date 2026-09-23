import io
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from ratiodiff_sdxl import (LossConfig, Calibration, ConfidenceHead, as_epsilon,
    confidence_features, confidence_loss, denoiser_loss, normalized_weights, sum_sq, vp_omega)
from hessian.train_ratiodiff_sdxl import image_tensor, validate_config, load_config


def inputs(initial=False):
    torch.manual_seed(4)
    rw, rl, nw, nl = [torch.randn(3, 4, 3, 3) for _ in range(4)]
    pw = (rw + (0 if initial else .02 * torch.randn_like(rw))).requires_grad_()
    pl = (rl + (0 if initial else .02 * torch.randn_like(rl))).requires_grad_()
    return pw, pl, rw, rl, nw, nl, torch.tensor([.1, .2, .3]), torch.tensor([.5, 1., 1.5])


@pytest.mark.parametrize("variant", ["full", "no_ratio_correction", "fixed_regularization"])
def test_init(variant):
    xs = inputs(True)
    loss, s = denoiser_loss(*xs, LossConfig(), variant, .2)
    assert torch.equal(s["margin"], torch.zeros(3))
    assert loss.item() == 0
    assert s["G_ref"].sum() == 0
    grad = torch.autograd.grad(loss, xs[:2])
    assert sum(x.abs().sum() for x in grad) > 0
    assert torch.isfinite(s["a_ref"]).all()


def test_init_grad_match():
    grads = []
    for variant in ("full", "no_ratio_correction"):
        xs = inputs(True)
        loss, _ = denoiser_loss(*xs, LossConfig(), variant)
        grads.append(torch.autograd.grad(loss, xs[:2]))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("variant", ["full", "no_ratio_correction"])
def test_margin_identity_swap_norm_and_budget(variant):
    xs = inputs(); pw, pl, rw, rl, nw, nl, om, weights = xs
    c = LossConfig()
    _, s = denoiser_loss(*xs, c, variant)
    expanded = om * (sum_sq(nw-pw)-sum_sq(nw-rw)-sum_sq(nl-pl)+sum_sq(nl-rl))
    if variant == "full":
        expanded += om * (sum_sq(pl-rl)-sum_sq(pw-rw))
    torch.testing.assert_close(s["margin"], expanded, atol=5e-6, rtol=2e-4)
    _, swapped = denoiser_loss(pl, pw, rl, rw, nl, nw, om, weights, c, variant)
    torch.testing.assert_close(swapped["margin"], -s["margin"])
    pref, _ = denoiser_loss(*xs, replace(c, rho_ref=0), variant)
    grads = torch.autograd.grad(pref * 3, (pw, pl), retain_graph=True)
    norm = (sum_sq(grads[0])+sum_sq(grads[1])).sqrt()
    torch.testing.assert_close(norm, s["G_pref"])
    ref = (om*(sum_sq(pw-rw)+sum_sq(pl-rl))).sum()
    grads = torch.autograd.grad(ref, (pw, pl))
    torch.testing.assert_close((sum_sq(grads[0])+sum_sq(grads[1])).sqrt(), s["G_ref"])
    assert (s["ref_budget_used"] <= c.rho_ref*s["G_pref"]+1e-7).all()


def test_basu_slope_and_sign():
    m = torch.tensor([-.3, 0, .8], requires_grad=True)
    c = LossConfig(); z = c.beta_R*m
    loss = c.gamma_R/(2*c.beta_R)*(torch.expm1((1+c.lambda_B)*z)/(1+c.lambda_B)
                                  -torch.expm1(-c.lambda_B*z)/c.lambda_B)
    slope, = torch.autograd.grad(loss.sum(), m)
    torch.testing.assert_close(slope, c.gamma_R/2*(torch.exp((1+c.lambda_B)*z)+torch.exp(-c.lambda_B*z)))
    pw, pl, rw, rl, nw, nl, om, w = inputs(True)
    _, s = denoiser_loss(rw+.01*(nw-rw), rl-.01*(nl-rl), rw, rl, nw, nl, om, w, c)
    assert (s["margin"] < 0).all()


def test_detach_and_confidence():
    xs = list(inputs()); xs[2].requires_grad_(); xs[3].requires_grad_()
    head = ConfidenceHead()
    features = confidence_features(xs[0], xs[1], xs[2], xs[3], torch.tensor([1,2,3]), 1000)
    assert features.shape == (3, 65) and not features.requires_grad
    logits = head(features)
    xs[-1] = normalized_weights(logits, .05)
    loss, s = denoiser_loss(*xs, LossConfig())
    loss.backward()
    assert all(p.grad is None for p in head.parameters())
    assert xs[2].grad is None and xs[3].grad is None
    assert not s["a_ref"].requires_grad
    xs[0].grad = None; xs[1].grad = None
    confidence_loss(logits, xs[2], xs[3], xs[4], xs[5], LossConfig()).backward()
    assert any(p.grad is not None for p in head.parameters())
    assert xs[0].grad is None and xs[1].grad is None


def test_fixed_calibration_and_no_adaptivity():
    c = Calibration()
    c.update(dict(a_ref=torch.tensor([1., 4.]), G_ref=torch.tensor([3., 2.])))
    assert c.coefficient() == 11/5
    assert math.isclose(c.coefficient()*c.denominator, c.numerator)
    clone = Calibration(); clone.load_state_dict(c.state_dict())
    assert clone.coefficient() == c.coefficient()
    for initial in (True, False):
        _, s = denoiser_loss(*inputs(initial), LossConfig(a_max=.01), "fixed_regularization", c.coefficient())
        torch.testing.assert_close(s["a_ref"], torch.full((3,), 2.2))
    with pytest.raises(ValueError):
        Calibration().coefficient()
    with pytest.raises(ValueError):
        denoiser_loss(*inputs(), LossConfig(), "fixed_regularization")


def test_schedule_no_endpoint_floor_and_conversion():
    ab = torch.cumprod(1-torch.linspace(.00085, .012, 1000, dtype=torch.float64), 0)
    omega, valid = vp_omega(ab)
    assert 0 not in valid and len(valid) == 999
    prev = ab[:-1]; a = ab[1:]/prev; beta = 1-a
    v = beta*(1-prev)/(1-ab[1:])
    b = beta/a.sqrt()/(1-ab[1:]).sqrt()
    torch.testing.assert_close(omega[1:].double(), b.square()/(2*v), rtol=1e-6, atol=1e-8)
    eps, x0 = torch.randn(3,4,3,3), torch.randn(3,4,3,3)
    t = torch.tensor([1,25,999]); alpha = ab[t].float().sqrt().reshape(-1,1,1,1)
    sigma = (1-ab[t]).float().sqrt().reshape(-1,1,1,1)
    xt = alpha*x0+sigma*eps
    torch.testing.assert_close(as_epsilon(alpha*eps-sigma*x0, xt, t, ab, "v_prediction"), eps)
    torch.testing.assert_close(as_epsilon(x0, xt, t, ab, "sample"), eps, atol=5e-6, rtol=1e-5)


def test_overflow_is_error_and_no_zero_lambda():
    with pytest.raises(ValueError):
        LossConfig(lambda_B=0)
    with pytest.raises(FloatingPointError):
        denoiser_loss(*inputs(), LossConfig(beta_R=1e9))


def test_real_crop_metadata():
    from PIL import Image
    f = io.BytesIO(); Image.new("RGB", (80, 40)).save(f, format="PNG")
    pixels, ids = image_tensor(f.getvalue(), 32)
    assert pixels.shape == (3,32,32)
    assert ids.tolist() == [40,80,0,16,32,32]


def test_batch_plan_and_pilot_is_separate():
    path = Path(__file__).resolve().parents[1]/"hessian/ratiodiff_xl_h100_1gpu.json"
    c = load_config(path)
    assert validate_config(c, 1) == (1329, 16, 1329)
    assert validate_config(load_config(path, True), 1) == (8, 16, 1329)
    broken = dict(c, micro_batch=3)
    with pytest.raises(ValueError):
        validate_config(broken, 1)


@pytest.mark.parametrize("variant", ["full", "no_ratio_correction"])
def test_sd15_winner_norm_and_both_budgets(variant):
    xs = inputs()
    cfg = LossConfig(winner_enabled=True, rho_win=.25, beta_D=.01, mu=.25)
    loss, stats = denoiser_loss(*xs, cfg, variant)
    pw, pl, rw, rl, nw, nl, om, w = xs
    # Freeze the gate and coefficients, as required by the stop-gradient objective.
    residual = (1-stats["kappa"]).reshape(-1,1,1,1)*(pw-rw)-(nw-rw)
    grad, = torch.autograd.grad(sum_sq(residual).sum(), pw, retain_graph=True)
    torch.testing.assert_close(sum_sq(grad).sqrt(), stats["G_win"])
    assert (stats["win_budget_used"] <= cfg.rho_win*stats["G_pref"]+1e-7).all()
    assert (stats["ref_budget_used"] <= cfg.rho_ref*stats["G_pref"]+1e-7).all()
    assert not stats["kappa"].requires_grad
    loss.backward()
    assert pw.grad is not None


def test_sd15_fixed_coefficients_and_calibration():
    cfg = LossConfig(winner_enabled=True, rho_win=.25)
    _, s = denoiser_loss(*inputs(), cfg)
    ref_cal, win_cal = Calibration(), Calibration()
    ref_cal.update(s); win_cal.update(s, "win")
    for cal in (ref_cal, win_cal):
        assert math.isclose(cal.coefficient()*cal.denominator, cal.numerator)
    for initial in (True, False):
        _, s = denoiser_loss(*inputs(initial), cfg, "fixed_regularization",
                             ref_cal.coefficient(), win_cal.coefficient())
        torch.testing.assert_close(s["a_win"], torch.full((3,), win_cal.coefficient()))
    with pytest.raises(ValueError):
        denoiser_loss(*inputs(), cfg, "fixed_regularization", 1.)


def test_xl_winner_truly_disabled():
    for variant in ("full", "no_ratio_correction", "fixed_regularization"):
        _, s = denoiser_loss(*inputs(), LossConfig(), variant, .2)
        for name in ("win_loss", "G_win", "a_win", "kappa", "win_budget_used"):
            assert torch.equal(s[name], torch.zeros(3))


def test_family_configs_disallow_cross_backbone_matrix():
    parent = Path(__file__).resolve().parents[1]/"hessian"
    sd15 = load_config(parent/"ratiodiff_sd15_h100_1gpu.json")
    assert validate_config(sd15, 1) == (1329, 4, 1329)
    assert sd15["loss"]["winner_enabled"]
    xl = load_config(parent/"ratiodiff_xl_h100_1gpu.json")
    assert xl["variants"] == ["full"]
    xl["variants"].append("no_ratio_correction")
    with pytest.raises(ValueError):
        validate_config(xl, 1)
    sd15["loss"]["winner_enabled"] = False
    with pytest.raises(ValueError):
        validate_config(sd15, 1)


def test_data_preflight_and_rank_resume(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from hessian.train_ratiodiff_sdxl import check_data
    from hessian.streaming_pickapic import PickAPicStreamingDataset
    filename = "train-00000.parquet"
    rows = dict(jpg_0=[b"x"]*10, jpg_1=[b"x"]*10, label_0=[1.]*10,
                caption=[str(i) for i in range(10)])
    pq.write_table(pa.table(rows), tmp_path/filename)
    manifest = dict(repo_id="test/local", target_rows=10, files=[filename],
                    file_rows={filename:10}, valid_row_indices={filename:list(range(10))})
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(manifest))
    c = dict(manifest=str(path), data_dir=str(tmp_path), samples=10)
    assert check_data(c)["target_rows"] == 10
    seen = []
    for rank in range(2):
        kwargs = dict(manifest_path=path, local_data_dir=tmp_path, stream_cache_root=tmp_path/"cache",
                      rank=rank, world_size=2, seed=42, pad_to_multiple=4)
        all_rows = list(PickAPicStreamingDataset(**kwargs))
        resumed = list(PickAPicStreamingDataset(**kwargs, start_sample=4))
        assert resumed == all_rows[4:]
        seen.extend(r["caption"] for r in all_rows[:5])
    assert len(seen) == len(set(seen)) == 10
    c["samples"] = 11
    with pytest.raises(ValueError):
        check_data(c)


def test_atomic_writes_parallel(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hessian.train_ratiodiff_sdxl import atomic_json
    path = tmp_path/"protocol.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: atomic_json(path, {"x":i}), range(30)))
    assert json.loads(path.read_text())["x"] in range(30)
    assert not list(tmp_path.glob("*.tmp"))
