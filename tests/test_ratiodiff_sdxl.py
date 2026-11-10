import io
import json
import math
import sys
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


def test_initialization_and_gradient():
    xs = inputs(True)
    loss, stats = denoiser_loss(*xs, LossConfig())
    assert torch.equal(stats["margin"], torch.zeros(3))
    assert loss.item() == 0
    assert stats["G_ref"].sum() == 0
    gradients = torch.autograd.grad(loss, xs[:2])
    assert sum(value.abs().sum() for value in gradients) > 0
    assert torch.isfinite(stats["a_ref"]).all()


def test_margin_identity_swap_norm_and_budget():
    xs = inputs(); pw, pl, rw, rl, nw, nl, omega, weights = xs
    config = LossConfig()
    _, stats = denoiser_loss(*xs, config)
    expanded = omega * (sum_sq(nw-pw)-sum_sq(nw-rw)-sum_sq(nl-pl)+sum_sq(nl-rl))
    expanded += omega * (sum_sq(pl-rl)-sum_sq(pw-rw))
    torch.testing.assert_close(stats["margin"], expanded, atol=5e-6, rtol=2e-4)
    _, swapped = denoiser_loss(pl, pw, rl, rw, nl, nw, omega, weights, config)
    torch.testing.assert_close(swapped["margin"], -stats["margin"])
    preference, _ = denoiser_loss(*xs, LossConfig(rho_ref=0))
    gradients = torch.autograd.grad(preference * 3, (pw, pl), retain_graph=True)
    norm = (sum_sq(gradients[0]) + sum_sq(gradients[1])).sqrt()
    torch.testing.assert_close(norm, stats["G_pref"])
    assert (stats["ref_budget_used"] <= config.rho_ref*stats["G_pref"]+1e-7).all()


def test_basu_slope_and_sign():
    margin = torch.tensor([-.3, 0, .8], requires_grad=True)
    config = LossConfig(); z = config.beta_R*margin
    loss = config.gamma_R/(2*config.beta_R)*(
        torch.expm1((1+config.lambda_B)*z)/(1+config.lambda_B)
        - torch.expm1(-config.lambda_B*z)/config.lambda_B)
    slope, = torch.autograd.grad(loss.sum(), margin)
    expected = config.gamma_R/2*(torch.exp((1+config.lambda_B)*z)+torch.exp(-config.lambda_B*z))
    torch.testing.assert_close(slope, expected)
    pw, pl, rw, rl, nw, nl, omega, weights = inputs(True)
    _, stats = denoiser_loss(rw+.01*(nw-rw), rl-.01*(nl-rl), rw, rl, nw, nl,
                             omega, weights, config)
    assert (stats["margin"] < 0).all()


def test_detach_and_confidence():
    xs = list(inputs()); xs[2].requires_grad_(); xs[3].requires_grad_()
    head = ConfidenceHead()
    features = confidence_features(xs[0], xs[1], xs[2], xs[3], torch.tensor([1,2,3]), 1000)
    assert features.shape == (3, 65) and not features.requires_grad
    logits = head(features)
    xs[-1] = normalized_weights(logits, .05)
    loss, stats = denoiser_loss(*xs, LossConfig())
    loss.backward()
    assert all(parameter.grad is None for parameter in head.parameters())
    assert xs[2].grad is None and xs[3].grad is None and not stats["a_ref"].requires_grad
    xs[0].grad = None; xs[1].grad = None
    confidence_loss(logits, xs[2], xs[3], xs[4], xs[5], LossConfig()).backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_calibration_accumulator():
    calibration = Calibration()
    calibration.update(dict(a_ref=torch.tensor([1., 4.]), G_ref=torch.tensor([3., 2.])))
    assert calibration.coefficient() == 11/5
    assert math.isclose(calibration.coefficient()*calibration.denominator, calibration.numerator)
    clone = Calibration(); clone.load_state_dict(calibration.state_dict())
    assert clone.coefficient() == calibration.coefficient()
    with pytest.raises(ValueError):
        Calibration().coefficient()


def test_schedule_no_endpoint_floor_and_conversion():
    ab = torch.cumprod(1-torch.linspace(.00085, .012, 1000, dtype=torch.float64), 0)
    omega, valid = vp_omega(ab)
    assert 0 not in valid and len(valid) == 999
    previous = ab[:-1]; alpha = ab[1:]/previous; beta = 1-alpha
    variance = beta*(1-previous)/(1-ab[1:])
    coefficient = beta/alpha.sqrt()/(1-ab[1:]).sqrt()
    torch.testing.assert_close(omega[1:].double(), coefficient.square()/(2*variance), rtol=1e-6, atol=1e-8)
    epsilon, x0 = torch.randn(3,4,3,3), torch.randn(3,4,3,3)
    timesteps = torch.tensor([1,25,999]); a = ab[timesteps].float().sqrt().reshape(-1,1,1,1)
    sigma = (1-ab[timesteps]).float().sqrt().reshape(-1,1,1,1)
    noisy = a*x0+sigma*epsilon
    torch.testing.assert_close(as_epsilon(a*epsilon-sigma*x0, noisy, timesteps, ab, "v_prediction"), epsilon)
    torch.testing.assert_close(as_epsilon(x0, noisy, timesteps, ab, "sample"), epsilon, atol=5e-6, rtol=1e-5)


def test_invalid_loss_scales_fail():
    with pytest.raises(ValueError):
        LossConfig(lambda_B=0)
    with pytest.raises(FloatingPointError):
        denoiser_loss(*inputs(), LossConfig(beta_R=1e9))


def test_real_crop_metadata():
    from PIL import Image
    handle = io.BytesIO(); Image.new("RGB", (80, 40)).save(handle, format="PNG")
    pixels, ids = image_tensor(handle.getvalue(), 32)
    assert pixels.shape == (3,32,32)
    assert ids.tolist() == [40,80,0,16,32,32]


def test_sdxl_config_and_batch_plan():
    path = Path(__file__).resolve().parents[1]/"hessian/ratiodiff_xl_h100_1gpu.json"
    config = load_config(path)
    assert validate_config(config, 1) == (1329, 16, 1329)
    assert validate_config(load_config(path, True), 1) == (8, 16, 1329)
    for mutation in (
        dict(model_family="unsupported"),
        dict(variants=["full", "experimental"]),
        dict(micro_batch=3),
    ):
        with pytest.raises(ValueError):
            validate_config(dict(config, **mutation), 1)
    config["loss"]["unexpected"] = True
    with pytest.raises(ValueError):
        validate_config(config, 1)


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
    config = dict(manifest=str(path), data_dir=str(tmp_path), samples=10)
    assert check_data(config)["target_rows"] == 10
    seen = []
    for rank in range(2):
        kwargs = dict(manifest_path=path, local_data_dir=tmp_path, stream_cache_root=tmp_path/"cache",
                      rank=rank, world_size=2, seed=42, pad_to_multiple=4)
        all_rows = list(PickAPicStreamingDataset(**kwargs))
        assert list(PickAPicStreamingDataset(**kwargs, start_sample=4)) == all_rows[4:]
        seen.extend(row["caption"] for row in all_rows[:5])
    assert len(seen) == len(set(seen)) == 10
    config["samples"] = 11
    with pytest.raises(ValueError):
        check_data(config)


def test_atomic_writes_parallel(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hessian.train_ratiodiff_sdxl import atomic_json
    path = tmp_path/"protocol.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda value: atomic_json(path, {"x":value}), range(30)))
    assert json.loads(path.read_text())["x"] in range(30)
    assert not list(tmp_path.glob("*.tmp"))
