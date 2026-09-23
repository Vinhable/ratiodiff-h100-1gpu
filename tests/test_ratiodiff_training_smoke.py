"""Exercise the REAL trainer on tiny local Diffusers models; no downloads/GPU.

This verifies plumbing/checkpoints, not CUDA fit or SDXL quality/performance.
"""
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch


def tiny_pipeline(directory, family):
    from diffusers import (AutoencoderKL, UNet2DConditionModel, DDPMScheduler,
                          StableDiffusionPipeline, StableDiffusionXLPipeline)
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    vocab = directory/"tokenizer_source"
    vocab.mkdir(parents=True)
    (vocab/"vocab.json").write_text(json.dumps({"<|startoftext|>":0, "<|endoftext|>":1}))
    (vocab/"merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(vocab/"vocab.json"), str(vocab/"merges.txt"), model_max_length=8)
    text_config = CLIPTextConfig(vocab_size=2, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2,
                                max_position_embeddings=8, projection_dim=16,
                                bos_token_id=0, eos_token_id=1, pad_token_id=1)
    text = CLIPTextModel(text_config)
    vae = AutoencoderKL(block_out_channels=(16,32), in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",)*2, up_block_types=("UpDecoderBlock2D",)*2,
        latent_channels=4, norm_num_groups=8, sample_size=32)
    extras = dict(addition_embed_type="text_time", addition_time_embed_dim=8,
                  projection_class_embeddings_input_dim=64) if family == "sdxl" else {}
    unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(16,32),
        norm_num_groups=8, layers_per_block=1, attention_head_dim=2,
        cross_attention_dim=32 if family == "sdxl" else 16, **extras)
    scheduler = DDPMScheduler(num_train_timesteps=30)
    if family == "sdxl":
        pipe = StableDiffusionXLPipeline(vae=vae, text_encoder=text,
            text_encoder_2=CLIPTextModelWithProjection(text_config), tokenizer=tokenizer,
            tokenizer_2=tokenizer, unet=unet, scheduler=scheduler)
    else:
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False)
    pipe.save_pretrained(directory/"model")
    vae.save_pretrained(directory/"vae")
    return directory/"model", directory/"vae"


@pytest.mark.parametrize("family", ["sdxl", "sd15"])
def test_tiny_training_and_resume(tmp_path, monkeypatch, family):
    import accelerate
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from safetensors.torch import load_file
    from hessian import train_ratiodiff_sdxl as trainer
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, vae = tiny_pipeline(tmp_path, family)
    payloads = []
    for color in ("red", "blue"):
        f = io.BytesIO(); Image.new("RGB", (40,32), color).save(f, format="PNG")
        payloads.append(f.getvalue())
    shard = "train.parquet"
    pq.write_table(pa.table(dict(jpg_0=[payloads[0]]*8, jpg_1=[payloads[1]]*8,
                                label_0=[1.]*8, caption=["test"]*8)), tmp_path/shard)
    manifest = dict(repo_id="local/test", files=[shard], target_rows=8,
                    file_rows={shard:8}, valid_row_indices={shard:list(range(8))})
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    config_name = "ratiodiff_xl_h100_1gpu.json" if family == "sdxl" else "ratiodiff_sd15_h100_1gpu.json"
    c = trainer.load_config(Path(trainer.__file__).with_name(config_name))
    c.update(model=str(model), vae=str(vae), samples=8, resolution=32,
        effective_batch=2, micro_batch=1, epochs=1, warmup_steps=0,
        calibration_start=2, calibration_end=3, checkpoint_steps=2, log_every=1,
        learning_rate=.003, data_dir=str(tmp_path), manifest=str(tmp_path/"manifest.json"),
        work_root=str(tmp_path/"runtime"))
    c["eval"]["resolution"] = 32
    config_file = tmp_path/"config.json"; config_file.write_text(json.dumps(c))
    real_accelerator = accelerate.Accelerator
    monkeypatch.setattr(accelerate, "Accelerator", lambda **kw: real_accelerator(cpu=True, **kw))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setenv("WORLD_SIZE", "1")
    def run(variant, resume=None):
        argv = ["train", "--config", str(config_file), "--variant", variant]
        if resume:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", argv)
        trainer.main()
    run("full")
    out = Path(c["work_root"])/"full/runs/full"
    complete = json.loads((out/"completed.json").read_text())
    assert complete["step"] == 4
    cal = json.loads((out/"calibration.json").read_text())
    assert cal["fixed_ref"] > 0
    assert (cal["fixed_win"] > 0) == (family == "sd15")
    original = {k: v.clone() for k, v in load_file(str(out/"unet/diffusion_pytorch_model.safetensors")).items()}
    # Simulate process loss before final export using this isolated test directory.
    (out/"completed.json").unlink()
    run("full", out/"checkpoint-2")
    resumed = load_file(str(out/"unet/diffusion_pytorch_model.safetensors"))
    for key in original:
        torch.testing.assert_close(original[key], resumed[key], rtol=0, atol=0)
    if family == "sd15":
        run("no_ratio_correction")
        run("fixed_regularization")
        resolved = json.loads((out.parent/"fixed_regularization/resolved_config.json").read_text())
        assert resolved["fixed_ref"] == cal["fixed_ref"]
        assert resolved["fixed_win"] == cal["fixed_win"]
