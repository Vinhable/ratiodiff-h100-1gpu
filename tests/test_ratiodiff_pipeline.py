import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from hessian.evaluate_ratiodiff_sdxl import prepare_prompts, validate_scores, DATASETS, METRICS
from hessian.evaluate_standalone import _load_evaluator, merge_scores
from hessian.prepare_ratiodiff_experiments import extract_prompts
from hessian.train_ratiodiff_sdxl import atomic_json


def test_plan_is_backbone_specific():
    root = Path(__file__).resolve().parents[1]/"hessian"
    for filename, family, count in (("run_ratiodiff_sdxl_matrix.py", "sdxl", 1),
                                     ("run_ratiodiff_sd15_ablations.py", "sd15", 3)):
        result = subprocess.run([sys.executable, "-B", str(root/filename)], check=True,
                                capture_output=True, text=True)
        plan = json.loads(result.stdout)
        assert plan["model_family"] == family and len(plan["variants"]) == count
        assert plan["optimizer_steps_per_run"] == 1329
        assert "PLAN ONLY" in plan["mode"]


def test_eval_coverage_report_and_prompt_recovery(tmp_path):
    from PIL import Image
    project = Path(__file__).resolve().parents[1]
    module = _load_evaluator(project, tmp_path/"runtime")
    source = tmp_path/"source"
    for dataset in DATASETS:
        atomic_json(source/f"{dataset}.json", [dict(id=i, prompt=f"test {i}") for i in range(2)])
    config = dict(prompt_dir=str(source))
    provenance = prepare_prompts(module, config, "full", False)
    assert len(provenance) == 3
    for dataset in DATASETS:
        for label in ("base", "checkpoint-4"):
            for i in range(2):
                path = module._image_path("full", dataset, label, i)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (512,512), "red").save(path)
    args = SimpleNamespace(eval_name="full", selected_checkpoint=4, scope="full")
    rows = [dict(dataset=dataset, model=model, prompt_id=i) for dataset in DATASETS
            for model in ("base", "checkpoint-4") for i in range(2)]
    fields = dict(pickscore=["pickscore"], hpsv2=["hpsv2"], aesthetics_clip=["aesthetics","clip"], imagereward=["imagereward"])
    for metric in METRICS:
        values = [dict(r, **{key: .2 + .01*(r["model"] != "base") for key in fields[metric]}) for r in rows]
        atomic_json(module.EVAL_ROOT/"full"/f"scores/full_{metric}.json", values)
    validate_scores(module, args)
    merge_scores(module, args, [4])
    module.build_report("full", 4)
    report = module._read_json(module.EVAL_ROOT/"full/report/summary.json")
    assert len(report) == 15 and all(r["n"] == 2 for r in report)
    recovered = extract_prompts(module.EVAL_ROOT/"full/scores/full_scores.csv")
    assert all(len(v) == 2 for v in recovered.values())
    broken = module.EVAL_ROOT/"full/scores/full_pickscore.json"
    values = module._read_json(broken); values.pop(); atomic_json(broken, values)
    with pytest.raises(ValueError, match="Missing/duplicated"):
        validate_scores(module, args)
    atomic_json(source/"pickapic_v2.json", [dict(id=0, prompt="different")])
    with pytest.raises(ValueError, match="changed prompts"):
        prepare_prompts(module, config, "full", False)
