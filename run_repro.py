"""集成验证：用真实精灵跑通 run_reference_pipeline 完整复现管线。

数据流（与原 Zoo-Bus 一致）：
  AssetPack(assets/ 真实精灵) → SceneParameterSampler 随机采样场景参数 →
  render_reference_scene 精灵合成 → IdealDetectionReplay 理想检测（生成器坐标即真值）→
  DetectionGeometryFilter 几何稳定性过滤 → ZooBusTemplateGenerator 29 种问题模板 →
  source_id 级 train/evaluation/test 分割 → 答案与题型平衡 →
  export_dataset（parquet + messages.jsonl + summary.json + 审计）。
"""

from __future__ import annotations

import json
from pathlib import Path

from synthetic_vqa.pipeline import run_reference_pipeline
from synthetic_vqa.reference import ReferenceBuildConfig


def main() -> None:
    # 小规模诊断构建：16 个源场景足以跑通全链路并产出可读产物
    config = ReferenceBuildConfig(num_scenes=16, seed=42, per_answer=2, per_question_type=4)
    out = Path("artifacts/zoo_bus_repro")
    summary = run_reference_pipeline(out, config=config, asset_root=Path("assets"))
    excerpt = {k: summary.get(k) for k in ("rows", "source_images", "splits", "question_types", "pipeline")}
    print(json.dumps(excerpt, ensure_ascii=False, indent=2))
    print("output dir:", out)


if __name__ == "__main__":
    main()
