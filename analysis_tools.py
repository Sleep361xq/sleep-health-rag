import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from tools.eeg_sleep_staging_tool import predict_sleep_stages
from tools.osa_prediction_tool import predict_osa_severity


@dataclass
class SleepStagingAnalysisResult:
    stage_sequence: list[str]
    epoch_count: int
    stage_counts: dict[str, int]
    stage_ratios: dict[str, float]
    output_path: Optional[str]


@dataclass
class OsaAnalysisResult:
    severity_label: str
    predicted_class_index: Optional[int]
    class_probabilities: dict[str, float]
    subject_id: Optional[str]
    output_path: Optional[str]


@dataclass
class SleepAnalysisBundle:
    staging: Optional[SleepStagingAnalysisResult]
    osa: Optional[OsaAnalysisResult]


def _stage_ratios(stage_counts: dict[str, int], epoch_count: int) -> dict[str, float]:
    if epoch_count <= 0:
        return {}
    return {stage: round(count / epoch_count, 4) for stage, count in stage_counts.items()}


def analyze_sleep_staging(npz_path: str | Path, save_artifact: bool = True) -> SleepStagingAnalysisResult:
    artifact_path = None
    if save_artifact:
        artifact_path = Path(tempfile.gettempdir()) / f"{Path(npz_path).stem}_sleep_staging.json"

    stage_sequence = predict_sleep_stages(
        npz_path,
        postprocess=True,
        output_path=artifact_path if artifact_path else None,
    )
    counts = dict(Counter(stage_sequence))
    return SleepStagingAnalysisResult(
        stage_sequence=stage_sequence,
        epoch_count=len(stage_sequence),
        stage_counts=counts,
        stage_ratios=_stage_ratios(counts, len(stage_sequence)),
        output_path=str(artifact_path) if artifact_path else None,
    )


def analyze_osa(npz_path: str | Path, save_artifact: bool = True) -> OsaAnalysisResult:
    artifact_path = None
    raw_result: dict[str, Any] = {}
    if save_artifact:
        artifact_path = Path(tempfile.gettempdir()) / f"{Path(npz_path).stem}_osa_prediction.json"

    severity_label = predict_osa_severity(
        npz_path,
        output_path=artifact_path if artifact_path else None,
    )

    if artifact_path and artifact_path.exists():
        raw_result = json.loads(artifact_path.read_text(encoding="utf-8"))

    return OsaAnalysisResult(
        severity_label=severity_label,
        predicted_class_index=raw_result.get("predicted_class_index"),
        class_probabilities=raw_result.get("class_probabilities", {}),
        subject_id=raw_result.get("id"),
        output_path=str(artifact_path) if artifact_path else None,
    )


def analyze_uploaded_sleep_data(
    staging_npz_path: Optional[str | Path] = None,
    osa_npz_path: Optional[str | Path] = None,
    save_artifact: bool = True,
) -> SleepAnalysisBundle:
    staging_result = analyze_sleep_staging(staging_npz_path, save_artifact=save_artifact) if staging_npz_path else None
    osa_result = analyze_osa(osa_npz_path, save_artifact=save_artifact) if osa_npz_path else None
    return SleepAnalysisBundle(staging=staging_result, osa=osa_result)


def build_analysis_summary(bundle: SleepAnalysisBundle) -> str:
    lines: list[str] = []

    if bundle.staging:
        lines.append("睡眠分期工具结果：")
        lines.append(f"- 总 epoch 数：{bundle.staging.epoch_count}")
        if bundle.staging.stage_counts:
            counts_text = "，".join([f"{stage}={count}" for stage, count in bundle.staging.stage_counts.items()])
            lines.append(f"- 分期计数：{counts_text}")
        if bundle.staging.stage_ratios:
            ratio_text = "，".join([f"{stage}={ratio:.1%}" for stage, ratio in bundle.staging.stage_ratios.items()])
            lines.append(f"- 分期占比：{ratio_text}")

    if bundle.osa:
        lines.append("OSA 分类工具结果：")
        lines.append(f"- 预测分级：{bundle.osa.severity_label}")
        if bundle.osa.class_probabilities:
            prob_text = "，".join([f"{label}={prob:.3f}" for label, prob in bundle.osa.class_probabilities.items()])
            lines.append(f"- 类别概率：{prob_text}")

    return "\n".join(lines).strip()


def bundle_to_dict(bundle: SleepAnalysisBundle) -> dict[str, Any]:
    return {
        "staging": asdict(bundle.staging) if bundle.staging else None,
        "osa": asdict(bundle.osa) if bundle.osa else None,
        "summary": build_analysis_summary(bundle),
    }
