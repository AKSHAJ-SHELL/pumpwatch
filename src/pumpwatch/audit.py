"""Confound audit: refuse experiments that merge class-confounded sources.

Dry-run labels exist only on the own rig; bearing/misalignment only in Twente.
Merging them makes rig identity the dry-run feature. This module hard-fails
such merges and reports class–machine mutual information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


@dataclass
class ConfoundReport:
    n_samples: int
    n_machines: int
    n_classes: int
    n_sources: int
    class_machine_nmi: float
    class_source_nmi: float
    machine_predict_accuracy: float
    machine_predict_baseline: float
    confounded: bool
    reasons: list[str]
    contingency_class_machine: dict
    # High machine-from-features accuracy within one source is expected when
    # pumps differ in speed/mount — LOMO tests generalization. It is a warning,
    # not an automatic hard-fail, unless paired with multi-source class exclusivity.
    feature_encodes_machine: bool = False
    warnings: list[str] = field(default_factory=list)


class ConfoundError(RuntimeError):
    """Raised when an experiment would train on a confounded merge."""


def _encode(labels: list[str]) -> np.ndarray:
    return LabelEncoder().fit_transform(labels)


def audit_confound(
    y_class: list[str],
    machine_ids: list[str],
    sources: list[str],
    X: Optional[np.ndarray] = None,
    nmi_threshold: float = 0.5,
    machine_probe_margin: float = 0.25,
    seed: int = 0,
) -> ConfoundReport:
    """Audit class–machine / class–source confounding.

    Hard-fail conditions (confounded=True):
      - a class appears in only one source while multiple sources are present
        (the dry-run/Twente merge pattern)
      - a class appears on only one machine while multiple machines are present
      - class–source or class–machine NMI above threshold with multiple groups

    Warning only (does not set confounded):
      - features predict machine identity within a single source (speed/mount
        encoding). LOMO exists to measure whether that kills generalization.
    """
    y_class = list(y_class)
    machine_ids = list(machine_ids)
    sources = list(sources)
    n = len(y_class)
    if not (len(machine_ids) == n and len(sources) == n):
        raise ValueError("y_class, machine_ids, sources must be same length")

    y = _encode(y_class)
    m = _encode(machine_ids)
    s = _encode(sources)

    nmi_cm = float(normalized_mutual_info_score(y, m))
    nmi_cs = float(normalized_mutual_info_score(y, s))

    classes = sorted(set(y_class))
    machines = sorted(set(machine_ids))
    source_set = sorted(set(sources))
    contingency = {c: {m_id: 0 for m_id in machines} for c in classes}
    for c, m_id in zip(y_class, machine_ids):
        contingency[c][m_id] += 1

    reasons: list[str] = []
    warnings: list[str] = []
    confounded = False
    multi_source = len(source_set) > 1
    multi_machine = len(machines) > 1

    for c in classes:
        machines_for_c = {m_id for m_id, lab in zip(machine_ids, y_class) if lab == c}
        sources_for_c = {src for src, lab in zip(sources, y_class) if lab == c}
        if len(machines_for_c) == 1 and multi_machine:
            reasons.append(f"class '{c}' appears on only one machine: {machines_for_c}")
            confounded = True
        if len(sources_for_c) == 1 and multi_source:
            reasons.append(f"class '{c}' appears in only one source: {sources_for_c}")
            confounded = True

    if nmi_cm >= nmi_threshold and multi_machine:
        reasons.append(f"class–machine NMI={nmi_cm:.3f} >= {nmi_threshold}")
        confounded = True
    if nmi_cs >= nmi_threshold and multi_source:
        reasons.append(f"class–source NMI={nmi_cs:.3f} >= {nmi_threshold}")
        confounded = True

    probe_acc = 0.0
    baseline = 1.0 / max(len(machines), 1)
    feature_encodes_machine = False
    if X is not None and multi_machine and n >= 20:
        clf = RandomForestClassifier(n_estimators=50, random_state=seed, max_depth=6)
        _, counts = np.unique(m, return_counts=True)
        n_folds = int(min(3, counts.min()))
        if n_folds >= 2:
            scores = cross_val_score(clf, X, m, cv=n_folds)
            probe_acc = float(scores.mean())
            if probe_acc >= baseline + machine_probe_margin:
                feature_encodes_machine = True
                msg = (
                    f"machine-from-features probe acc={probe_acc:.3f} "
                    f"(baseline={baseline:.3f})"
                )
                if multi_source:
                    # Features encode machine AND we mixed sources — lethal
                    reasons.append(msg + " under multi-source merge")
                    confounded = True
                else:
                    warnings.append(
                        msg + " — expected if pumps differ in speed/mount; "
                        "LOMO measures whether this kills generalization"
                    )

    return ConfoundReport(
        n_samples=n,
        n_machines=len(machines),
        n_classes=len(classes),
        n_sources=len(source_set),
        class_machine_nmi=nmi_cm,
        class_source_nmi=nmi_cs,
        machine_predict_accuracy=probe_acc,
        machine_predict_baseline=baseline,
        confounded=confounded,
        reasons=reasons,
        contingency_class_machine=contingency,
        feature_encodes_machine=feature_encodes_machine,
        warnings=warnings,
    )


def assert_not_confounded(report: ConfoundReport) -> None:
    if report.confounded:
        raise ConfoundError(
            "Refusing to run experiment on confounded data merge.\n"
            + "\n".join(f"  - {r}" for r in report.reasons)
            + "\nDry-run labels belong to own-rig only; do not merge with Twente "
            "fault classes. Run ML dry-run experiments within own-rig data alone."
        )


def mutual_info_class_machine(y_class: list[str], machine_ids: list[str]) -> float:
    return float(mutual_info_score(_encode(y_class), _encode(machine_ids)))
