"""Shared evaluation harness: leakage ladder + per-fold reporting.

Extracted from scripts/run_experiment.py so that more than one dataset can be put
through identical machinery. The synthetic rig stand-in and ESPset differ in almost
everything — waveforms vs spectra, seeded vs in-service faults, 2 machines vs 11 —
so the one thing that must NOT differ is how they are split and scored. Any
difference in results then belongs to the data rather than the harness.
"""

from __future__ import annotations

import json

import numpy as np

from pumpwatch.evaluate import (
    bootstrap_ci,
    classify_report,
    friedman_nemenyi_allowed,
    reliability_bins,
    risk_coverage_curve,
)
from pumpwatch.gateway.baselines import fit_predict
from pumpwatch.splits import (
    describe_fold,
    normalize_features,
    split_component_wise,
    split_cross_operating,
    split_lomo,
    split_random_window,
    split_record_wise,
)


def _fmt(per_machine: dict, key: str) -> str:
    vals = [m[key] for m in per_machine.values() if m.get(key) is not None]
    return f"{np.mean(vals):.3f}" if vals else "n/a"


def build_ladder(machines, groups, n_samples: int) -> dict[str, object]:
    """Construct every rung of the leakage ladder that this dataset can support.

    A rung whose grouping key is missing or degenerate is omitted rather than
    silently substituted with a weaker split — reporting a "component-wise" number
    that was actually record-wise is exactly the failure the ladder exists to expose.
    """
    ladder: dict[str, object] = {}
    ladder["0_random_window"] = split_random_window(n_samples, seed=0)

    candidates = [
        ("1_record_wise", groups.get("record"), split_record_wise),
        ("2_component_wise", groups.get("component"), split_component_wise),
        ("3_cross_operating", groups.get("operating"), split_cross_operating),
        ("4_lomo", machines, split_lomo),
    ]
    for name, keys, fn in candidates:
        if not keys or len(set(keys)) < 2 or any(k == "" for k in keys):
            print(f"  [skip] {name}: grouping key absent or degenerate")
            continue
        ladder[name] = fn(keys)
    return ladder


def run_split(
    X,
    y,
    machines,
    model_factory,
    model_name: str,
    result,
    norm_strategy: str = "unsupervised_per_machine",
    verbose: bool = False,
) -> dict:
    """Evaluate one rung of the leakage ladder under an explicit normalisation.

    Persists the full per-fold report — PR-AUC (the declared headline metric),
    ROC-AUC, ECE, Brier, coverage, raw confusion counts and latency — not just
    macro-F1. Keeping only macro-F1 is how a repo ends up unable to report the
    metric its own config declares as headline.
    """
    per_machine: dict[str, dict] = {}
    all_true, all_pred = [], []
    fit_latencies, predict_latencies = [], []

    for fold in result.folds:
        Xn = normalize_features(X, machines, fold.train_idx, strategy=norm_strategy)
        model = model_factory()
        pred = fit_predict(
            model,
            Xn[fold.context_idx],
            y[fold.context_idx],
            Xn[fold.test_idx],
            model_name,
        )
        report = classify_report(y[fold.test_idx], pred.y_pred, pred.y_proba, pred.classes)
        entry = {
            **report.as_dict(),
            "fold": describe_fold(fold, machines, y.tolist()),
            "latency_fit_s": pred.latency_fit_s,
            "latency_predict_s": pred.latency_predict_s,
        }
        if pred.y_proba is not None and pred.classes is not None:
            entry["reliability"] = reliability_bins(
                y[fold.test_idx], pred.y_proba, pred.classes
            )
            entry["risk_coverage"] = risk_coverage_curve(
                y[fold.test_idx], pred.y_pred, pred.y_proba.max(axis=1)
            )
        per_machine[fold.held_out] = entry
        all_true.extend(y[fold.test_idx].tolist())
        all_pred.extend(pred.y_pred.tolist())
        fit_latencies.append(pred.latency_fit_s)
        predict_latencies.append(pred.latency_predict_s)
        if verbose:
            print(json.dumps(describe_fold(fold, machines, y.tolist()), indent=2))

    overall = classify_report(np.array(all_true), np.array(all_pred))
    f1s = [m["macro_f1"] for m in per_machine.values()]
    return {
        "model": model_name,
        "norm_strategy": norm_strategy,
        "split_level": int(result.level),
        "split_verdict": result.verdict,
        "per_machine": per_machine,
        "per_machine_macro_f1": {k: v["macro_f1"] for k, v in per_machine.items()},
        "overall_macro_f1": overall.macro_f1,
        "overall_accuracy": overall.accuracy,
        "overall_weighted_f1": overall.weighted_f1,
        "overall_coverage": overall.coverage,
        "overall_confusion": overall.confusion.tolist(),
        "overall_labels": [str(x) for x in overall.labels],
        # Bootstrap at the fold's held-out unit. With 2-3 folds this is
        # near-meaningless and is reported so the reader can see that for themselves.
        "macro_f1_bootstrap_ci": bootstrap_ci(np.array(f1s)),
        "bootstrap_unit": "held_out_group",
        "bootstrap_warning": (
            None if len(f1s) >= 5
            else f"CI over {len(f1s)} folds is not interpretable; report per-fold values"
        ),
        "mean_latency_fit_s": float(np.mean(fit_latencies)),
        "mean_latency_predict_s": float(np.mean(predict_latencies)),
        "n_folds": len(result.folds),
        "friedman_allowed": friedman_nemenyi_allowed(len(result.folds)),
        # Predictions retained so model pairs can be compared with McNemar.
        "_y_true": all_true,
        "_y_pred": all_pred,
    }


def run_split_repeated(
    X,
    y,
    machines,
    model_factory,
    model_name: str,
    result,
    norm_strategy: str = "unsupervised_per_machine",
    seeds: tuple = (0, 1, 2, 3, 4),
    verbose: bool = False,
) -> dict:
    """Run one split protocol under several seeds and aggregate.

    Every number in this repo was previously a single deterministic run. That is
    least defensible for the model the headline claim rests on: TabPFN randomises
    its ensemble permutations, so a single run reports one draw from a distribution
    whose width nobody has measured. LightGBM's subsampling is stochastic too.

    `model_factory` MUST accept a `seed` keyword for this to measure anything. It
    used to fall back to re-running an identical model when the factory took no
    seed — which silently burned N times the compute to produce a spread of exactly
    zero. Now that case raises, because a std of 0.000 across five seeds is
    indistinguishable from a working deterministic model and there is no way to
    tell from the output which one you got.
    """
    import inspect

    try:
        takes_seed = "seed" in inspect.signature(model_factory).parameters
    except (TypeError, ValueError):
        takes_seed = False
    if not takes_seed:
        raise TypeError(
            f"model_factory for {model_name!r} takes no `seed` argument, so running "
            f"{len(seeds)} seeds would re-evaluate an identical model and report a "
            f"spread of zero. Give the factory a `seed` keyword (and thread it into "
            f"random_state), or call run_split once instead."
        )

    runs = []
    for s in seeds:
        factory = (lambda s=s: model_factory(seed=s))
        runs.append(
            run_split(
                X, y, machines, factory, model_name, result,
                norm_strategy=norm_strategy, verbose=verbose,
            )
        )

    def spread(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_seeds": len(vals),
            "values": [float(v) for v in vals],
        }

    base = dict(runs[0])
    base.update({
        "seeds": list(seeds),
        "macro_f1_over_seeds": spread("overall_macro_f1"),
        "accuracy_over_seeds": spread("overall_accuracy"),
        "coverage_over_seeds": spread("overall_coverage"),
        # Report the mean as the headline so a lucky seed cannot become the result.
        "overall_macro_f1": float(np.mean([r["overall_macro_f1"] for r in runs])),
        "overall_accuracy": float(np.mean([r["overall_accuracy"] for r in runs])),
    })
    return base


def run_gate_per_machine(
    X,
    y,
    machines,
    feature_names,
    healthy_label: str = "healthy",
    seed: int = 0,
    field_fault_prevalence: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Commission the MCU gate per pump and measure what it escalates.

    Stage 1 of the two-tier architecture, and the only quantitative content of the
    architecture claim: the gate decides which feature vectors are worth a LoRa
    transmission, so its escalation rate is what links classification accuracy to
    battery life.

    Protocol note — the gate is fitted on **each pump's own healthy baseline**,
    because that is the deployment model: a node is installed on a known-good pump
    and watches for departures from *its* normal. Fitting on other pumps' healthy
    data makes every window on the target look anomalous and escalates ~100% of
    them, which is a statement about between-pump variability rather than about the
    gate. Half the healthy windows commission the node; the rest are evaluation.

    Lives here rather than in a script because both the synthetic and the ESPset
    experiments need it, and copying it would recreate the drift that the model
    registry was introduced to remove.
    """
    from pumpwatch.baseline_lifecycle import commissioning_length
    from pumpwatch.node.gates import (
        evaluate_gate,
        fit_composite_gate,
        select_gate_features,
    )

    machines_arr = np.asarray(machines)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}

    # The gate runs on a small, physically-chosen subset: its dimensionality is
    # bounded by how long commissioning takes (Mahalanobis needs n > 10p), not by
    # what the extractor can compute.
    try:
        gate_cols = select_gate_features(list(feature_names))
    except ValueError as exc:
        # No gate feature set covers this schema. Skipping is right — the gate is
        # one step of a larger experiment — but it must be visible, because a
        # silently absent gate looks identical to a gate that escalates nothing.
        if verbose:
            print(f"  [skip] gate not defined for this feature schema: {exc}")
        return {}
    gate_names = [feature_names[i] for i in gate_cols]
    plan = commissioning_length(len(gate_cols))

    for machine in sorted(set(machines_arr.tolist())):
        m_idx = np.flatnonzero(machines_arr == machine)
        healthy_idx = m_idx[y[m_idx] == healthy_label]
        if len(healthy_idx) < 10:
            if verbose:
                print(f"  [skip] {machine}: too few healthy commissioning samples")
            continue

        shuffled = rng.permutation(healthy_idx)
        n_fit = len(shuffled) // 2
        fit_idx, held_healthy = shuffled[:n_fit], shuffled[n_fit:]
        eval_idx = np.concatenate([held_healthy, m_idx[y[m_idx] != healthy_label]])

        # Strategy stated rather than defaulted. This project's whole normalisation
        # discipline is that the choice must be explicit, and it applies to the gate
        # too: a node self-commissions on its own pump using unlabelled data it
        # collects there, which is exactly the per-machine (transductive) reading.
        # train_pooled would be wrong here - there is no pool, the gate sees one pump.
        Xn = normalize_features(
            X, machines, fit_idx, strategy="unsupervised_per_machine"
        )[:, gate_cols]
        adequate = n_fit >= plan.min_samples
        try:
            gate = fit_composite_gate(Xn[fit_idx], feature_names=gate_names)
        except ValueError as exc:
            # Mahalanobis refuses an under-conditioned covariance; that is the
            # guard working, and the shortfall is reported rather than bypassed.
            if verbose:
                print(f"  [skip] {machine}: {exc}")
            continue

        stats = evaluate_gate(
            gate, Xn[eval_idx], y[eval_idx],
            healthy_label=healthy_label,
            field_fault_prevalence=field_fault_prevalence,
        )
        stats.update({
            "n_commissioning": int(n_fit),
            "n_gate_features": len(gate_cols),
            "gate_features": gate_names,
            "commissioning_required": plan.min_samples,
            "commissioning_adequate": bool(adequate),
        })
        out[machine] = stats
        if verbose:
            flag = "" if adequate else f"  [under-conditioned: {n_fit} < {plan.min_samples}]"
            print(
                f"  {machine}: escalate healthy={stats['escalation_rate_healthy']:.2f} "
                f"faulty={stats['escalation_rate_faulty']:.2f} "
                f"field={stats['escalation_rate_field']:.3f}{flag}"
            )
    return out


def summarise_gate(gate_results: dict, runtime_hours_per_day: float = 3.0) -> dict:
    """Turn per-machine gate stats into the battery number the architecture claim needs."""
    from pumpwatch.node.energy import event_triggered_energy

    if not gate_results:
        return {}
    mean_field = float(np.mean([g["escalation_rate_field"] for g in gate_results.values()]))
    mean_recall = float(np.mean([g["escalation_rate_faulty"] for g in gate_results.values()]))
    energy = event_triggered_energy(runtime_hours_per_day, escalation_rate=mean_field)
    adequate = [g["commissioning_adequate"] for g in gate_results.values()]
    return {
        "n_machines": len(gate_results),
        "mean_escalation_rate_testset": float(
            np.mean([g["escalation_rate_overall"] for g in gate_results.values()])
        ),
        "mean_field_escalation_rate": mean_field,
        "gate_recall_ceiling": mean_recall,
        "battery_years_at_field_rate": energy.battery_years,
        "uplinks_per_day_at_field_rate": energy.transmissions_per_day,
        "energy_breakdown_mAh_per_day": energy.breakdown_mAh,
        "tx_fraction": energy.tx_fraction,
        "commissioning_adequate_on_all_machines": all(adequate),
        "n_machines_adequately_commissioned": int(sum(adequate)),
        "note": (
            "Gateway accuracy is an upper bound conditioned on escalation: end-to-end "
            "fault recall <= gate_recall_ceiling. The test-set escalation rate reflects "
            "how many faulty examples were collected, not field prevalence; battery "
            "life is driven by the field rate, dominated by healthy false-escalation."
        ),
    }
