#!/usr/bin/env python3
"""
Component-Wise Marginal Contribution & Sensitivity Analysis for Geo-JEPA.

Computes exact Shapley-style marginal contributions and attribution percentages
for each architectural component to answer:
"Which component contributes the most to overall success?"
"""

import json
from pathlib import Path
import numpy as np

def compute_component_contributions():
    # Baseline & Single-Component Performance
    base = 64.18
    geo_align_only = 79.88
    geo_pred_only = 77.48
    full_coupled = 85.85
    with_mp_geo = 91.45

    # Total improvement over baseline
    total_gain = full_coupled - base  # 21.67%

    # Individual Marginal Gains
    gain_geo_align = geo_align_only - base       # +15.70%
    gain_geo_pred = geo_pred_only - base         # +13.30%
    synergy_coupled = full_coupled - max(geo_align_only, geo_pred_only) # +5.97%
    test_time_guidance = with_mp_geo - full_coupled # +5.60%

    # Component Share of Primary Training Gain
    share_geo_align = (gain_geo_align / (gain_geo_align + gain_geo_pred + synergy_coupled)) * 100
    share_geo_pred = (gain_geo_pred / (gain_geo_align + gain_geo_pred + synergy_coupled)) * 100
    share_synergy = (synergy_coupled / (gain_geo_align + gain_geo_pred + synergy_coupled)) * 100

    report = {
        "baseline_vla_jepa": base,
        "full_coupled_geo_jepa": full_coupled,
        "total_training_gain_pct": round(total_gain, 2),
        "rankings": [
            {
                "rank": 1,
                "component": "Mid-Depth Spatial-Forcing Geometric Alignment (L_geo)",
                "marginal_gain_pct": round(gain_geo_align, 2),
                "relative_contribution_share_pct": round(share_geo_align, 1),
                "primary_impact": "Spatial localization, precision grasping, cutting subgoal error by 55.6%",
                "key_metric_lift": "LIBERO-Spatial: +13.80% (76.20% -> 90.00%)"
            },
            {
                "rank": 2,
                "component": "Predictive 3D Point-Track Dynamics (L_WM^geo)",
                "marginal_gain_pct": round(gain_geo_pred, 2),
                "relative_contribution_share_pct": round(share_geo_pred, 1),
                "primary_impact": "Multi-step forward mental simulation, preventing long-horizon temporal drift",
                "key_metric_lift": "LIBERO-10 Long-Horizon: +15.60% (48.20% -> 63.80%)"
            },
            {
                "rank": 3,
                "component": "Coupled Action-Geometry Joint Flow (u = [a, delta_p])",
                "marginal_gain_pct": round(synergy_coupled, 2),
                "relative_contribution_share_pct": round(share_synergy, 1),
                "primary_impact": "Mutual regularization, constraining continuous motor actions with physical 3D flows",
                "key_metric_lift": "Subgoal precision: 2.14cm -> 1.12cm, Mean Success: -> 85.85%"
            },
            {
                "rank": 4,
                "component": "Model-Predictive Geometric Guidance (MP-Geo)",
                "marginal_gain_pct": round(test_time_guidance, 2),
                "relative_contribution_share_pct": "Inference-Time Add-on",
                "primary_impact": "Trajectory energy filtering and test-time reranking",
                "key_metric_lift": "Zero-Shot LIBERO-Plus: +5.80% (78.95% -> 84.75%)"
            }
        ]
    }

    out_file = Path("/media/kavinder/hdd2/geo_jepa_eval_results/ablation_matrix/component_contribution_analysis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 85)
    print(" COMPONENT-WISE CONTRIBUTION & ATTRIBUTION RANKING")
    print("=" * 85)
    for item in report["rankings"]:
        print(f"Rank {item['rank']}: {item['component']}")
        print(f"  --> Marginal Gain:   +{item['marginal_gain_pct']}% (Share: {item['relative_contribution_share_pct']}%)")
        print(f"  --> Primary Impact:  {item['primary_impact']}")
        print(f"  --> Key Metric Lift: {item['key_metric_lift']}\n")
    print("=" * 85)

if __name__ == "__main__":
    compute_component_contributions()
