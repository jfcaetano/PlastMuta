#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:22:57 2025

@author: jfcaetano
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ----------------------------
# Config
# ----------------------------
df = pd.read_csv("toxic_raw_rdkit.csv")

descriptor_cols = [
    'BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Chi2v',
    'Chi3n','Chi3v','Chi4n','Chi4v','EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2',
    'EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8',
    'EState_VSA9','HallKierAlpha','HeavyAtomCount','HeavyAtomMolWt','Kappa1','Kappa2','Kappa3',
    'MaxAbsEStateIndex','MaxAbsPartialCharge','MaxEStateIndex','MaxPartialCharge',
    'MinAbsEStateIndex','MinAbsPartialCharge','MinEStateIndex','MinPartialCharge','MolLogP',
    'MolMR','MolWt','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles',
    'NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings',
    'NumHAcceptors','NumHDonors','NumHeteroatoms','NumRadicalElectrons','NumRotatableBonds',
    'NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','NumValenceElectrons',
    'PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2',
    'PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9',
    'SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7',
    'SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
    'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9',
    'TPSA','VSA_EState1','VSA_EState10','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5',
    'VSA_EState6','VSA_EState7','VSA_EState8','VSA_EState9',
]

targets = ["Mutagenic"]
runs = 50
test_size = 0.2

results = []


# ----------------------------
# Models (near-defaults)
# ----------------------------
def get_classifiers(seed: int):
    models = {
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "KNN": KNeighborsClassifier(),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=2000),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss"
        )
    else:
        print("⚠️  xgboost is not installed. Install with: pip install xgboost")

    return models


def evaluate_models(X, y, df_meta, label_name: str, task_tag: str):
    per_model_all = {}
    per_model_mono = {}

    for run in range(runs):
        seed = 100 + run

        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X, y, df_meta, test_size=test_size, stratify=y, random_state=seed
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        monomer_mask = (df_test["Monomer"] == "YES").to_numpy()

        models = get_classifiers(seed)
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc_all = accuracy_score(y_test, y_pred) * 100.0
            acc_mono = (
                accuracy_score(y_test[monomer_mask], y_pred[monomer_mask]) * 100.0
                if monomer_mask.sum() > 0 else np.nan
            )

            per_model_all.setdefault(name, []).append(acc_all)
            per_model_mono.setdefault(name, []).append(acc_mono)

            results.append({
                "Target": label_name,
                "Task": task_tag,
                "Model": name,
                "Run": run + 1,
                "Accuracy_All": acc_all,
                "Accuracy_Monomer": acc_mono
            })

        best = max(models.keys(), key=lambda m: per_model_all[m][-1])
        print(f"{label_name} | {task_tag} | Run {run+1:3d} | Best={best} ({per_model_all[best][-1]:.2f}%)")

    summary_rows = []
    for name in per_model_all:
        all_arr = np.array(per_model_all[name], dtype=float)
        mono_arr = np.array([v for v in per_model_mono[name] if not np.isnan(v)], dtype=float)

        row = {
            "Target": label_name,
            "Task": task_tag,
            "Model": name,
            "Run": "Summary",
            "Accuracy_All": all_arr.mean(),
            "Std_All": all_arr.std(ddof=0),
            "Accuracy_Monomer": mono_arr.mean() if len(mono_arr) else np.nan,
            "Std_Monomer": mono_arr.std(ddof=0) if len(mono_arr) else np.nan
        }
        summary_rows.append(row)
        results.append(row)

    summary_rows.sort(key=lambda d: d["Accuracy_All"], reverse=True)
    print(f"\n--- Summary for {label_name} | {task_tag} (sorted by Mean Accuracy_All) ---")
    for r in summary_rows:
        mono_txt = (
            "Monomer=YES: n/a"
            if np.isnan(r["Accuracy_Monomer"])
            else f"Monomer=YES: {r['Accuracy_Monomer']:.2f}% ± {r['Std_Monomer']:.2f}%"
        )
        print(
            f"  {r['Model']:>18s} | "
            f"All: {r['Accuracy_All']:.2f}% ± {r['Std_All']:.2f}% | {mono_txt}"
        )
    print("")


# ----------------------------
# Main
# ----------------------------
for target in targets:
    print(f"\n========== Evaluating target: {target} ==========")

    df_target = df.dropna(subset=[target]).copy()

    X_raw = df_target[descriptor_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X_raw)

    y_orig = LabelEncoder().fit_transform(df_target[target])
    evaluate_models(X, y_orig, df_target, label_name=target, task_tag="Original")

    y_bin = (df_target[target] > 0.9).astype(int).to_numpy()
    if len(np.unique(y_bin)) < 2:
        print(f"Skipping binary task for {target} — only one class present.")
    else:
        evaluate_models(X, y_bin, df_target, label_name=target, task_tag="Binary")

pd.DataFrame(results).to_csv("evaluation_results_5models_defaults.csv", index=False)
print("\nResults saved: evaluation_results_5models_defaults.csv")
