#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


# -----------------------------
# CONFIG (speed knobs)
# -----------------------------
DATA_PATH = "toxic_raw_rdkit.csv"
TARGET_COL = "Mutagenic"
MONOMER_COL = "Monomer"
MONOMER_YES_VALUE = "YES"        

RANDOM_STATE = 42

N_SPLITS = 5
N_REPEATS = 4

DO_PERMUTATION_TEST = True
N_PERMUTATIONS = 10         

RF_TREES = 150                 
RF_MAX_DEPTH = None             

OUT_SUMMARY_CSV = "monomer_0p5_vs_0_summary.csv"
OUT_CV_SCORES_CSV = "monomer_0p5_vs_0_cv_scores.csv"
OUT_PERM_SCORES_CSV = "monomer_0p5_vs_0_perm_scores.csv"


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


# -----------------------------
# HELPERS
# -----------------------------
def permutation_test_auc(model, X, y, cv, n_perm=50, random_state=0):
    """
    Permutation test for mean CV ROC-AUC.
    Returns dict with observed, perm_mean, perm_std, p_value, perm_scores
    """
    rng = np.random.RandomState(random_state)

    observed = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    perm_scores = np.empty(n_perm, dtype=float)
    y = np.asarray(y)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_scores[i] = cross_val_score(model, X, y_perm, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    p_value = (np.sum(perm_scores >= observed) + 1) / (n_perm + 1)

    return {
        "observed_auc": float(observed),
        "perm_mean_auc": float(np.mean(perm_scores)),
        "perm_std_auc": float(np.std(perm_scores)),
        "p_value": float(p_value),
        "perm_scores": perm_scores,
    }


def gaussian_bhattacharyya_distance(Z0, Z1, eps=1e-6):
    """
    Bhattacharyya distance between Gaussians fit to Z0 and Z1 (PCA space).
    Smaller => more overlap (less separable).
    """
    mu0, mu1 = Z0.mean(axis=0), Z1.mean(axis=0)
    S0 = np.cov(Z0, rowvar=False)
    S1 = np.cov(Z1, rowvar=False)
    S = 0.5 * (S0 + S1)

    d = S.shape[0]
    S0 = S0 + eps * np.eye(d)
    S1 = S1 + eps * np.eye(d)
    S  = S  + eps * np.eye(d)

    dmu = (mu1 - mu0).reshape(-1, 1)
    term1 = 0.125 * float(dmu.T @ np.linalg.inv(S) @ dmu)

    detS  = np.linalg.det(S)
    detS0 = np.linalg.det(S0)
    detS1 = np.linalg.det(S1)
    term2 = 0.5 * np.log(detS / np.sqrt(detS0 * detS1))

    return float(term1 + term2)


def fisher_separation_1d(a, b, eps=1e-12):
    """
    Fisher separation for 1D distributions a and b.
    Smaller => more overlap.
    """
    a = np.asarray(a); b = np.asarray(b)
    return float((a.mean() - b.mean())**2 / (a.var() + b.var() + eps))


def build_models():
    lr_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    rf_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return lr_model, rf_model


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = pd.read_csv(DATA_PATH)

    # required columns
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}.")
    if MONOMER_COL not in df.columns:
        raise ValueError(f"Monomer column '{MONOMER_COL}' not found in {DATA_PATH}.")

    missing_desc = [c for c in descriptor_cols if c not in df.columns]
    if missing_desc:
        raise ValueError(f"Missing descriptor columns in CSV: {missing_desc[:10]} ... (total {len(missing_desc)})")

    # monomer-only filter
    monomer = df[MONOMER_COL].astype(str).str.upper().str.strip()
    df_m = df.loc[monomer == str(MONOMER_YES_VALUE).upper()].copy()

    # numeric target and filter 0 vs 0.5 only
    df_m = df_m.dropna(subset=[TARGET_COL]).copy()
    y_num = pd.to_numeric(df_m[TARGET_COL], errors="coerce")
    mask = y_num.isin([0.0, 0.5])

    df_sub = df_m.loc[mask].copy()
    y = (y_num.loc[mask] == 0.5).astype(int).values 

    if len(y) == 0:
        raise ValueError("No rows found after filtering to Monomer==YES and target in {0, 0.5}.")

    if np.unique(y).size < 2:
        raise ValueError("Only one class present after filtering to {0, 0.5} (Monomer==YES). Cannot evaluate.")

    X = df_sub[descriptor_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).values

    n1 = int(y.sum())
    n0 = int((1 - y).sum())

    print(f"\n=== Monomer==YES | 0.5 vs 0 separability check for '{TARGET_COL}' ===")
    print(f"N={len(y)} | class1 (0.5/H341) = {n1} | class0 (0) = {n0}")

    # CV definition (fast)
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    lr_model, rf_model = build_models()

    records = []
    cv_scores_rows = []

    for model_name, model in [("LogisticRegression", lr_model), ("RandomForest", rf_model)]:
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        pr_scores  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)

        records.append({
            "scope": "Monomer==YES",
            "target": TARGET_COL,
            "task": "0.5_vs_0",
            "model": model_name,
            "n_total": int(len(y)),
            "n_class1_0p5": n1,
            "n_class0_0": n0,
            "cv_n_splits": N_SPLITS,
            "cv_n_repeats": N_REPEATS,
            "roc_auc_mean": float(np.mean(auc_scores)),
            "roc_auc_std": float(np.std(auc_scores)),
            "auprc_mean": float(np.mean(pr_scores)),
            "auprc_std": float(np.std(pr_scores)),
        })

        for i, (auc, pr) in enumerate(zip(auc_scores, pr_scores), start=1):
            cv_scores_rows.append({
                "scope": "Monomer==YES",
                "target": TARGET_COL,
                "task": "0.5_vs_0",
                "model": model_name,
                "fold_index": i,
                "roc_auc": float(auc),
                "auprc": float(pr),
            })

        print(f"\n{model_name} (CV):")
        print(f"  ROC-AUC = {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
        print(f"  AUPRC   = {np.mean(pr_scores):.3f} ± {np.std(pr_scores):.3f}")

    proc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_proc = proc.fit_transform(X)

    n_pcs = min(10, X_proc.shape[1])
    Z = PCA(n_components=n_pcs, random_state=RANDOM_STATE).fit_transform(X_proc)

    Z0 = Z[y == 0]
    Z1 = Z[y == 1]

    Db = gaussian_bhattacharyya_distance(Z0, Z1)
    fisher_pc1 = fisher_separation_1d(Z0[:, 0], Z1[:, 0])

    overlap_row = {
        "scope": "Monomer==YES",
        "target": TARGET_COL,
        "task": "0.5_vs_0",
        "pca_n_components": int(n_pcs),
        "bhattacharyya_distance": float(Db),
        "fisher_separation_pc1": float(fisher_pc1),
    }

    print("\nOverlap proxies (PCA space):")
    print(f"  Bhattacharyya distance (first {n_pcs} PCs): {Db:.3f}  (smaller => more overlap)")
    print(f"  Fisher separation on PC1:                 {fisher_pc1:.3f}  (smaller => more overlap)")

    perm_rows = []
    perm_summary = {}
    if DO_PERMUTATION_TEST:
        perm = permutation_test_auc(
            rf_model, X, y, cv=cv, n_perm=N_PERMUTATIONS, random_state=7
        )
        perm_summary = {
            "scope": "Monomer==YES",
            "target": TARGET_COL,
            "task": "0.5_vs_0",
            "model": "RandomForest",
            "n_permutations": int(N_PERMUTATIONS),
            "observed_auc": perm["observed_auc"],
            "perm_mean_auc": perm["perm_mean_auc"],
            "perm_std_auc": perm["perm_std_auc"],
            "p_value": perm["p_value"],
        }
        for i, s in enumerate(perm["perm_scores"], start=1):
            perm_rows.append({
                "scope": "Monomer==YES",
                "target": TARGET_COL,
                "task": "0.5_vs_0",
                "model": "RandomForest",
                "perm_index": i,
                "perm_auc": float(s),
            })

        print("\nPermutation test (RF, mean CV ROC-AUC):")
        print(f"  Observed AUC: {perm['observed_auc']:.3f}")
        print(f"  Permuted AUC: {perm['perm_mean_auc']:.3f} ± {perm['perm_std_auc']:.3f}")
        print(f"  p-value (perm >= obs): {perm['p_value']:.4f}")

    # Write CSV outputs
    summary_df = pd.DataFrame(records)

    # merge overlap + perm into the first row for convenience (also kept separately as columns)
    summary_df = summary_df.merge(
        pd.DataFrame([overlap_row]),
        on=["scope", "target", "task"],
        how="left"
    )
    if DO_PERMUTATION_TEST:
        summary_df = summary_df.merge(
            pd.DataFrame([perm_summary]),
            on=["scope", "target", "task", "model"],
            how="left"
        )

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    pd.DataFrame(cv_scores_rows).to_csv(OUT_CV_SCORES_CSV, index=False)

    if DO_PERMUTATION_TEST:
        pd.DataFrame(perm_rows).to_csv(OUT_PERM_SCORES_CSV, index=False)

    print("\n✅ CSV files written:")
    print(f"  - {OUT_SUMMARY_CSV}")
    print(f"  - {OUT_CV_SCORES_CSV}")
    if DO_PERMUTATION_TEST:
        print(f"  - {OUT_PERM_SCORES_CSV}")


if __name__ == "__main__":
    main()
