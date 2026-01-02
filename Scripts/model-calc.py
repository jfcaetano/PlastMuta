#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 15:59:23 2025

@author: jfcaetano
"""


##########################################
##########################################

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel

import shap

warnings.filterwarnings("ignore", category=UserWarning)

# --- Load data ---
df = pd.read_csv("toxic_raw_rdkit.csv")

# --- Descriptor columns ---
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

targets = ['Mutagenic']
runs = 50
test_size = 0.2
USE_FEATURE_SELECTION = True

# storage
results = []
feature_importances = []
shap_values_storage = []

# function-level storage
function_performance = []

def _safe_pct(numer, denom):
    if denom is None or denom == 0:
        return np.nan
    return 100.0 * (numer / denom)

def _confusion_percentages_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "TN_pct": _safe_pct(tn, n),
        "FP_pct": _safe_pct(fp, n),
        "FN_pct": _safe_pct(fn, n),
        "TP_pct": _safe_pct(tp, n),
        "N_test": int(n),
    }

def _parse_functions_cell(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";")]
    return [p for p in parts if p]

def _update_function_stats(stats_dict, funcs, y_t, y_p, is_binary):
    if not funcs:
        return

    correct = int(y_t == y_p)

    if is_binary:
        if y_t == 0 and y_p == 0:
            outcome = "TN"
        elif y_t == 0 and y_p == 1:
            outcome = "FP"
        elif y_t == 1 and y_p == 0:
            outcome = "FN"
        else:
            outcome = "TP"

    for f in funcs:
        if f not in stats_dict:
            stats_dict[f] = {"N": 0, "Correct": 0, "TN": 0, "FP": 0, "FN": 0, "TP": 0}
        stats_dict[f]["N"] += 1
        stats_dict[f]["Correct"] += correct
        if is_binary:
            stats_dict[f][outcome] += 1

def _extract_positive_shap(shap_values):
    """
    Robustly extract SHAP values for the positive class across SHAP versions:
    - list of arrays (binary: [class0, class1])
    - array (n_samples, n_features) (already positive)
    - array (n_samples, n_features, n_classes) (take class=1 if exists)
    """
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    sv = np.asarray(shap_values)

    if sv.ndim == 2:
        return sv

    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) OR (n_classes, n_samples, n_features) in some odd cases
        if sv.shape[-1] in (2, 3, 4, 5):
            class_idx = 1 if sv.shape[-1] > 1 else 0
            return sv[:, :, class_idx]
        # alternative orientation: (n_classes, n_samples, n_features)
        if sv.shape[0] in (2, 3, 4, 5):
            class_idx = 1 if sv.shape[0] > 1 else 0
            return sv[class_idx, :, :]
        return sv.mean(axis=-1)

    return np.squeeze(sv)

def evaluate_model(X, y, df_meta, label_name, binary=False):
    acc_all_list = []
    acc_monomer_list = []

    tn_pct_list, fp_pct_list, fn_pct_list, tp_pct_list = [], [], [], []
    tn_pct_mono_list, fp_pct_mono_list, fn_pct_mono_list, tp_pct_mono_list = [], [], [], []

    for run in range(runs):
        seed = 100 + run

        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X, y, df_meta, test_size=test_size, stratify=y, random_state=seed
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --- Feature selection ---
        if USE_FEATURE_SELECTION:
            selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=seed))
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            selected_features = np.array(descriptor_cols)[selector.get_support()]
        else:
            selected_features = np.array(descriptor_cols)

        # --- Model ---
        clf = RandomForestClassifier(
            n_estimators=150,
            min_samples_split=2,
            max_depth=None,
            max_features=15,
            min_samples_leaf=1,
            class_weight="balanced",
            bootstrap=True,
            random_state=seed,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # --- Accuracies ---
        acc_all = accuracy_score(y_test, y_pred) * 100
        acc_all_list.append(acc_all)

        monomer_mask = (df_test["Monomer"] == "YES") if ("Monomer" in df_test.columns) else np.zeros(len(df_test), dtype=bool)
        if monomer_mask.sum() > 0:
            acc_monomer = accuracy_score(y_test[monomer_mask], y_pred[monomer_mask]) * 100
        else:
            acc_monomer = np.nan
        acc_monomer_list.append(acc_monomer)

        tag = "Binary (2 vs others)" if binary else "Original"

        # --- Confusion metrics (binary) ---
        conf_all = {"TN_pct": np.nan, "FP_pct": np.nan, "FN_pct": np.nan, "TP_pct": np.nan, "N_test": len(y_test)}
        conf_mono = {"TN_pct": np.nan, "FP_pct": np.nan, "FN_pct": np.nan, "TP_pct": np.nan, "N_test_monomer": int(monomer_mask.sum())}

        if binary:
            conf_all_full = _confusion_percentages_binary(y_test, y_pred)
            conf_all = {
                "TN_pct": conf_all_full["TN_pct"],
                "FP_pct": conf_all_full["FP_pct"],
                "FN_pct": conf_all_full["FN_pct"],
                "TP_pct": conf_all_full["TP_pct"],
                "N_test": conf_all_full["N_test"],
            }
            tn_pct_list.append(conf_all["TN_pct"])
            fp_pct_list.append(conf_all["FP_pct"])
            fn_pct_list.append(conf_all["FN_pct"])
            tp_pct_list.append(conf_all["TP_pct"])

            if monomer_mask.sum() > 0:
                conf_m = _confusion_percentages_binary(y_test[monomer_mask], y_pred[monomer_mask])
                conf_mono = {
                    "TN_pct": conf_m["TN_pct"],
                    "FP_pct": conf_m["FP_pct"],
                    "FN_pct": conf_m["FN_pct"],
                    "TP_pct": conf_m["TP_pct"],
                    "N_test_monomer": conf_m["N_test"],
                }
                tn_pct_mono_list.append(conf_m["TN_pct"])
                fp_pct_mono_list.append(conf_m["FP_pct"])
                fn_pct_mono_list.append(conf_m["FN_pct"])
                tp_pct_mono_list.append(conf_m["TP_pct"])

        # --- Function-level evaluation (TEST set) ---
        if "Harmonized_functions" in df_test.columns:
            func_stats = {}
            func_series = df_test["Harmonized_functions"].reset_index(drop=True)

            y_test_arr = np.asarray(y_test)
            y_pred_arr = np.asarray(y_pred)

            for i in range(len(df_test)):
                funcs = _parse_functions_cell(func_series.iloc[i])
                _update_function_stats(func_stats, funcs, int(y_test_arr[i]), int(y_pred_arr[i]), is_binary=binary)

            for func_name, st in func_stats.items():
                n = int(st["N"])
                function_performance.append({
                    "Target": label_name,
                    "Mode": tag,
                    "Run": run + 1,
                    "Function": func_name,
                    "N_test_in_function": n,
                    "Correct_in_function": int(st["Correct"]),
                    "Accuracy_in_function": _safe_pct(st["Correct"], n),
                    "TN_in_function": int(st["TN"]) if binary else np.nan,
                    "FP_in_function": int(st["FP"]) if binary else np.nan,
                    "FN_in_function": int(st["FN"]) if binary else np.nan,
                    "TP_in_function": int(st["TP"]) if binary else np.nan,
                })

        # --- Print ---
        if binary:
            print(
                f"{label_name} | {tag} | Run {run+1:2d}: "
                f"Acc(All)={acc_all:.2f}% | "
                f"TN={conf_all['TN_pct']:.2f}% FP={conf_all['FP_pct']:.2f}% "
                f"FN={conf_all['FN_pct']:.2f}% TP={conf_all['TP_pct']:.2f}% | "
                f"Acc(Monomer=YES)={acc_monomer:.2f}%"
            )
        else:
            print(f"{label_name} | {tag} | Run {run+1:2d}: All = {acc_all:.2f}%, Monomer=YES = {acc_monomer:.2f}%")

        # --- Save run-level result ---
        results.append({
            "Target": label_name,
            "Mode": tag,
            "Run": run + 1,
            "Accuracy_All": acc_all,
            "Accuracy_Monomer": acc_monomer,
            "TN_pct_All": conf_all["TN_pct"],
            "FP_pct_All": conf_all["FP_pct"],
            "FN_pct_All": conf_all["FN_pct"],
            "TP_pct_All": conf_all["TP_pct"],
            "TN_pct_Monomer": conf_mono["TN_pct"],
            "FP_pct_Monomer": conf_mono["FP_pct"],
            "FN_pct_Monomer": conf_mono["FN_pct"],
            "TP_pct_Monomer": conf_mono["TP_pct"],
            "N_test_All": conf_all["N_test"],
            "N_test_Monomer": conf_mono["N_test_monomer"],
        })

        # --- Feature importance and SHAP (binary only) ---
        if binary:
            fi = clf.feature_importances_
            for f, imp in zip(selected_features, fi):
                feature_importances.append({
                    "Target": label_name,
                    "Run": run + 1,
                    "Feature": f,
                    "Importance": float(np.ravel(imp)[0])
                })

            # SHAP (robust scalar conversion)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)

            shap_pos = _extract_positive_shap(shap_values)
            shap_pos = np.asarray(shap_pos)

            if monomer_mask.sum() > 0:
                shap_monomer = shap_pos[monomer_mask]
                mean_abs_shap = np.abs(shap_monomer).mean(axis=0)

                mean_abs_shap = np.asarray(mean_abs_shap).reshape(-1)

                for f, val in zip(selected_features, mean_abs_shap):
                    shap_values_storage.append({
                        "Target": label_name,
                        "Run": run + 1,
                        "Feature": f,
                        "MeanAbsSHAP_Monomer": float(np.ravel(val)[0])
                    })

    # --- Summary ---
    tag = "Binary (2 vs others)" if binary else "Original"
    mean_all = float(np.nanmean(acc_all_list))
    std_all = float(np.nanstd(acc_all_list))

    mono_vals = [a for a in acc_monomer_list if not np.isnan(a)]
    if len(mono_vals) > 0:
        mean_mono = float(np.nanmean(mono_vals))
        std_mono = float(np.nanstd(mono_vals))
    else:
        mean_mono, std_mono = np.nan, np.nan

    print(f"\n--- Summary for {label_name} | {tag} ---")
    print(f"  Mean Accuracy (All): {mean_all:.2f}% ± {std_all:.2f}%")
    if not np.isnan(mean_mono):
        print(f"  Mean Accuracy (Monomer=YES): {mean_mono:.2f}% ± {std_mono:.2f}%")
    else:
        print("  Monomer=YES subset not present in any run.\n")

    if binary and len(tn_pct_list) > 0:
        tn_mean, fp_mean, fn_mean, tp_mean = map(float, [
            np.nanmean(tn_pct_list), np.nanmean(fp_pct_list),
            np.nanmean(fn_pct_list), np.nanmean(tp_pct_list)
        ])
        tn_std, fp_std, fn_std, tp_std = map(float, [
            np.nanstd(tn_pct_list), np.nanstd(fp_pct_list),
            np.nanstd(fn_pct_list), np.nanstd(tp_pct_list)
        ])

        print("  Mean Confusion Percentages (All):")
        print(f"    TN={tn_mean:.2f}% ± {tn_std:.2f}% | FP={fp_mean:.2f}% ± {fp_std:.2f}% | FN={fn_mean:.2f}% ± {fn_std:.2f}% | TP={tp_mean:.2f}% ± {tp_std:.2f}%")

        tn_mean_m = fp_mean_m = fn_mean_m = tp_mean_m = np.nan
        tn_std_m = fp_std_m = fn_std_m = tp_std_m = np.nan
        if len(tn_pct_mono_list) > 0:
            tn_mean_m, fp_mean_m, fn_mean_m, tp_mean_m = map(float, [
                np.nanmean(tn_pct_mono_list), np.nanmean(fp_pct_mono_list),
                np.nanmean(fn_pct_mono_list), np.nanmean(tp_pct_mono_list)
            ])
            tn_std_m, fp_std_m, fn_std_m, tp_std_m = map(float, [
                np.nanstd(tn_pct_mono_list), np.nanstd(fp_pct_mono_list),
                np.nanstd(fn_pct_mono_list), np.nanstd(tp_pct_mono_list)
            ])
            print("  Mean Confusion Percentages (Monomer=YES):")
            print(f"    TN={tn_mean_m:.2f}% ± {tn_std_m:.2f}% | FP={fp_mean_m:.2f}% ± {fp_std_m:.2f}% | FN={fn_mean_m:.2f}% ± {fn_std_m:.2f}% | TP={tp_mean_m:.2f}% ± {tp_std_m:.2f}%")

        results.append({
            "Target": label_name,
            "Mode": tag,
            "Run": "Summary",
            "Accuracy_All": mean_all,
            "Accuracy_Monomer": mean_mono,
            "Std_All": std_all,
            "Std_Monomer": std_mono,
            "TN_pct_All": tn_mean,
            "FP_pct_All": fp_mean,
            "FN_pct_All": fn_mean,
            "TP_pct_All": tp_mean,
            "TN_pct_All_std": tn_std,
            "FP_pct_All_std": fp_std,
            "FN_pct_All_std": fn_std,
            "TP_pct_All_std": tp_std,
            "TN_pct_Monomer": tn_mean_m,
            "FP_pct_Monomer": fp_mean_m,
            "FN_pct_Monomer": fn_mean_m,
            "TP_pct_Monomer": tp_mean_m,
            "TN_pct_Monomer_std": tn_std_m,
            "FP_pct_Monomer_std": fp_std_m,
            "FN_pct_Monomer_std": fn_std_m,
            "TP_pct_Monomer_std": tp_std_m,
        })
    else:
        results.append({
            "Target": label_name,
            "Mode": tag,
            "Run": "Summary",
            "Accuracy_All": mean_all,
            "Accuracy_Monomer": mean_mono,
            "Std_All": std_all,
            "Std_Monomer": std_mono,
        })


# --- Main loop ---
for target in targets:
    print(f"\n========== Evaluating target: {target} ==========")

    df_target = df.dropna(subset=[target]).copy()

    X_raw = df_target[descriptor_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X_raw)

    # Original multi-class
    y_orig = LabelEncoder().fit_transform(df_target[target])
    evaluate_model(X, y_orig, df_target, label_name=target, binary=False)

    # Binary
    y_num = pd.to_numeric(df_target[target], errors="coerce")
    y_bin = (y_num > 0.9).astype(int)

    if y_bin.nunique() < 2:
        print(f"Skipping binary task for {target} — only one class present.")
    else:
        evaluate_model(X, y_bin, df_target, label_name=target, binary=True)


# --- Save run-level outputs ---
pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)
pd.DataFrame(feature_importances).to_csv("feature_importances.csv", index=False)
pd.DataFrame(shap_values_storage).to_csv("shap_values.csv", index=False)
pd.DataFrame(function_performance).to_csv("function_performance_runs.csv", index=False)

# --- Aggregate function performance across runs ---
if function_performance:
    fp_df = pd.DataFrame(function_performance)
    fp_df.columns = fp_df.columns.astype(str).str.strip()

    if "N_test_in_function" not in fp_df.columns and "Correct_in_function" in fp_df.columns:
        fp_df["N_test_in_function"] = fp_df.groupby(["Target","Mode","Run","Function"])["Correct_in_function"].transform("count")

    fp_summary = fp_df.groupby(["Target", "Mode", "Function"], as_index=False).agg(
        Runs_Observed=("Run", "count"),
        Total_N_test=("N_test_in_function", "sum"),
        Total_Correct=("Correct_in_function", "sum"),
        Mean_Accuracy=("Accuracy_in_function", "mean"),
        Std_Accuracy=("Accuracy_in_function", "std"),
    )
    fp_summary["Pooled_Accuracy"] = 100.0 * fp_summary["Total_Correct"] / fp_summary["Total_N_test"].replace(0, np.nan)

    needed_counts = {"TN_in_function", "FP_in_function", "FN_in_function", "TP_in_function"}
    if needed_counts.issubset(fp_df.columns):
        fp_bin_sum = fp_df[fp_df["Mode"] == "Binary (2 vs others)"].groupby(["Target", "Mode", "Function"], as_index=False).agg(
            TN=("TN_in_function", "sum"),
            FP=("FP_in_function", "sum"),
            FN=("FN_in_function", "sum"),
            TP=("TP_in_function", "sum"),
            N=("N_test_in_function", "sum"),
        )
        fp_bin_sum["TN_pct_pooled"] = 100.0 * fp_bin_sum["TN"] / fp_bin_sum["N"].replace(0, np.nan)
        fp_bin_sum["FP_pct_pooled"] = 100.0 * fp_bin_sum["FP"] / fp_bin_sum["N"].replace(0, np.nan)
        fp_bin_sum["FN_pct_pooled"] = 100.0 * fp_bin_sum["FN"] / fp_bin_sum["N"].replace(0, np.nan)
        fp_bin_sum["TP_pct_pooled"] = 100.0 * fp_bin_sum["TP"] / fp_bin_sum["N"].replace(0, np.nan)

        fp_summary = fp_summary.merge(
            fp_bin_sum[["Target", "Mode", "Function", "TN_pct_pooled", "FP_pct_pooled", "FN_pct_pooled", "TP_pct_pooled"]],
            on=["Target", "Mode", "Function"],
            how="left"
        )
    else:
        fp_summary["TN_pct_pooled"] = np.nan
        fp_summary["FP_pct_pooled"] = np.nan
        fp_summary["FN_pct_pooled"] = np.nan
        fp_summary["TP_pct_pooled"] = np.nan

    fp_summary = fp_summary.sort_values(
        by=["Target", "Mode", "Pooled_Accuracy", "Total_N_test"],
        ascending=[True, True, False, False]
    )
    fp_summary.to_csv("function_performance_summary.csv", index=False)

print(
    "\n Results saved:"
    "\n  - evaluation_results.csv"
    "\n  - feature_importances.csv"
    "\n  - shap_values.csv"
    "\n  - function_performance_runs.csv"
    "\n  - function_performance_summary.csv"
)
