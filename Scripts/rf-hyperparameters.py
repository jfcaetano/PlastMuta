#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:54:35 2025

@author: jfcaetano
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Config (speed-focused)
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

target = "Mutagenic"
test_size = 0.2
seed = 123  # one split only (fast)

# RandomizedSearch settings (fast)
N_ITER = 20       
CV_FOLDS = 5     
VERBOSE = 1

# ----------------------------
# Data prep (do ONCE)
# ----------------------------
df_target = df.dropna(subset=[target]).copy()

X_raw = df_target[descriptor_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X = SimpleImputer(strategy="median").fit_transform(X_raw)

# choose task: original multiclass OR binary
TASK = "binary" 

if TASK == "original":
    y = LabelEncoder().fit_transform(df_target[target])
else:
    y = (df_target[target] > 0.9).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Binary task has only one class. Adjust threshold or target.")

# Split once
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df_target, test_size=test_size, stratify=y, random_state=seed
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# RF hyperparameter search (around your baseline)
# ----------------------------
base_rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=seed,
    n_jobs=-1
)

param_dist = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [15, 20, 25, None],
    "max_features": [5, 10, 15, None],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [2, 5, 10],
    "bootstrap": [True, False],
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring="accuracy", 
    cv=cv,
    random_state=seed,
    n_jobs=-1,
    verbose=VERBOSE
)

search.fit(X_train, y_train)

best_rf = search.best_estimator_
best_params = search.best_params_
best_cv = search.best_score_

# Evaluate on held-out test
y_pred = best_rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred) * 100.0

# Optional: monomer subset accuracy
monomer_mask = (df_test["Monomer"] == "YES").to_numpy()
test_acc_monomer = (
    accuracy_score(y_test[monomer_mask], y_pred[monomer_mask]) * 100.0
    if monomer_mask.sum() > 0 else np.nan
)

print("\nBest params:", best_params)
print(f"Best CV accuracy: {best_cv*100:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%")
print(f"Test accuracy (Monomer=YES): {test_acc_monomer:.2f}%")

# ----------------------------
# Save best result to CSV
# ----------------------------
best_row = {
    "Target": target,
    "Task": TASK,
    "Seed": seed,
    "TestSize": test_size,
    "CV_Folds": CV_FOLDS,
    "N_Iter": N_ITER,
    "BestCV_Accuracy": best_cv * 100.0,
    "Test_Accuracy_All": test_acc,
    "Test_Accuracy_Monomer": test_acc_monomer,
    **{f"param_{k}": v for k, v in best_params.items()},
}

pd.DataFrame([best_row]).to_csv("best_rf_result.csv", index=False)
print("\n✅ Saved: best_rf_result.csv")
