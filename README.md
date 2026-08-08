# PlastMuta
Supporting Information for the article "AI-Driven Mutagenic Screening Tool of Plastic Monomers for Instant SSbD Assessment"

<div align="center">
  <img src="Supporting Information/Figu1.png" alt="image" width="550" height="300">
</div>

## [Link to Published PDF](https://cmsweb.com.sg/rps2prod/esrel2026/epro/pdf/esrel26-p26196.pdf)

DOI: http://doi.org/10.3850/ESREL2026061419_esrel26-p26196-cd

## Script Overview
_rdkit-calc.py_: This script takes a CSV dataset of molecules, identifies all medium–radical combinations, and for each pair computes a series of RDKit descriptors for both the base SMILES and the radical SMILES. It assembles these descriptors into a new, expanded table as a CSV file.

_model_search.py_: Script tests several ML classification algorithms for the final selection.

rf-hyperparameters.py_: The code carries out a parallel grid search to tune a Random Forest regressor on the descriptor dataset, using a train–test split with preprocessing for categorical and numeric features and records each hyperparameter set’s test R² and MAE in a CSV file.

_model-calc.py_: This script performs the final classification model calculations, and outputs the final metrics presented in the article.

_class-separation.py_: Script for the analysis for possible non-overlapping scenario for Mutagenicity data labelled 0.5 and 0.
