# HieRMVir

**HieRMVir** (Hierarchical Random forest and Mutual information-based Viral genome classifier) is a hierarchical deep learning framework for viral genome classification. It performs classification across three taxonomic levels: Virus/ERV/NonERV → Baltimore Classes → Viral Species.

## Features
- k-mer frequency extraction
- Random Forest-based feature importance weighting
- MI-guided attention regularization
- Modular and scalable pipeline

## Structure
- `hierichalmodel.py`: Master script for the full pipeline
- `kmerextraction.py`: k-mer extraction with chunking
- `featuresimportance&scaling.py`: Feature importance & scaling
- `NNwithMIreg.py`: Neural network training with MI regularization
