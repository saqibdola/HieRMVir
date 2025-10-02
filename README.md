# HieRMVir

**HieRMVir** (Hierarchical Random forest and Mutual information-based Viral genome classifier) is a hierarchical deep learning framework for viral genome classification. It performs classification across three taxonomic levels: Virus/ERV/NonERV → Baltimore Classes → Viral Species.


## Features
- k-mer frequency extraction
- Random Forest-based feature importance weighting
- MI-guided attention regularization
- Modular and scalable pipeline
- Hierarchical classification: 
  - Level 1: Virus / ERV / NonERV  
  - Level 2: Baltimore Classes (C1–C7)  
  - Level 3: Viral Species

| Script | Description |
|--------|-------------|
| `newhierichalmodelwithvaryingkIUPAC.py` | Master pipeline to process FASTA, label, extract features, and run models |
| `step2withchunks4hierichial.py` | Chunked k-mer frequency extraction |
| `step3newwithoutchunking4hier.py` | Feature scaling based on Random Forest importances |
| `hiernewstep4withoutchunking_optimization.py` | Attention-based neural network training with MI guidance |

## Requirements

```text
python 3.10
pandas 2.1.3
numpy 1.26.0
torch 2.2.1
scikit-learn 1.3.2
joblib 1.3.2
tqdm 4.66.1
psutil5.9.6
```


## Experimental Protocol Details
Summary of the experimental setup:

### Train/Test Split
- **80% training / 20% testing**  
- **Stratified sampling** with a fixed random seed (**42**)  
- **No separate validation set**; performance was evaluated on the held-out test set  

### Preprocessing Pipeline
1. Random Forest-based feature importance** computed on a stratified 20% subsample  
2 Both RF and DL used the **same 80/20 split** and **scaled feature matrix**  
3. Ensures **full consistency and reproducibility**
