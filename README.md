# HieRMVir

**HieRMVir** (Hierarchical Random forest and Mutual information-based Viral genome classifier) is a hierarchical deep learning framework for viral genome classification. It performs classification across three taxonomic levels: Virus/ERV/NonERV → Baltimore Classes → Viral Species.



## Features

- k-mer extraction with chunking for scalability
- Feature importance scoring using Random Forest
- Mutual Information (MI)-guided attention supervision
- Hierarchical classification: 
  - Level 1: Virus / ERV / NonERV  
  - Level 2: Baltimore Classes (C1–C7)  
  - Level 3: Viral Species


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
pandas
numpy
torch
scikit-learn
tqdm
