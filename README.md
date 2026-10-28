# Federated Learning Framework for Smart Healthcare Diagnostics

Topics:Federated-learning, healthcare-ai, differential-privacy, privacy-preserving-machine-learning, medical-diagnostics, pytorch, personalized-federated-learning**.

> Semester Project | Apex Institute of Technology, Chandigarh University | 2026

---

## Table of Contents
- [Overview](#overview)
- [Key Idea](#key-idea)
- [Novelty](#novelty)
- [Results](#results)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team](#team)
- [License](#license)

---

## Overview

Hospitals hold valuable patient data, but privacy regulations prevent them from pooling it. Smaller or specialty-focused hospitals also rarely have enough diverse data to train accurate diagnostic models on their own.

This project shows how several hospitals can train a shared diagnostic model while keeping all patient data local, and then personalize that model for their own patient population. Differential Privacy protects the model updates that are shared between hospitals and the central server.

## Key Idea

1. Each hospital trains a model locally. Raw patient data never leaves the hospital.
2. Only model weights are sent to a central server, which combines them using **Federated Averaging (FedAvg)**.
3. The updated global model is sent back, and each hospital **personalizes** it to its own specialty.
4. **Differential Privacy** (gradient clipping + Gaussian noise) is applied to the shared updates.

## Novelty

Most existing work treats hospital differences as abstract statistical (non-IID) data splits. This project grounds heterogeneity in **real medical specialty differences** (Respiratory, Cardiology, Infectious Disease, General) and studies how a naive global model, personalization, and Differential Privacy behave under those conditions. The contribution is a realistic, rigorous application and evaluation of existing techniques, not a new algorithm.

---

## Results

| Hospital | Global-only Accuracy | Personalized Accuracy |
|---|---|---|
| Respiratory | ~0% | ~100% |
| Cardiology | ~0% | ~100% |
| Infectious Disease | ~9% | ~90% |
| General | ~97% | ~100% |

- The shared global model is dominated by the hospital with the most data (General) and **fails almost completely** on smaller specialty hospitals. This is a known FL effect (label-distribution skew / client drift).
- A lightweight personalization step resolves this almost entirely.
- Differential Privacy adds a measurable accuracy cost, most visible on the General hospital (about 74–79% with DP enabled), while the other hospitals stay largely stable.

*Exact numbers can vary slightly between runs and environments.*


<img width="1113" height="604" alt="Screenshot 2026-10-28 145021" src="https://github.com/user-attachments/assets/4c8189fd-82b1-4818-adad-c5616952282d" />




---

## System Architecture

<img width="1399" height="611" alt="Screenshot 2026-08-20 165436" src="https://github.com/user-attachments/assets/1357a65c-44e0-4e06-bfe8-665bb1e0ec67" />


**Workflow:** Hospitals (local data) → Local training → Differential Privacy on updates → Central aggregator (FedAvg) → Global model → Sent back to hospitals → Local personalization → Evaluation

---

## Dataset

**Disease Prediction Using Machine Learning** (public dataset)
- About 4,960 patient records
- 132 binary symptom features
- 41 diseases

Source: [Kaggle](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

Patients are partitioned into four simulated hospitals by medical specialty:

| Hospital | Example Diseases |
|---|---|
| Respiratory | Bronchial Asthma, Tuberculosis, Common Cold, Pneumonia |
| Cardiology | Hypertension, Heart Attack, Varicose Veins, Paralysis (brain hemorrhage) |
| Infectious Disease | Malaria, Dengue, Typhoid, Hepatitis A–E, Chicken Pox, AIDS, Alcoholic Hepatitis |
| General | All remaining diseases |

> This dataset has no oncology/cancer cases, so Infectious Disease was used as the fourth specialty.

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── code/
│   ├── README.md
│   └── FL_Healthcare_RealData.ipynb
└── docs/
    ├── flowchart.png
    └── results_comparison.png
```

## Tech Stack

- Python 3.x
- PyTorch
- Pandas, NumPy, Scikit-learn
- Matplotlib
- Google Colab

---

## Quick Start

**Google Colab (recommended)**
1. Open `code/FL_Healthcare_RealData.ipynb` in Google Colab.
2. Select Runtime → Run all.
3. The dataset downloads automatically. No login is needed.

**Local**
```bash
git clone https://github.com/vanshika2005kanwar/Federated-Learning-Framework-for-Smart-Healthcare-Diagnostics.git
cd Federated-Learning-Framework-for-Smart-Healthcare-Diagnostics
pip install -r requirements.txt
jupyter notebook code/FL_Healthcare_RealData.ipynb
```

See [`code/README.md`](code/README.md) for details on the pipeline and configuration.

---

## Limitations

- Differential Privacy uses a simplified DP-SGD style mechanism (gradient clipping + Gaussian noise) and does **not** compute a formal privacy budget (ε).
- Hospitals are **simulated** from a single public dataset, not real multi-institution data.
- Evaluation currently reports accuracy only.

## Future Work

- Formal privacy accounting using Opacus
- Precision, recall, F1-score and ROC-AUC evaluation
- Real clinical datasets (e.g., MIMIC-III/IV)
- Explainable AI (XAI) for clinical interpretability
- More hospital specialties, including oncology data

---

## Team

| Name | Roll No. |
|---|---|
| Vanshika Kanwar | [Roll No.] |
| [Teammate 2] | [Roll No.] |
| [Teammate 3] | [Roll No.] |
| [Teammate 4] | [Roll No.] |

**Supervisor:** [Supervisor Name]

## License

Released under the [MIT License](LICENSE).
