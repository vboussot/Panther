[![Grand Challenge](https://img.shields.io/badge/Grand%20Challenge-PANTHER_2025-blue)](https://panther.grand-challenge.org/) 
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-PANTHER_2025-orange)](https://huggingface.co/VBoussot/Panther)

# PANTHER 2025: Pancreatic Tumor Segmentation on T2W MRI (2nd place 🥈)

This repository provides the full code and configurations used for our submission to the **PANTHER Challenge 2025**, focused on **automatic pancreatic tumor segmentation on MR-Linac T2-weighted MRI**.  
Our approach leverages **transfer learning**, **fine-tuning**, and **ensemble methods** to tackle the few-shot setting (50 annotated cases).

---

## 📊 Results

### 🏁 Official Challenge Leaderboard

| Rank | Team / User | Method                          | DSC (↑) | NSD (↑) | HD95 [mm] (↓) | ASD [mm] (↓) | Vol. [mm³] (↑) |
|------|-------------|---------------------------------|---------|---------|---------------|--------------|----------------|
| 1st  | MIC-DKFZ    | Bauchspeichel-Doppel-Düse       | **0.5289** | **0.6999** | **23.0110** | **5.1319** | 17163.5753 (4) |
| 2nd  | BreizhSeg   | docker_Task2                    | 0.4910  | 0.6736  | 25.2634 (4)   | 6.1722 (2)   | 16714.9043 (2) |
| 2nd  | LiboZhang   | Task2Baseline                   | 0.4814  | 0.6507  | 24.0864 (2)   | 7.6881 (3)   | **16647.2197 (1)** |
| 4th  | amparobt9   | PANTHER baseline (MRSegmentator)| 0.4784  | 0.6457  | 24.1735 (3)   | 8.1224 (4)   | 16846.2910 (3) |


🔗 [View the full leaderboard](https://panther.grand-challenge.org/evaluation/closed-testing-phase-task-2/leaderboard/)

---

## 🧩 Methods

Our approach explored three main strategies:

1. **MRI → CT Transfer**  
   - Used SynthRAD2025 models to generate synthetic CT (sCT) from T2 MRI https://github.com/vboussot/Synthrad2025_Task_1.  
   - Applied pretrained CT tumor segmentation models from Curvas2025 challenge https://github.com/vboussot/CurvasPDACVI.  
   - Limitation: tumors often blurred or absent in sCT.

2. **Direct MRI Fine-tuning**  
   - Fine-tuned a pretrained **PANORAMA ResUNetL** (winner of CT segmentation) on Panther T2W MRIs.  
   - Showed competitive performance but high inter-patient variability.  

3. **SAM 2.1 Fine-tuning with Prompts**  
   - Used method (2) segmentations as prompts for **SAM 2.1**.  
   - Fine-tuned SAM on Panther dataset. 
   - Improvement marginal; SAM struggled with medical tumor characteristics.  

### ✅ Final Submission  
We combined the panther curvas with multiple fine-tuned models with **STAPLE ensembling**, which provided the **most robust approach** across cases.  

---

## 📦 Models & Weights

- 🔗 **Our fine-tuned Panther weights**: [Hugging Face – VBoussot/Panther](https://huggingface.co/VBoussot/Panther)  
- 🔗 **Official baseline model (MRSegmentator v1.2.0)**: [GitHub Release](https://github.com/hhaentze/MRSegmentator/releases/tag/v1.2.0)  

---
