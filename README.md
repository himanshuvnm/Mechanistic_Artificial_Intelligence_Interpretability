# Mechanistic Artificial Intelligence Interpretability

This repository contains a mechanistic interpretability study of transformer models using causal activation patching and circuit localization. The project applies intervention-based methods to reverse-engineer how GPT-style transformers perform relational reasoning tasks.

The core goal is to treat neural networks as computational systems and uncover the internal mechanisms and information flow that give rise to reasoning behavior.

---

## 🔬 Project Overview

This project implements a full causal interpretability pipeline using TransformerLens:

- Construction of clean and corrupted reasoning prompts  
- Definition of a causal decision metric  
- Residual stream activation patching across layers  
- Localization of reasoning circuits  
- Attention head analysis and information flow visualization  

The primary experiment studies indirect-object reasoning (e.g., "A gave the book to B. Therefore, ___") and identifies which transformer layers causally implement the relational inference.

Rather than relying on correlation analysis, this work uses causal interventions to localize the computation responsible for model decisions.

---

## 🧠 Key Results

Using activation patching on GPT-2-small:

- Localized the reasoning circuit to mid-to-late transformer layers (layers 8–10)
- Demonstrated causal dependence of output behavior on internal residual stream representations
- Identified the layer region where relational information is stored and transformed
- Produced layer-wise recovery curves and attention heatmaps

These results replicate known transformer reasoning phenomena and demonstrate a complete mechanistic analysis workflow.

---

## 📊 Methods

- TransformerLens (HookedTransformer)
- Residual stream activation patching
- Causal intervention on internal activations
- Layer-wise circuit localization
- Attention head inspection and visualization
- Decision-metric based evaluation

-- -- --

In the following figure, on x-axis we have Tranformer Layers index from 0-11 and y-axis plot the decision metric. We clearly see the dashed line value and solid line in the figure. More interpretation details are as follows:
| Layer Index | Interpretation |
|---|---|
| 0-1-2-3 | Weak influence (mostly syntax & token structure) | 
| 4-5-6 | Partial influence (entity tracking starts) | 
| 7-8-9 | Strong influence (core reasoning happens here)|
| 10 | Strong Influence |
| 11 | Weak Again (logit formatting layer) |

From the following figure **“Activation Patching: Which layers matter most?”** we see that the curve dipping strongly at layers 8, 9, 10 which means these layers are where the model is computing “who received the object”. This tells exactly about the known-reasoning issues of the transformer model, that is: *Early layers = form*; *Middle layers = meaning* and *Late layers = decision*. 

<img width="1411" height="781" alt="Unknown-4" src="https://github.com/user-attachments/assets/311831a7-1261-4146-908d-8dc198d072a8" />

From the following figure **Normalized Recovery by Layer**, we provide the following rationales:
| Layer Index | Recovery | Interpretation |
|---|---|---|
| 0-1-2 | approx 0.4 | Slight contribution |
| 3-4-5 | approx 0.5-0.7 | Partial Reasoning |
| 6-7 | approx 0.8 | Important |
| 8 | approx 1.1 | Core Reasoning |
| 9 | approx 1.2 | Core Reasoning |
| 10 | approx 1.15 | Core Reasoning |
| 11 | approx 0.33 | Mostly Formatting |

<img width="1373" height="779" alt="Unknown-5" src="https://github.com/user-attachments/assets/52389490-8a25-4f70-81a0-ae406e07d17b" />

The causal reasoning circuit lives primarily in layers 8–10. 

---
.
├── AI_Interpretability.ipynb     # Main experiment notebook
├── ai_interpretability.py       # Script version of core logic
├── requirements.txt             # Python dependencies
├── LICENSE
└── README.md

