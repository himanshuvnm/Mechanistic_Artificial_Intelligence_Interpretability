# Mechanistic Artificial Intelligence Interpretability

In the following result — we are looking at a real mechanistic interpretability localization experiment. In this experiment (attached code) we run an activation patching on the residual stream (resid_post) at the final token. In practice, we created a clean prompt (correct names), then a corrupted prompt in the form of swapped names, followed by measuring a decision metric (which name is preferred). Lastly, we then ran the corrupted prompt, patched in internal activations from the clean run at one layer and measured whether the model’s decision recovered.

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

