# Mechanistic Artificial Intelligence Interpretability

In the following result — we are looking at a real mechanistic interpretability localization experiment. In this experiment (attached code) we run an activation patching on the residual stream (resid_post) at the final token. In practice, we created a clean prompt (correct names), then a corrupted prompt in the form of swapped names, followed by measuring a decision metric (which name is preferred). Lastly, we then ran the corrupted prompt, patched in internal activations from the clean run at one layer and measured whether the model’s decision recovered.

<img width="1411" height="781" alt="Unknown-4" src="https://github.com/user-attachments/assets/311831a7-1261-4146-908d-8dc198d072a8" />

<img width="1373" height="779" alt="Unknown-5" src="https://github.com/user-attachments/assets/52389490-8a25-4f70-81a0-ae406e07d17b" />

