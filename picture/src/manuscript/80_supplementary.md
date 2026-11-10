!pagebreak

!columns 1

## Supplementary Material

### A. Implementation details

!table wide 1.0 5.0
caption: Table S1. Settings of the evaluated model.
| Component | Setting
| Backbone | π_{0.5}: PaliGemma (SigLIP image encoder, Gemma-2B) and a 300M-parameter action expert
| Reactive expert | Copy of the action expert (18 layers); cross-attention to the sensor memory in every layer, output projection initialised to zero
| Action chunk | *H* = 10 actions at 10 Hz: TCP pose increments relative to the chunk start, absolute gripper target
| Sensor window | *K* = 10 history rows and *H* chunk rows; latency *L* = 3 rows (0.3 s)
| Arm stream | 7 joint torques and 6-D TCP wrench → GRU (hidden size 512) → 1024-D tokens
| Finger encoder | 3 strided convolutions on [*D*, Δ*D*] (2 × 48 × 64), 8 contact-patch features, MLP of the 6-D fingertip wrench; 256-D feature
| Contact label | Peak of the pooled indentation map above δ = 0.05 (sensor units)
| Hand stream | GRU over the two gated finger features, their contact probabilities and the gripper aperture
| Sensor dropout | Whole hand 10%, each finger 10%
| Loss weights | λ_{c} = 0.05 (contact gate), λ_{f} = 0.1 (forward model)
| Flow matching | *s* ~ Beta(1.5, 1); 10 Euler steps at inference
| Optimisation | AdamW, learning rate 5 × 10^{−5} after 10k warm-up steps, gradient clipping 1.0, EMA 0.999, batch size 128 on four H100 GPUs
| Deployment | Policy served from one H100; TCP targets streamed at 100 Hz to the Cartesian impedance controller

### B. Counterfactual readings

Every chunk is sampled from the recorded readings and, with the same noise, from each of the following variants. *Arm swapped:* the arm window of another sample. *Arm level swapped:* the recorded changes relative to the reading at the chunk start, added to the chunk-start reading of another sample. *Arm changes removed:* every row set to the reading at the chunk start. *Touch removed:* both fingers flagged as missing, so the policy sees *e*_{miss}; the gripper aperture is kept. *Touch swapped:* indentation maps, fingertip wrenches and aperture of another sample. *History removed:* the *K* history rows flagged invalid. *All sensors removed:* the whole memory masked. Table S2 extends Table 2 to rotation and gripper actions.

!table wide 2.0 1.25 1.25 1.25
caption: Table S2. Attribution of the trained shared-fusion policy for translation, rotation and gripper actions: change in standard deviations of the recorded actions, free space / finger in contact (305 / 207 chunks).
| Counterfactual readings | Translation | Rotation | Gripper
| Arm swapped | 0.351 / 0.470 | 0.281 / 0.303 | 0.015 / 0.021
| Arm level swapped | 0.323 / 0.448 | 0.242 / 0.262 | 0.014 / 0.027
| Arm changes removed | 0.042 / 0.061 | 0.049 / 0.067 | 0.004 / 0.005
| Touch removed | 0.002 / 0.003 | 0.002 / 0.004 | 0.001 / 0.002
| Touch swapped | 0.003 / 0.007 | 0.003 / 0.005 | 0.001 / 0.003
| History removed | 0.134 / 0.191 | 0.114 / 0.143 | 0.010 / 0.021
| All sensors removed | 0.191 / 0.314 | 0.144 / 0.202 | 0.012 / 0.025

### C. Motion smoothness

Figure S1 is the output of the smoothness tool for the three demonstration sets. In the typical episode of each set the jerk concentrates in a few short events in the last third of the episode, and the chunk-boundary ratio of the teleoperated commands is close to 1 (1.01–1.02), the reference for a policy whose chunks join without a step. {{Authors: record the rollouts of every method and regenerate this figure with them (one --run per method); the demonstrations then serve as the reference rows.}}

!figure wide results_smoothness
caption: Figure S1. Motion smoothness of the demonstrations: SPARC, log dimensionless jerk, RMS jerk and chunk-boundary ratio (mean over episodes, 95% bootstrap intervals), and the TCP speed and jerk of the median-SPARC episode of each set.

### D. Network with balanced fusion

!figure wide musclevlta_v2_architecture
caption: Figure S2. muscleVLTA with balanced fusion; NEW marks the differences from Figure 1. The arm reading is split into a load token (*x*_{ref}, the first valid reading of the window) and the interaction force *x* − *x*_{ref}, and the reactive expert keeps one attention distribution per stream (load, interaction, touch). Inputs, losses and the robot client are unchanged.

### E. Real-robot success rates

!figure column results_real_robot
caption: Figure S3. Success rates of Table 1 with 95% Wilson intervals. Dashed boxes are cells that have not been measured.
