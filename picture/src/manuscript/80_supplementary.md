!pagebreak

!columns 1

## Supplementary Material

### A. Implementation details

!table wide 1.0 5.0
caption: Table S1. Settings of the evaluated model and of muscleVLTA-RL.
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
| ---
| RL critic | 201-bin distributional value; inputs: mean-pooled SigLIP tokens of both cameras, TCP pose and aperture, 1 s arm history (GRU) and finger history (hand encoder); 4 folds by episode, 4000 steps each
| RL labels | *n* = 50 steps; positive: best 50% of the demonstration frames and 40% of the policy frames per task, all human corrections and the frames after completion
| RL fine-tuning | From the policy checkpoint with its normalisation statistics; constant learning rate 5 × 10^{−5}, 10k steps; “Advantage: negative” appended to the instruction of negative chunks

### B. Counterfactual readings

Every chunk is sampled from the recorded readings and, with the same noise, from each of the following variants. *Arm swapped:* the arm window of another sample. *Arm level swapped:* the recorded changes relative to the reading at the chunk start, added to the chunk-start reading of another sample. *Arm changes removed:* every row set to the reading at the chunk start. *Touch removed:* both fingers flagged as missing, so the policy sees *e*_{miss}; the gripper aperture is kept. *Touch swapped:* indentation maps, fingertip wrenches and aperture of another sample. *History removed:* the *K* history rows flagged invalid. *All sensors removed:* the whole memory masked. Table S2 extends Table 2 to rotation and gripper actions, and Figure S1 shows Tables 2 and S3 as charts.

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

!figure column results_sensor_attribution
caption: Figure S1. Top: action change of the trained policy under counterfactual readings (Table 2). Bottom: increase of the held-out action error when the force level is swapped, for shared and balanced fusion (Table S3).

### C. Training step and reactive rollout

!algorithm Algorithm S1. One training step of muscleVLTA
**Input:** demonstration window: images *I*, instruction ℓ, arm rows *x*_{−K:H−1}, hand rows, validity *v*, action chunk *A*
1: Drop the whole hand or single fingers (10% each) → *e*_{miss};  [balanced] drop the interaction stream w.p. *p*_{arm}
2: Prefix ← VLM(*I*, ℓ)
3: [balanced] *x*_{ref} ← first valid arm row;  load ← MLP(*x*_{ref});  *x* ← *x* − *x*_{ref}
4: *m*^{arm} ← GRU_{arm}(*x*, *v*);   (*m*^{hand}, *c*) ← HandEncoder(*D*, Δ*D*, *w*, *g*, *v*)
5: Sample *s*, ε, ε′;   *A*^{s} ← *s*ε + (1 − *s*)*A*;   *A*′^{s} ← *s*ε′ + (1 − *s*)*A*
6: (*u*, *u*′) ← Experts(prefix, *A*^{s}, *A*′^{s}, *s*; memory *m*, mask *M*)        // reactive expert cross-attends to *m*
7: ℒ_{nom} ← ‖*u* − (ε − *A*)‖²;   ℒ_{rea} ← ‖*u*′ − (ε′ − *A*)‖²
8: ℒ_{gate} ← BCE(*c*, 1[max *D* > δ]);   ℒ_{fwd} ← loss(*f*_{ψ}(*m*_{p}, *a*_{p−1}, *a*_{p}), *y*_{p+1})        // training only
9: Update θ, ψ with ∇(ℒ_{nom} + ℒ_{rea} + λ_{c}ℒ_{gate} + λ_{f}ℒ_{fwd});  update the EMA of θ

!algorithm Algorithm S2. Reactive rollout of one action chunk (10 Hz)
**Input:** cached policy, history buffer *B* of the last *K* arm and hand rows, horizon *H*, latency *L*
1: Observe images *I*, instruction ℓ;  cache ← VLM(*I*, ℓ);  sample ε, ε′ once
2: rows ← *B* ∪ {current row};   *n* ← 0        // *n*: newest measured chunk row
3: **for** *i* = 0, …, *H* − 1 **do**
4:     **if** action *i* not yet computed **then**
5:         integrate the flow of both experts for 10 steps on (cache, rows, mask *M*)        // one request, ≈ 60 ms
6:         keep reactive actions *i*, …, min(*n* + *L*, *H* − 1)        // all rows they may read exist
7:     execute action *i*: pose increment → TCP target → impedance controller (100 Hz stream)
8:     wait for the next control tick;  measure the new row;  *n* ← *n* + 1;  rows ← rows ∪ {row};  *B* ← *B* ∪ {row}
9: **end for**

### D. Network with shared fusion

!figure wide musclevlta_architecture
caption: Figure S2. muscleVLTA with shared fusion, the model of the “shared” row of Table 1. (a) Slow semantic stream and fast sensor stream; one attention distribution over the tokens of all streams. (b) Contact-gated hand encoder. (c) Visibility of the sensor rows for each action of the chunk. (d) Training objectives. (e) Real-robot rollout.

### E. Motion smoothness

Figure S3 is the output of the smoothness tool for the three charger demonstration sets. In the typical episode of each set the jerk concentrates in a few short events in the last third of the episode, and the chunk-boundary ratio of the teleoperated commands is close to 1 (1.01–1.02), the reference for a policy whose chunks join without a step. {{Authors: regenerate this figure with the recorded rollouts of every method (one --run per method), so that it shows the policy rows of Table 4.}}

!figure wide results_smoothness
caption: Figure S3. Motion smoothness of the demonstrations: SPARC, log dimensionless jerk, RMS jerk and chunk-boundary ratio (mean over episodes, 95% bootstrap intervals), and the TCP speed and jerk of the median-SPARC episode of each set.

### F. Real-robot success rates

!figure column results_real_robot
caption: Figure S4. Success rates of occluded insertion (Table 1) with 95% Wilson intervals. Dashed boxes are cells without a success count: not measured yet, or recorded as a rate (ForceVLA2, near start).

### G. Shared versus balanced fusion: offline pilots

We compare shared and balanced fusion in matched pilot runs on the two insertion sets: 5k steps, batch size 32, one seed, with every eighth episode held out (16 of 130 episodes, 768 chunks). Table S3 reports the action error on the held-out episodes and its relative increase under two counterfactuals. Swapping the force level moves the actions of the balanced model by 0.20 and 0.23 standard deviations instead of 0.28 and 0.31, while freezing the force changes moves them by 0.13 and 0.10 instead of 0.06 and 0.07. Discarding the load token makes the actions invariant to the level by construction but raises the error by 2% in free space and 8% in contact. Removing touch raises the error of chunks in contact by 1.1–1.5%; dropping the interaction stream in training (*p*_{arm} = 0.3) raises this to 4.1%, and to 6.9% without the load token, but the held-out error itself then rises by 19–29%.

!table wide 2.1 1.3 1.3 1.3
caption: Table S3. Shared versus balanced fusion on held-out episodes (5k-step pilots; free space / finger in contact). Error: mean squared error of the normalised actions, × 10^{−2}; the other columns give its relative increase under the counterfactual.
| Fusion | Error | Level swapped | Touch removed
| Shared | 5.05 / 5.36 | 44% / 58% | 0.2% / 1.5%
| Balanced | 4.98 / 5.36 | 25% / 34% | 1.6% / 1.1%
| Balanced, *p*_{arm} = 0.3 | 5.93 / 6.41 | 10% / 16% | 1.2% / 4.1%
| No load token, *p*_{arm} = 0.3 | 6.03 / 6.93 | 0% / 0% | 1.3% / 6.9%

### H. Discussion: what should a policy predict?

World action models (WAMs) are argued to generalise better than VLAs because predicting future video makes them learn dynamics [@dreamzero; @wamstudy]. The open questions are what to predict and when, and the WAM literature states them itself: Fast-WAM remains competitive without future prediction at test time [@fastwam], and the GR00T 2 whitepaper calls it unclear “whether prediction helps mainly as a training signal or also as imagination at inference time” and lists as open “how to add force and touch as inputs without eroding visual priors, and at which control rate” [@gr00t2]. muscleVLTA gives one answer for the regime of contact. It predicts the next arm force and fingertip readings, a few dozen numbers in the modality that carries the outcome of an occluded insertion; it predicts in training only; and it adds force and touch through a reactive expert whose cross-attention starts at zero, at 10 Hz and under an explicit latency. Force prediction and video prediction are complementary, and the same objective could supervise a WAM [@tactilewam]. We have not evaluated a WAM on our tasks, so this is a design argument and not an experimental ranking.
