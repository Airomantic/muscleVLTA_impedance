## 4. Experiments

### 4.1. Setup

**Platform and task.** We use a Rokae AR3 7-DoF arm under Cartesian impedance control with a DM Tac G2 two-finger visuotactile gripper, a front and a wrist RGB camera (640 × 480), and one H100 GPU serving the policy. In *occluded charger insertion* a two-pronged charger lies on the table; the robot grasps it and inserts it into a socket. Once the plug is above the socket, plug and gripper hide the holes and the prongs from both cameras, so the final alignment cannot be seen. We evaluate two start conditions: the gripper starts within 20 cm of the socket (*near*), or 40–50 cm away (*far*), where the approach before the occlusion is long. {{Protocol: success criterion, time limit, variation of start and plug poses (B2).}}

**Low-cost force sensing.** The arm carries no add-on wrist force–torque sensor: joint torques and TCP wrench are the built-in readings of its controller, which suffice for compliant control but are repeatable rather than accurate. At rest the reported force varies by 0.14–0.23 N per axis, yet with neither finger touching anything its magnitude has a median of 2.9–3.6 N over the three demonstration sets (95th percentile 9–19 N), an offset that depends on posture and motion (Section 3.4). The fingertip sensors measure the force between finger and object directly; their reading in contact is 12–75 times that of a free finger. They are meant to supply the contact detail that the arm cannot resolve, which Section 4.3 tests. {{Sensing principle, rated accuracy and units of the built-in wrench (B8).}}

**Data and training.** All policies are fine-tuned from the π_{0.5} base checkpoint on teleoperated demonstrations recorded at 10 Hz on this platform: 130 insertion episodes from two sessions (65 episodes of 30 s and 65 of 35 s) and 84 unplugging episodes, 67,700 frames in total. {{130 or 120 demonstrations? (B3)}} We train for 141k steps with batch size 128 on four H100 GPUs, AdamW, a learning rate of 5 × 10^{−5} after 10k warm-up steps, gradient clipping at 1.0 and an EMA of 0.999. {{Confirm the evaluated checkpoint: step 141,000 (B2).}}

**Compared methods.** π_{0.5} [@pi05] sees images and language only. LIFT [@lift] adds a reactive expert that reads the recent end-effector force. *Ours w/o touch* removes the hand stream and its losses from muscleVLTA. ForceVLA2 [@forcevla2] and GR00T [@gr00tn1] are trained on the same demonstrations. {{For every baseline: inputs on this robot, training budget, same controller and start poses. State the GR00T version (N1.5–N1.7); the code of GR00T 2, announced on 9 November 2026, is scheduled for December (B1).}}

### 4.2. Real-robot results

Table 1 reports success counts over 60 trials per cell with 95% Wilson intervals [@wilson] (shown as a chart in the supplementary material). From the far start, π_{0.5} never completes the insertion (0/60), the model without touch succeeds in 3/60 trials and muscleVLTA in 13/60. The difference between muscleVLTA and its variant without touch is significant (Fisher's exact test, *p* = 0.014), as is the difference to π_{0.5} (*p* = 1.2 × 10^{−4}); the variant without touch is not significantly better than π_{0.5} (*p* = 0.24). From the near start muscleVLTA succeeds in 55/60 trials. {{+ baseline counts and one comparing sentence (B1)}}

The gap between the two start conditions (91.7% versus 21.7%) locates the remaining difficulty: errors accumulated during a long approach must be corrected after the socket has disappeared from view, with a contact-rich search that 130 demonstrations cover only sparsely.

!table column 2.1 1.45 1.45
caption: Table 1. Occluded charger insertion on the real robot: successes / trials and success rate with its 95% Wilson interval. Highlighted cells have not been measured yet.
| Method | Near start (≤ 20 cm) | Far start (40–50 cm)
| π_{0.5} [@pi05] | {{TBD}} | 0/60, 0.0% [0.0, 6.0]
| GR00T [@gr00tn1] | {{TBD}} | {{TBD}}
| ForceVLA2 [@forcevla2] | {{TBD}} | {{TBD}}
| LIFT [@lift] | {{TBD}} | {{TBD}}
| Ours w/o touch | {{TBD}} | 3/60, 5.0% [1.7, 13.7]
| muscleVLTA, shared fusion | 55/60, 91.7% [81.9, 96.4] | 13/60, 21.7% [13.1, 33.6]
| muscleVLTA, balanced fusion | {{TBD}} | {{TBD}}

### 4.3. What does the policy read?

Adding a sensor to a policy does not mean that the policy uses it. We measure reliance directly: every action chunk is sampled twice with the same noise, once from the recorded readings and once from *counterfactual* readings in which one stream is removed or exchanged with that of another sample. We report the change of the chunk in standard deviations of the recorded actions, separately for chunks that start in free space and with a finger in contact. For the arm we distinguish the *level* of the force window, by giving the recorded changes the level of another sample, from its *changes*, by freezing the readings at the chunk start.

!table column 2.6 1.2 1.2
caption: Table 2. Counterfactual sensor attribution of the trained shared-fusion policy (512 chunks of its training episodes): change of the action chunk in standard deviations of the recorded actions (translation).
| Counterfactual readings | Free space | Finger in contact
| Arm force level swapped, changes kept | 0.32 | 0.45
| Arm force changes removed, level kept | 0.04 | 0.06
| Sensor history removed | 0.13 | 0.19
| Touch removed (both fingers flagged missing) | 0.002 | 0.003
| Touch and aperture of another sample | 0.003 | 0.007
| All sensors removed | 0.19 | 0.31

Table 2 gives two findings. First, the policy follows the level of the arm force far more than its changes. In free space the first three joint torques are predictable from the TCP pose alone (*R*² of 0.85–0.95 on held-out episodes), so their level stands in for the proprioceptive state that the policy does not receive. Second, the actions are practically invariant to the tactile input: removing both fingers, or giving the policy the touch of another sample, moves the chunk by less than 0.01 standard deviations.

The real-robot gain of the visuotactile model over its variant without touch (13/60 versus 3/60) can therefore not be explained by the policy reading touch at test time. It is consistent with touch acting through training: the contact gate and the forward model make the arm encoder predict fingertip wrenches and contact geometry from force and commands, and this supervision reaches the tokens that the policy does read. {{Two experiments decide this (B4): (i) the evaluated checkpoint with both fingers masked at test time, where unchanged success confirms that touch is needed in training only; (ii) retraining with the forward model but without hand tokens in the memory. Until then this paragraph states a hypothesis.}}

!figure column results_sensor_attribution
caption: Figure 3. Top: action change of the trained policy under counterfactual readings (Table 2). Bottom: increase of the held-out action error when the force level is swapped, for shared and balanced fusion (Table 3).

### 4.4. Balanced fusion

We compare shared and balanced fusion in matched pilot runs on the two insertion sets: 5k steps, batch size 32, one seed, with every eighth episode held out (16 of 130 episodes, 768 chunks). Table 3 reports the action error on the held-out episodes and its relative increase under two counterfactuals.

!table column 2.1 1.3 1.3 1.3
caption: Table 3. Shared versus balanced fusion on held-out episodes (5k-step pilots; free space / finger in contact). Error: mean squared error of the normalised actions, × 10^{−2}; the other columns give its relative increase under the counterfactual.
| Fusion | Error | Level swapped | Touch removed
| Shared | 5.05 / 5.36 | 44% / 58% | 0.2% / 1.5%
| Balanced | 4.98 / 5.36 | 25% / 34% | 1.6% / 1.1%
| Balanced, *p*_{arm} = 0.3 | 5.93 / 6.41 | 10% / 16% | 1.2% / 4.1%
| No load token, *p*_{arm} = 0.3 | 6.03 / 6.93 | 0% / 0% | 1.3% / 6.9%

Balanced fusion reaches the held-out error of shared fusion and depends less on the force level: swapping the level raises the error by 25% and 34% instead of 44% and 58% and moves the actions by 0.20 and 0.23 standard deviations instead of 0.28 and 0.31, while freezing the force changes now moves them by 0.13 and 0.10 instead of 0.06 and 0.07. The policy thus reads the interaction force more and the posture-dependent load less, which should make it less sensitive to offsets such as sensor bias or a changed payload. Discarding the load token makes the actions invariant to the level by construction but raises the error further (last row against the row above it), by 2% in free space and 8% in contact: the posture information carried by the level is useful, and the load token keeps it in one place.

Reliance on touch stays small in every variant. Removing touch raises the error of chunks in contact by 1.1–1.5%; dropping the interaction stream in training (*p*_{arm} = 0.3) raises this to 4.1%, and to 6.9% without the load token, but the held-out error itself then rises by 19–29%. We therefore keep *p*_{arm} = 0. These are short single-seed runs evaluated offline; {{+ real-robot result of balanced fusion with the full budget (B5)}}

### 4.5. Latency and temporal consistency

Replaying a held-out episode through the policy server exactly as the robot client queries it, one reactive request takes about 60 ms on one H100, within the 100 ms control period. The actions obtained step by step are identical to those obtained from one request that sees the whole recorded sensor sequence (maximum absolute difference 0), which verifies that no action depends on a row measured later than its latency allows.

### 4.6. Motion smoothness

Operators report that muscleVLTA moves the arm and the gripper more smoothly than the baselines. To quantify this we compute, on the TCP path recorded at 10 Hz, the spectral arc length (SPARC) and the log dimensionless jerk (LDLJ) [@sparc2012; @sparc], the RMS jerk [@flashhogan], and a *chunk-boundary ratio*: the mean second difference of the commanded position across action-chunk boundaries divided by that inside chunks, which is 1 when a new chunk continues as smoothly as the chunk itself. Table 4 gives the values of the human demonstrations as a reference, and the supplementary material shows the corresponding figure. {{Record the rollouts of every method and fill the table with scripts/ar3_smoothness.py (B6); until then this is an observation.}}

!table column 2.2 0.9 0.85 1.1 1.25
caption: Table 4. Motion smoothness (mean over episodes). SPARC and LDLJ: closer to 0 is smoother.
| Trajectories | SPARC | LDLJ | RMS jerk [m/s³] | Boundary ratio
| Demonstrations, 30 s session | −3.87 | −18.2 | 1.24 | 1.01
| Demonstrations, 35 s session | −4.29 | −18.5 | 1.21 | 1.01
| muscleVLTA | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}}
| LIFT [@lift] | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}}
| ForceVLA2 [@forcevla2] | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}}
| GR00T [@gr00tn1] | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}}
