## 4. Experiments

### 4.1. Setup

**Platform and tasks.** We use a Rokae AR3 7-DoF arm under Cartesian impedance control with a DM Tac G2 two-finger visuotactile gripper, a front and a wrist RGB camera, and one H100 GPU serving the policy. The tasks are chosen so that the cameras cannot decide their outcome. In *occluded charger insertion* the robot grasps a two-pronged charger from the table and inserts it into a socket; once the plug is above the socket, plug and gripper hide the holes and the prongs from both cameras. The gripper starts within 20 cm of the socket (*near*) or 40–50 cm away (*far*). In *unplugging* the robot pulls the charger out and places it on the table. In *shake-to-identify* the robot picks a pill bottle from a drawer and shakes it; a bottle with pills is placed on top of the drawer, an empty one is put back and another bottle is tried, and whether a bottle is empty shows only in the force and touch response to the shake. {{Cap opening (B11); protocol of every task (B2, B9).}}

**Low-cost force sensing.** The arm carries no add-on wrist force–torque sensor: joint torques and TCP wrench are the built-in readings of its controller, which suffice for compliant control but are repeatable rather than accurate. With neither finger touching anything, the reported force has a median magnitude of 2.9–3.6 N, an offset that depends on posture and motion. The fingertip sensors measure the force between finger and object directly; their reading in contact is 12–75 times that of a free finger. {{Sensing principle (B8).}}

**Data and training.** All policies are fine-tuned from the π_{0.5} base checkpoint on teleoperated demonstrations recorded at 10 Hz: 130 insertion, 84 unplugging and 83 shake-to-identify episodes, 107,540 frames in total. The insertion results of Table 1 come from models trained on the 67,700 frames of the charger sets for 141k steps with batch size 128 on four H100 GPUs (AdamW, learning rate 5 × 10^{−5}, EMA 0.999). {{Checkpoints of the other tasks (B1, B5).}}

**Compared methods.** π_{0.5} [@pi05] and GR00T [@gr00tn1] see images and language only. ForceVLA2 [@forcevla2] and LIFT [@lift] additionally read the end-effector force. *muscleVLTA w/o touch* removes the hand stream and its losses and keeps the joint torques and the wrench. All are trained on the same demonstrations. {{Baseline details (B1).}}

### 4.2. Real-robot results

!table wide 1.9 1.9 1.9 1.0 1.25 1.0
caption: Table 1. Real-robot success: successes / trials, rate and 95% Wilson interval [@wilson]; insertion from a near (≤ 20 cm) and a far (40–50 cm) start. Highlighted cells have not been measured yet; ForceVLA2 near was recorded as a rate.
| Method | Insertion, near | Insertion, far | Unplugging | Shake-to-identify | Cap opening
| π_{0.5} [@pi05] | 1/60, 1.7% [0.3, 8.9] | 0/60, 0.0% [0.0, 6.0] | {{TBD}} | {{TBD}} | {{TBD}}
| GR00T [@gr00tn1] | 1/60, 1.7% [0.3, 8.9] | 0/60, 0.0% [0.0, 6.0] | {{TBD}} | {{TBD}} | {{TBD}}
| ForceVLA2 [@forcevla2] | 60.5% | 5/60, 8.3% [3.6, 18.1] | {{TBD}} | {{TBD}} | {{TBD}}
| LIFT [@lift] | 4/60, 6.7% [2.6, 15.9] | 2/60, 3.3% [0.9, 11.4] | {{TBD}} | {{TBD}} | {{TBD}}
| ---
| muscleVLTA w/o touch | 3/60, 5.0% [1.7, 13.7] | 3/60, 5.0% [1.7, 13.7] | {{TBD}} | {{TBD}} | {{TBD}}
| muscleVLTA, shared | **55/60, 91.7%** [81.9, 96.4] | **13/60, 21.7%** [13.1, 33.6] | {{TBD}} | {{TBD}} | {{TBD}}
| muscleVLTA, balanced | 53/60, 88.3% [77.8, 94.2] | **13/60, 21.7%** [13.1, 33.6] | {{TBD}} | {{TBD}} | {{TBD}}
| muscleVLTA-RL (ours) | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}}

Table 1 reports success counts over 60 trials per cell (a chart is given in the supplementary material). From the near start muscleVLTA succeeds in 55/60 trials. π_{0.5} and GR00T succeed once, LIFT in 4/60 and the model without touch in 3/60 trials (Fisher's exact test against muscleVLTA, *p* < 10^{−20} each); ForceVLA2 reaches 60.5%. From the far start π_{0.5} and GR00T never complete the insertion, LIFT succeeds in 2/60, the model without touch in 3/60, ForceVLA2 in 5/60 and muscleVLTA in 13/60 trials. The differences to π_{0.5} and GR00T (*p* = 1.2 × 10^{−4}), to LIFT (*p* = 0.004) and to the model without touch (*p* = 0.014) are significant; the difference to ForceVLA2 is not at the 5% level (*p* = 0.071). Balanced fusion matches shared fusion on the robot, with 53/60 and 13/60 (*p* = 0.76 and 1.0).

The gap between the two starts (91.7% versus 21.7%) locates the remaining difficulty: errors accumulated during a long approach must be corrected after the socket has disappeared from view. This is the regime that muscleVLTA-RL is meant for (Section 4.5). {{+ other tasks (B1).}}

### 4.3. What does the policy read?

Adding a sensor to a policy does not mean that the policy uses it. We measure reliance directly: every action chunk is sampled twice with the same noise, once from the recorded readings and once from *counterfactual* readings in which one stream is removed or exchanged with that of another sample, and we report the change of the chunk in standard deviations of the recorded actions. For the arm we distinguish the *level* of the force window, by giving the recorded changes the level of another sample, from its *changes*, by freezing the readings at the chunk start.

!table column 2.6 1.2 1.2
caption: Table 2. Counterfactual sensor attribution of the trained shared-fusion policy (512 chunks of its training episodes): change of the action chunk in standard deviations of the recorded actions (translation).
| Counterfactual readings | Free space | Finger in contact
| Arm force level swapped, changes kept | 0.32 | 0.45
| Arm force changes removed, level kept | 0.04 | 0.06
| Sensor history removed | 0.13 | 0.19
| Touch removed (both fingers flagged missing) | 0.002 | 0.003
| Touch and aperture of another sample | 0.003 | 0.007
| All sensors removed | 0.19 | 0.31

Table 2 gives two findings. First, the policy follows the level of the arm force far more than its changes. In free space the first three joint torques are predictable from the TCP pose alone (*R*² of 0.85–0.95 on held-out episodes), so their level stands in for the proprioceptive state that the policy does not receive; a load that the demonstrations do not contain, such as a payload or a dragging cable, would be read as a different posture. Second, the actions are practically invariant to the tactile input: removing both fingers, or giving the policy the touch of another sample, moves the chunk by less than 0.01 standard deviations.

The real-robot gain of the visuotactile model over its variant without touch (55/60 versus 3/60 and 13/60 versus 3/60) can therefore not be explained by the policy reading touch at test time. It is consistent with touch acting through training: the contact gate and the forward model make the arm encoder predict fingertip wrenches and contact geometry from force and commands, and this supervision reaches the tokens that the policy does read. {{A hypothesis until the test-time ablation is run (B4).}}

### 4.4. Balanced fusion

In matched 5k-step pilots with every eighth episode held out (Table S3 of the supplementary material), balanced fusion reaches the held-out error of shared fusion and depends less on the force level: swapping the level raises the error by 25% and 34% (free space and contact) instead of 44% and 58%. The policy thus reads the interaction force more and the posture-dependent load less. Discarding the load token makes the actions invariant to the level by construction but raises the error: the posture information carried by the level is useful, and the load token keeps it in one place. On the robot the fully trained balanced model reaches the success of shared fusion (Table 1), so the lower sensitivity to the force level costs no success; whether it pays off under a changed payload or sensor offset has not been tested.

### 4.5. Offline reinforcement learning

Before any rollout enters, we check the parts of muscleVLTA-RL offline on the charger demonstrations (Table 3). The cross-fitted critic predicts the time to success of episodes it has not seen with a mean absolute error of about 7% of the longest episode. Labels from critics with different seeds agree better for longer horizons, hence *n* = 50. As a positive control we assign the labels at random and shift the actions of the negative chunks by 10 mm: after 1000 steps the fine-tuned policy separates the two conditions by 9.8 mm on held-out chunks, so the indicator reaches the actions. With the labels of the demonstrations alone, which contain no failure, 3000 steps of fine-tuning leave the policy unchanged: the indicator moves the actions no more than an arbitrary appended phrase does. Improvement therefore has to come from rollouts with failures and corrections. {{Real-robot rounds and the last row of Table 1 (B10).}}

!table column 3.4 1.6
caption: Table 3. Offline checks of muscleVLTA-RL on the charger demonstrations.
| Quantity | Value
| Critic error, unplugging / insertion (held-out episodes) | 2.1 s / 2.6 s
| Label agreement between seeds, κ (*n* = 10 / 50 / episode end) | 0.52 / 0.67 / 0.80
| Positive control: action shift recovered (injected: 10 mm) | 9.8 mm [9.75, 9.91]
| Demonstrations only: effect of the indicator / of an arbitrary phrase | 0.35 mm / 0.28 mm
| Demonstrations only: held-out action error before / after | 0.72 mm / 0.80 mm

### 4.6. Latency and motion smoothness

One reactive request takes about 60 ms on one H100, within the 100 ms control period, and the actions obtained step by step equal those of one request that sees the whole recorded sensor sequence: no action depends on a row measured later than its latency allows.

We compute, on the TCP path recorded at 10 Hz, the spectral arc length (SPARC) and the log dimensionless jerk (LDLJ) [@sparc2012; @sparc], the RMS jerk [@flashhogan], and a *chunk-boundary ratio*: the mean second difference of the commanded position across action-chunk boundaries divided by that inside chunks (Table 4). muscleVLTA moves as smoothly as the human demonstrations in all four measures, and GR00T comes close. LIFT and ForceVLA2 are less smooth, with boundary ratios of 1.65 and 1.52: their consecutive chunks join with a step. {{Rollouts per method (B6).}}

!table column 2.2 0.9 0.85 1.1 1.25
caption: Table 4. Motion smoothness (mean over episodes). SPARC and LDLJ: closer to 0 is smoother.
| Trajectories | SPARC | LDLJ | RMS jerk [m/s³] | Boundary ratio
| Demonstrations, 30 s session | −3.87 | −18.2 | 1.24 | 1.01
| Demonstrations, 35 s session | −4.29 | −18.5 | 1.21 | 1.01
| ---
| GR00T [@gr00tn1] | −4.80 | −21.0 | 1.23 | 1.02
| ForceVLA2 [@forcevla2] | −5.61 | −31.77 | 2.68 | 1.52
| LIFT [@lift] | −5.12 | −27.92 | 1.29 | 1.65
| muscleVLTA | **−3.91** | **−18.3** | **1.22** | **1.01**
