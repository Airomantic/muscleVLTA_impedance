!pagebreak

!columns 1

## Author checklist — remove before submission

This page is for the authors and is not part of the paper. It lists where every number comes from and what is still missing. Yellow text in the paper marks the open items; the codes B1–B7 refer to part B below.

### A. Numbers in the paper and their sources

**Reported by the authors (real robot).** Occluded charger insertion, 60 trials per cell: π_{0.5} 0/60, muscleVLTA without touch 3/60 and muscleVLTA 13/60 from the far start (40–50 cm); muscleVLTA 55/60 from the near start (within 20 cm). The Wilson intervals and Fisher tests of Section 4.2 are computed from these counts by scripts/build_result_figures.py; edit picture/src/results.json and re-run it when counts change. These results belong to the shared-fusion model (config pi05_ar3_muscle_vlta, run muscle_vlta_30k_plugin_30s35s). They must not be attributed to balanced fusion.

**Measured offline (GPU server, commit d1283cf).** Tables 2 and S2: scripts/ar3_sensor_attribution.py on checkpoint 141000 of that run. Table 3: four 5k-step pilots on the two insertion sets with OPENPI_AR3_HOLDOUT_EVERY=8; three checkpoints are kept in /data/jiang/train_checkpoints_pilot, the run without load token was evaluated and removed. Section 4.5: scripts/ar3_lift_replay_check.py against the balanced-fusion pilot served by scripts/serve_ar3_lift.sh. Table 4 (demonstration rows) and Figure S1: scripts/ar3_smoothness.py on the demonstration sets. Contact fractions (44% and 26% of the frames): peak of the pooled indentation map above the contact threshold, from the tactile caches of the three sets. The *R*² values of Section 4.3: regression of the free-space joint torques (gripper open) on pose features, tested on held-out episodes of the 30 s insertion set.

**Literature.** The 50 references were checked against arXiv, publisher or Crossref records on 10 November 2026. The GR00T 2 statements quoted in Section 5 are on pages 26–27 of the whitepaper, the Fast-WAM statements in its abstract. Venues of recent preprints may change before the camera-ready version.

### B. Still missing (yellow in the paper)

**B1.** Baseline counts for Table 1 at both start distances (LIFT, ForceVLA2, GR00T), π_{0.5} and the model without touch from the near start, and for every baseline its inputs on this robot and its training budget. State which GR00T version was run: the code of GR00T 2 is announced for December 2026, the versions that can be fine-tuned today are N1.5 to N1.7.

**B2.** Evaluation protocol: success criterion, time limit, variation of start and plug poses, and the evaluated checkpoint step.

**B3.** Number of demonstrations: the sets used for training hold 130 insertion episodes (65 + 65) and 84 unplugging episodes; the earlier abstract said 120.

**B4.** Test-time touch ablation: run the evaluated checkpoint with both fingers masked (add --mask-tactile to scripts/run_ar3_lift_robot.sh; same server, same protocol, 60 trials from the far start). This decides whether the paper may say that touch is needed in training only (Section 4.3), and it is the first experiment a reviewer will ask for. It needs no training.

**B5.** Real-robot result of balanced fusion. The full training (config pi05_ar3_muscle_vlta_v2, run muscle_vlta_v2_3sets) was started on 10 November 2026; evaluate a checkpoint with the same 60-trial protocol and check its held-out error with scripts/ar3_sensor_attribution.py --holdout-every 8.

**B6.** Smoothness (Table 4, Figure S1): record the rollouts of every method with --record-dir and run scripts/ar3_smoothness.py with one --run per method. Until then “smoother than the baselines” is an observation and is not stated as a result.

**B7.** Other tasks (door opening, further household tasks): no data has been provided, so the paper makes no claim about them.

### C. Claims the current evidence does not support

The paper therefore does not say that muscleVLTA outperforms LIFT, ForceVLA2 or GR00T (no counts yet), that it is state of the art, that the policy uses touch at test time (Table 2 shows that it does not), that seven joint torques make the motion smoother (not measured), or that balanced fusion improves real-robot success (not tested). Each statement can be added once the corresponding item of part B is measured.

### D. Submission

CVPR 2027 dates (cvpr.thecvf.com, read on 10 November 2026): paper registration 10 November, submission 16 November, supplementary material 23 November 2026, all Anywhere on Earth. CVPR requires the official LaTeX template; this Word file is a draft in a similar layout, in which the main text ends on page 8. Recheck the length in the template once the placeholders are filled. Figures are available as PDF for LaTeX (the .pdf files in picture/) and as SVG and PNG. The review is anonymous (remove the author line), and the LLM usage statement must name the tools actually used.

### E. Regenerating the document

Text: edit the .md files in picture/src/manuscript and references.json, then run scripts/build_manuscript.py (--png-only for Word versions that do not display SVG). Result charts: edit picture/src/results.json, then run scripts/build_result_figures.py. Smoothness figure: scripts/ar3_smoothness.py --out picture/results_smoothness. Architecture figures: edit the .svg sources in picture/src, then run scripts/build_figures.py.
