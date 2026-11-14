!pagebreak

!columns 1

## Author checklist — remove before submission

This page is for the authors and is not part of the paper. It lists where every number comes from and what is still missing. Yellow text in the paper marks the open items; the codes B1–B13 refer to part B below.

### A. Numbers in the paper and their sources

**Reported by the authors (real robot).** Table 1, insertion columns, and the policy rows of Table 4 are the values entered in muscleVLTA_manuscript_v2_填写数据v1.docx on 14 November 2026. Three cells were entered as percentages and converted with 60 trials: LIFT near 6.67% = 4/60, muscleVLTA w/o touch near 5.0% = 3/60, balanced fusion far 21.67% = 13/60. ForceVLA2 near was entered as 60.5%, which 60 trials cannot give (36/60 = 60.0%, 37/60 = 61.7%); it is printed as entered, without count and interval. Please confirm two coincidences: balanced and shared fusion both have 13/60 from the far start, and the model without touch has 3/60 from both starts. The Wilson intervals and Fisher tests of Section 4.2 are computed from the counts by scripts/build_result_figures.py; edit picture/src/results.json and re-run it when counts change. Shared fusion is config pi05_ar3_muscle_vlta (run muscle_vlta_30k_plugin_30s35s), balanced fusion pi05_ar3_muscle_vlta_v2. The smoothness values of the policies were not recomputed here: no rollout recordings are on the GPU server.

**Measured offline (GPU server).** Tables 2 and S2: scripts/ar3_sensor_attribution.py on checkpoint 141000 of the shared-fusion run. Table S3: four 5k-step pilots on the two insertion sets with OPENPI_AR3_HOLDOUT_EVERY=8. Table 3: critic error from /data/jiang/train_checkpoints/ar3_rl_critic/demos_full/report.json (4 folds, 84 unplugging and 129 insertion episodes, no failed episode); label agreement, positive control (108 held-out chunks, 1000 steps) and demonstrations-only fine-tuning (216 held-out chunks, 3000 steps, batch 64) from the comparison runs of 11 November, whose checkpoints were removed afterwards. The 0.72 mm error before fine-tuning belongs to a policy that had seen the held-out episodes in training. Section 4.6, latency: scripts/ar3_lift_replay_check.py. Table 4, demonstration rows, and Figure S3: scripts/ar3_smoothness.py. Data: 65 + 65 insertion, 84 unplugging and 83 shake-to-identify episodes (two faulty episodes of the latter were deleted on 13 November), 107,540 frames.

**Literature.** 58 references. The 50 earlier ones were checked against arXiv, publisher or Crossref records on 10 November 2026. Added on 14 November: π*0.6, TORL-VLA, ForceRFT and TacCoRL, whose titles and authors were checked against the arXiv API; what Section 2 says about the last three rests on their abstracts. The entries of Raibert and Craig, Mason, DAgger, and Haddadin et al. were written from memory: check volume and pages. The remark that a joint torque under closed-loop control is a result of the motion and not a command (Sections 1 and 3.2) is distilled from the article on joint modules in velocity and torque mode that the authors supplied (Zane Zhang, 27 October 2026); it is textbook servo control and is not cited.

### B. Still missing (yellow in the paper)

**B1.** Results for unplugging, shake-to-identify and cap opening for every method (Table 1), the success count of ForceVLA2 from the near start, and for every baseline its inputs on this robot, its training budget and, for GR00T, the version. State which model and checkpoint produced the results of the three new tasks. Report Fisher's exact test wherever a difference is claimed.

**B2.** Evaluation protocol of every task: success criterion, time limit, variation of start and object poses, and the evaluated checkpoint step.

**B4.** Test-time touch ablation: run the evaluated checkpoint with both fingers masked (add --mask-tactile to scripts/run_ar3_lift_robot.sh; same protocol, 60 trials from the far start). This decides whether the paper may say that touch is needed in training only (Section 4.3), and it is the first experiment a reviewer will ask for. A masked result between 5/60 and 23/60 does not differ significantly from 13/60; 4/60 or fewer shows that the policy uses touch at test time, and Sections 4.3 and 5 must then be rewritten.

**B5.** Balanced fusion: name the evaluated checkpoint and its training budget. A test under a changed payload or a sensor offset would show what the lower sensitivity to the force level is good for; without it the paper only says that balanced fusion matches shared fusion.

**B6.** Smoothness: state the number of rollouts per method, keep the recordings, and regenerate Figure S3 with scripts/ar3_smoothness.py (one --run per method).

**B8.** Built-in wrench: state how the controller obtains it (built-in sensor or estimate from the joint torques) and its rated accuracy, and confirm that its units are N and Nm.

**B9.** Shake-to-identify. The recorded instruction is “Pick up a bottle and shake it, using force feedback to sense pills moving inside. If it contains medicine, place it on top of the drawer; otherwise, put it back and try another.”; use the identical string at inference. For the result to show what force and touch add, the bottles must look identical and their content must be random and balanced, so that a policy without force and touch is at chance. The policy has no memory beyond its 1 s sensor window: the decision must follow the shake without a pause, and an empty bottle must be put back where the cameras can tell it from the untried ones.

**B10.** muscleVLTA-RL on the robot. No RL-fine-tuned checkpoint exists yet. Record rollouts of the evaluated policy with their outcomes, failures included, and corrections where possible (scripts/run_ar3_lift_robot.sh --record-dir; the recording of the fingers during rollouts is untested on the hardware), run one round with scripts/train_ar3_muscle_vlta_rl.sh, and evaluate the result with the protocol of Table 1. Until then the last row of Table 1 and the RL sentences marked in the abstract and the conclusion stay open, and the paper claims the method and its offline checks only.

**B11.** Cap opening: no dataset is on the GPU server. Describe the task and its demonstrations in Section 4.1 or remove the column.

**B12.** LLM usage statement. Your edit reduced it to “a small amount of model network optimization”. The model and training code of this repository were written with Copilot, so that wording would understate the use. The statement in the paper keeps the actual use and states the authors' part first; adjust the wording as you see fit, but keep it accurate, and name the exact tools and model versions as CVPR 2027 requires (this revision: GitHub Copilot CLI with Claude Fable 5.2; add any other model used for earlier drafts).

**B13.** Changes of the arm. The paper says that joint torques let the robot sense loads on its own body and that the load token is designed to tolerate them; it does not say that the policy is unaffected by a pulled cable or a lost part, because this has not been measured, and Table 2 points the other way for shared fusion: its actions follow the level of the arm force, so an unseen load would be read as a different posture. The experiment: repeat the near-start protocol with a weight fixed to the forearm and with the cable pulled, for shared and balanced fusion, and report the counts beside Table 1. Only then may the abstract claim robustness.

### C. Claims the current evidence does not support

The paper does not say that muscleVLTA is significantly better than ForceVLA2 from the far start (13/60 against 5/60, *p* = 0.071), that the policy uses touch at test time (Table 2 shows that it does not), that balanced fusion improves real-robot success (it matches shared fusion), that RL improves success (not measured), that this is the first RL fine-tuning of a VLA with force or touch (TORL-VLA, ForceRFT and TacCoRL precede it), that the joint torques are what makes the motion smooth (not isolated), or that the method is state of the art.

### D. Submission

CVPR 2027 dates (cvpr.thecvf.com, read on 10 November 2026): submission 16 November, supplementary material 23 November 2026, Anywhere on Earth. CVPR requires the official LaTeX template and at most eight pages of main text without the references; this Word file is a draft in a similar layout, in which the main text and the LLM usage statement end in the right column of page 8 (LibreOffice rendering of 14 November; Word may break pages a few lines differently). Algorithms, the shared-fusion network, the fusion pilots and the discussion of world action models were moved to the supplementary material to make room for muscleVLTA-RL and the new tasks. Recheck the length in the template once the placeholders are filled. Figures are available as PDF for LaTeX (the .pdf files in picture/) and as SVG and PNG. The review is anonymous (remove the author line).

### E. Regenerating the document

Text: edit the .md files in picture/src/manuscript and references.json, then run scripts/build_manuscript.py (--png-only for Word versions that do not display SVG). Tables are three-line tables; a row “---” in a table draws a rule. Result charts: edit picture/src/results.json, then run scripts/build_result_figures.py. Smoothness figure: scripts/ar3_smoothness.py --out picture/results_smoothness. Architecture figures (musclevlta_v2_architecture, musclevlta_rl, musclevlta_architecture): edit the .svg sources in picture/src, then run scripts/build_figures.py.
