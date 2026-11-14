## 5. Limitations

*Reliability and scope.* From the far start muscleVLTA succeeds in about one trial out of five, and its margin over the strongest baseline is not significant there; the measured results concern one task on one robot. {{Update with the other tasks (B1).}} *Changes of the arm.* The load token is meant to let the policy keep working under a changed payload or a dragging cable; this has not been tested. *Reinforcement learning.* muscleVLTA-RL has been verified offline; its effect on real-robot success is not measured yet, and with demonstrations alone it changes nothing. *Touch at test time.* The shared-fusion policy does not read its tactile input at test time (Section 4.3); whether the fingers are needed only in training must still be confirmed on the robot. *Rates.* The policy acts at 10 Hz and reads a sensor row 0.3 s after it is measured.

## 6. Conclusion

muscleVLTA treats force as the quantity that links a robot's commands to what its arm and its fingertips feel. A reactive expert reads a joint memory of joint torques, end-effector wrench and gated visuotactile fingers under an explicit latency, and a forward model trained to predict the next force and touch readings shapes that memory at no inference cost. On occluded charger insertion the resulting policy succeeds in 55 of 60 trials from a near start, where vision-only policies fail. Measuring what the policy reads revealed that the arm force level acts as a posture cue and that touch contributes through training. Balanced fusion separates the sustained load of the arm from the interaction force at equal success, and muscleVLTA-RL lets the same force and touch history judge the rollouts of the policy. {{RL result (B10); confirm the release of code and tools.}}

## LLM Usage Statement

The research questions, the sensor and task design, the data collection and all real-robot experiments are the work of the authors. GitHub Copilot was used, under the authors' direction, to implement the model, the training and inference code and the analysis tools on top of the LIFT code base, to run the offline analyses of Sections 4.3–4.6, to draw the figures, and to draft and edit this manuscript. The authors verified the implementation, the citations and the claims and take responsibility for the content. {{See B12.}}

## References

!references
