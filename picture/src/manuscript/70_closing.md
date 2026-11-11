## 6. Limitations

*Reliability and scope.* From the far start muscleVLTA succeeds in about one trial out of five, so the task is not solved, and all real-robot results so far concern one task on one robot. {{+ door opening and other tasks once measured with counts (B7)}} *Touch at test time.* The shared-fusion policy does not read its tactile input at test time (Section 4.3). Whether the fingers are needed only in training, as the attribution suggests, must still be confirmed on the robot, and balanced fusion has been evaluated offline with short runs only. *Sensing.* We have not compared the built-in arm readings with a dedicated wrist force–torque sensor. *Rates.* The policy acts at 10 Hz and reads a sensor row 0.3 s after it is measured. Faster events are absorbed by the impedance controller, and tasks that need reflexes below this latency are out of reach of the present design.

## 7. Conclusion

muscleVLTA treats force as the quantity that links a robot's commands to what its fingertips feel. A reactive expert reads a joint memory of joint torques, end-effector wrench and gated visuotactile fingers under an explicit latency, and a forward model trained to predict the next force and touch readings shapes that memory at no inference cost. On occluded charger insertion the resulting policy succeeds where π_{0.5} fails. Measuring what the policy reads, rather than assuming it, revealed that the arm force level acts as a posture cue and that touch contributes through training; balanced fusion addresses the first and makes the second testable. We release the model and the attribution and smoothness tools so that both findings can be checked on other robots. {{Confirm the release.}}

## LLM Usage Statement

GitHub Copilot was used to implement the model, the training and inference code and the analysis tools, to run the offline analyses of Sections 4.3–4.5, to draw the figures, and to draft and edit this manuscript. An earlier draft was prepared with an OpenAI model. The real-robot experiments were conducted by the authors, who verified the implementation, the citations and the claims and take responsibility for the content. {{Complete with the exact tools and model versions, as CVPR 2027 requires (this revision: GitHub Copilot CLI with Claude Fable 5.2; earlier draft: per your records).}}

## References

!references
