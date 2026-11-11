"""Offline reinforcement learning for the AR3 muscleVLTA policies: advantage-conditioned fine-tuning with a
force-touch critic, in rounds (the RECAP recipe of pi*0.6, see README "Offline RL").

One round, all on the training server (``scripts/train_ar3_muscle_vlta_rl.sh``):

1. ``outcomes``: every episode is labelled "success" or "failure" (``scripts/ar3_rl.py label``); the value of a
   state is minus the normalized number of steps left until the task is complete, -1 for failures.
2. ``critic``: a distributional value function reads the cached vision features, the TCP pose and the sensor history
   of the policy (joint torques + TCP wrench, Tac G2 fingers). It is cross-fitted: every episode is scored by a
   critic that was trained without it.
3. ``advantage``: n-step advantages of every action chunk, labelled positive / negative per task and data source;
   human corrections are always positive.
4. The policy is fine-tuned on all data with the label of every chunk in its prompt (``pi05_ar3_muscle_vlta_rl``):
   negative chunks get "Advantage: negative", the others keep the prompt, with which the policy is served. The
   ``*_rl_recap`` presets write both indicators, as RECAP does, and are served with "Advantage: positive".

Nothing here changes the model or the losses of the policy; configs without ``advantage_tag`` are unaffected.
"""
