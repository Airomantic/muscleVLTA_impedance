## 3. Method

### 3.1. Problem setting and backbone

A 7-DoF arm under Cartesian impedance control [@hogan] carries a two-finger visuotactile gripper. Every 0.1 s the system provides a front and a wrist RGB image, a language instruction, the *arm reading* *x*_{t} = [τ_{t}; *F*_{t}] ∈ ℝ^{13} made of the seven measured joint torques τ_{t} and the six-dimensional TCP wrench *F*_{t}, and the *hand reading*: for each finger *f* ∈ {1, 2} an indentation depth map *D*^{f}_{t} (pooled to 48 × 64), a six-dimensional fingertip wrench *w*^{f}_{t}, and the gripper aperture *g*_{t}. The policy outputs chunks of *H* = 10 actions (1 s), each a TCP pose increment relative to the pose at the chunk start plus an absolute gripper target. As in the implementation of LIFT [@lift], the proprioceptive state is zeroed, so the policy has no direct access to the absolute pose.

We build on π_{0.5} [@pi05]: a PaliGemma vision-language model (SigLIP image encoder, Gemma-2B) [@paligemma; @siglip] encodes the two current images and the instruction into a prefix, and a 300M-parameter action expert generates the chunk by flow matching [@flowmatching; @pi0]. Following LIFT, a second expert of the same size, the *reactive expert*, is initialised as a copy of the action expert and shares self-attention with it: reactive token *i* attends to the prefix, to the nominal action tokens after *i* (the plan ahead), and to reactive tokens up to *i*. In every one of its 18 layers the reactive expert additionally cross-attends to a memory of sensor tokens; the output projection of this cross-attention starts at zero, so training starts from the behaviour of the pre-trained policy. Figure 1 gives the complete pipeline.

### 3.2. Joint force memory

**Sensor window.** For a chunk starting at step *t* the policy reads *R* = *K* + *H* time-aligned rows per stream, *j* = −*K*, …, *H* − 1, with a history of *K* = 10 steps (1 s). Rows before the start of the episode are flagged invalid; they are skipped by the recurrent encoders and masked in attention, instead of being read as zero force.

**Arm stream.** A GRU reads the arm rows in time order and emits one 1024-dimensional token per row, *m*^{arm}_{j} = GRU_{arm}(*x*_{−K}, …, *x*_{j}). Under closed-loop control a joint torque is a result, not a command: the actuator supplies what the motion requires, namely the inertial, gravity and friction torques of the arm plus every external load. The seven torques thus report loads anywhere along the kinematic chain, where the wrench summarises what is transmitted through the tool.

**Hand stream with a contact gate.** Finger contact is intermittent: in our demonstrations a finger touches something in 44% of the insertion frames and 26% of the unplugging frames. The indentation maps and fingertip wrenches carry signal only then, and the noise of a free finger is not a weak version of that signal. Each finger reading therefore passes a learned gate (Figure 1b). Three strided convolutions read the map together with its change since the previous row, Δ*D*, where contact onsets, slip and in-hand rotation appear. They are summed with a linear map of eight contact-patch features *g*(*D*) of the indentation above a threshold δ and an MLP of the fingertip wrench, and normalised to a feature φ. A contact probability *c* then interpolates between this feature and a learned no-contact embedding:

$$ c = σ(MLP([g(D), w])) $$

$$ z = cφ + (1 − c)e_{free} $$

A finger without a valid sample is replaced by a separate embedding *e*_{miss}, which is also what the policy sees for fingers dropped in training. A second GRU fuses the two finger tokens, their contact probabilities and the gripper aperture over time into hand tokens *m*^{hand}_{j}.

**Latency-causal reading.** Arm and hand tokens receive a learned stream embedding and share one time index. Action *i* ∈ {0, …, *H* − 1} of the chunk may read row *j* only if the row is valid and was measured at least *L* = 3 steps (0.3 s) before the action is executed:

$$ M_{ij} = 1[ j ≤ i − L ]·v_{j} $$

with *v*_{j} the validity flag. The first *L* actions thus depend on history only, and every later action becomes computable as soon as its last visible row has been measured. With *shared fusion* the reactive token *r*_{i} is updated in every layer by one attention distribution over all visible memory tokens *m*_{j} of both streams, with query *q*_{i} = *W*_{q}*r*_{i} and keys *k*_{j} = *W*_{k}*m*_{j} (one head shown):

$$ α_{ij} = softmax_{j}(q_{i}^{⊤}k_{j} + log M_{ij}) $$

$$ r_{i} ← r_{i} + W_{o} Σ_{j} α_{ij}W_{v}m_{j} $$

!figure wide musclevlta_v2_architecture
caption: Figure 1. muscleVLTA. (a) A slow semantic stream encodes the current images and the instruction once per action chunk; a fast stream feeds joint torques, TCP wrench and the two visuotactile fingers into a joint force memory that the reactive expert reads through latency-causal cross-attention. (b) Contact-gated hand encoder. (c) Balanced fusion (Section 3.4). (d) Training objectives; the forward model is used in training only. (e) Real-robot rollout. NEW marks what balanced fusion changes with respect to shared fusion (supplementary material).

### 3.3. Action-conditioned forward model

The memory should encode how commands change contact, not only what the sensors currently read. During training a small forward model *f*_{ψ} takes the memory tokens of both streams at chunk step *p* together with the commands *a*_{p−1} and *a*_{p}, and predicts the readings of step *p* + 1, for *p* = 1, …, *H* − 2:

$$ ŷ_{p+1} = f_{ψ}([m^{arm}_{p}; m^{hand}_{p}; a_{p−1}; a_{p}]) $$

The target *y*_{p+1} contains the arm reading, the two fingertip wrenches with the contact-patch features of their maps, and the contact state of each finger. The forward model and its targets exist only in training: prediction costs nothing at inference, and the gradients of predicting touch flow into the arm encoder. We refer to this path, command → arm force → fingertip deformation, as the action–force–touch chain; “causal” is used in the temporal sense that no token can depend on a later measurement.

### 3.4. Balanced fusion

A counterfactual analysis of the trained shared-fusion policy (Section 4.3) shows that its actions follow the *level* of the arm reading and are almost invariant to the hand stream. Without a proprioceptive state, the level of the joint torques, which in free space is dominated by the gravity load of the posture, is a substitute for the pose: a shortcut in the sense of causal confusion [@dehaan]. We therefore separate what the arm reading says about posture from what it says about interaction, and remove the competition between streams (Figure 1c).

**Load–interaction decomposition.** Let *x*_{ref} be the oldest valid arm reading of the window, one second before the chunk start once the history is full. It is embedded by an MLP into a single *load token*, visible under the same latency rule as the row it comes from. The arm GRU reads the rows relative to it, *x*_{j} − *x*_{ref}, the *interaction force* accumulated within the window, and the forward model predicts this interaction force. What changes slowly (the gravity load of the posture, a payload, a dragging cable, friction, sensor bias) now reaches the policy through one token, and the interaction stream starts from zero in every window: a change of the arm itself shifts the load token and leaves the signal of contact intact. {{Perturbation test (B13).}}

**Stream-wise attention.** With a single softmax the keys of all streams compete for the same attention mass. We instead normalise within each stream *s* ∈ {load, interaction, touch}, giving weights α^{s}_{ij} with the softmax of Eq. (4) taken over the rows *j* ∈ *s* only, and sum the results, with shared projections and the same mask:

$$ r_{i} ← r_{i} + W_{o} Σ_{s} Σ_{j∈s} α^{s}_{ij}W_{v}m_{j} $$

A stream none of whose rows is visible contributes zero.

### 3.5. Training and reactive inference

**Training.** Both experts are trained by conditional flow matching. With a flow time *s* ~ Beta(1.5, 1), ε ~ 𝒩(0, *I*) and *A*^{s} = *s*ε + (1 − *s*)*A*, each expert regresses the velocity ε − *A* from its own noise sample, which gives the losses ℒ_{nom} of the nominal and ℒ_{rea} of the reactive expert. The total loss adds the supervision of the sensor encoders:

$$ ℒ = ℒ_{nom} + ℒ_{rea} + λ_{c}ℒ_{gate} + λ_{f}ℒ_{fwd} $$

ℒ_{gate} is the binary cross-entropy between the contact probabilities and the labels max *D* > δ, ℒ_{fwd} a Huber loss on the predicted forces and patch features plus a cross-entropy on the predicted contact, λ_{c} = 0.05 and λ_{f} = 0.1.

**Reactive inference.** At deployment the vision-language prefix is computed once per chunk and cached, and the noise of both experts is drawn once per chunk, so that repeated integrations of the flow are consistent. After every control step the client appends the newest arm and hand rows and requests the actions that have become computable; because action *i* reads rows up to *i* − *L* only, the answer equals that of a policy that waited for all rows, while up to *L* control periods remain for communication and inference. Pose increments are converted to absolute TCP targets and streamed at 100 Hz to the impedance controller. The supplementary material gives both procedures as algorithms.

### 3.6. muscleVLTA-RL: offline improvement from rollouts

Demonstrations contain successful contacts only, and on-policy RL would need fresh rollouts for every update, which a single real arm cannot supply. We therefore improve the policy offline, in rounds, with the advantage-conditioned recipe of π^{*}_{0.6} [@recap] (Figure 2): the current policy runs on the robot, every episode is stored with its outcome, and the operator may take over, which yields corrections as in DAgger [@dagger].

!figure wide musclevlta_rl
caption: Figure 2. muscleVLTA-RL. (1) Demonstrations, rollouts with their outcomes and human corrections. (2) A distributional critic reads image features, the pose and the force and touch history of the policy. (3) Every recorded chunk is labelled by its *n*-step advantage. (4) The whole policy is fine-tuned with the label in the instruction and deployed with the plain instruction; its rollouts start the next round.

**Force–touch critic.** The reward is −1 per step until the task is complete, so the value of a state is minus the time left to success, normalised by the longest episode of the task; failed episodes sit at −1. A distributional critic *V*_{φ} with 201 bins predicts this value from the mean-pooled SigLIP tokens of both cameras, the TCP pose and gripper aperture, and the one-second sensor history that the policy reads: the arm reading through a GRU and the fingers through the hand encoder of Section 3.2. It is cross-fitted in four folds by episode, so that every frame is scored by a critic that has not seen its episode.

**Advantage-conditioned fine-tuning.** Every recorded chunk receives the *n*-step advantage

$$ A_{t} = Σ_{k=0}^{n−1} r_{t+k} + V_{φ}(s_{t+n}) − V_{φ}(s_{t}) $$

with *n* = 50 steps (5 s); beyond the end of an episode its outcome replaces *V*_{φ}. Per task, the best 50% of the demonstration frames and the best 40% of the policy frames are labelled positive, as are all human corrections. The policy is fine-tuned from its checkpoint on all data with the loss of Eq. (8). Chunks with a negative label are trained with the indicator “Advantage: negative” appended to the instruction, the others with the instruction as it is; at deployment the instruction carries no indicator, which conditions the policy on the positive chunks. The model, its inputs and the reactive client stay unchanged.
