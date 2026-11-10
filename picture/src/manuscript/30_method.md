## 3. Method

### 3.1. Problem setting and backbone

A 7-DoF arm under Cartesian impedance control [@hogan] carries a two-finger visuotactile gripper. Every 0.1 s the system provides a front and a wrist RGB image, a language instruction, the *arm reading* *x*_{t} = [τ_{t}; *F*_{t}] ∈ ℝ^{13} made of the seven measured joint torques τ_{t} and the six-dimensional TCP wrench *F*_{t}, and the *hand reading*: for each finger *f* ∈ {1, 2} an indentation depth map *D*^{f}_{t} (pooled to 48 × 64), a six-dimensional fingertip wrench *w*^{f}_{t}, and the gripper aperture *g*_{t}. The policy outputs chunks of *H* = 10 actions (1 s), each a TCP pose increment relative to the pose at the chunk start plus an absolute gripper target. As in the implementation of LIFT [@lift], the proprioceptive state is zeroed, so the policy has no direct access to the absolute pose.

We build on π_{0.5} [@pi05]: a PaliGemma vision-language model (SigLIP image encoder, Gemma-2B) [@paligemma; @siglip] encodes the two current images and the instruction into a prefix, and a 300M-parameter action expert generates the chunk by flow matching [@flowmatching; @pi0]. Following LIFT, a second expert of the same size, the *reactive expert*, is initialised as a copy of the action expert and shares self-attention with it: reactive token *i* attends to the prefix, to the nominal action tokens after *i* (the plan ahead), and to reactive tokens up to *i*. In every one of its 18 layers the reactive expert additionally cross-attends to a memory of sensor tokens; the output projection of this cross-attention starts at zero, so training starts from the behaviour of the pre-trained policy. Figure 1 gives the complete pipeline.

!figure wide musclevlta_architecture
caption: Figure 1. muscleVLTA with shared fusion. (a) A slow semantic stream encodes the current images and the instruction once per action chunk; a fast stream feeds joint torques, TCP wrench and the two visuotactile fingers into a joint force memory that the reactive expert reads through latency-causal cross-attention. (b) Contact-gated hand encoder. (c) Visibility of the sensor rows for each action of the chunk. (d) Training objectives; the forward model is used in training only. (e) Real-robot rollout. Circled numbers mark the contributions.

### 3.2. Joint force memory

**Sensor window.** For a chunk starting at step *t* the policy reads *R* = *K* + *H* time-aligned rows per stream, *j* = −*K*, …, *H* − 1, with a history of *K* = 10 steps (1 s). Rows before the start of the episode are flagged invalid; they are skipped by the recurrent encoders and masked in attention, instead of being read as zero force.

**Arm stream.** A GRU reads the arm rows in time order and emits one 1024-dimensional token per row, *m*^{arm}_{j} = GRU_{arm}(*x*_{−K}, …, *x*_{j}). Joint torques and TCP wrench are complementary views of the same load: the wrench summarises what is transmitted through the tool, whereas the torques distribute it over the kinematic chain and also carry the gravity load of the current posture.

**Hand stream with a contact gate.** Finger contact is intermittent: in our demonstrations a finger touches something in 44% of the insertion frames and 26% of the unplugging frames. The indentation maps and fingertip wrenches carry signal only then, and the noise of a free finger is not a weak version of that signal. Each finger reading therefore passes a learned gate (Figure 1b). Three strided convolutions read the map together with its change since the previous row, Δ*D*, where contact onsets, slip and in-hand rotation appear. They are summed with a linear map of eight contact-patch features *g*(*D*) (peak and mean indentation, contact area, centroid and second moments of the indentation above a threshold δ) and an MLP of the fingertip wrench, and normalised to a feature φ. A contact probability *c* then interpolates between this feature and a learned no-contact embedding:

$$ c = σ(MLP([g(D), w])) $$

$$ z = cφ + (1 − c)e_{free} $$

A finger without a valid sample is replaced by a separate embedding *e*_{miss}, which is also what the policy sees for fingers dropped in training (10% for the whole hand and 10% for each finger). A second GRU fuses the two finger tokens, their contact probabilities and the gripper aperture over time into hand tokens *m*^{hand}_{j}.

**Latency-causal reading.** Arm and hand tokens receive a learned stream embedding and share one time index. Action *i* ∈ {0, …, *H* − 1} of the chunk may read row *j* only if the row is valid and was measured at least *L* = 3 steps (0.3 s) before the action is executed:

$$ M_{ij} = 1[ j ≤ i − L ]·v_{j} $$

with *v*_{j} the validity flag. The first *L* actions thus depend on history only, and every later action becomes computable as soon as its last visible row has been measured (Figure 1c). With *shared fusion* the reactive token *r*_{i} is updated in every layer by one attention distribution over all visible memory tokens *m*_{j} of both streams, with query *q*_{i} = *W*_{q}*r*_{i} and keys *k*_{j} = *W*_{k}*m*_{j} (one head shown):

$$ α_{ij} = softmax_{j}(q_{i}^{⊤}k_{j} + log M_{ij}) $$

$$ r_{i} ← r_{i} + W_{o} Σ_{j} α_{ij}W_{v}m_{j} $$

### 3.3. Action-conditioned forward model

The memory should encode how commands change contact, not only what the sensors currently read. During training a small forward model *f*_{ψ} takes the memory tokens of both streams at chunk step *p* together with the commands *a*_{p−1} and *a*_{p}, and predicts the readings of step *p* + 1, for *p* = 1, …, *H* − 2:

$$ ŷ_{p+1} = f_{ψ}([m^{arm}_{p}; m^{hand}_{p}; a_{p−1}; a_{p}]) $$

The target *y*_{p+1} contains the arm reading, the two fingertip wrenches with the contact-patch features of their maps, and the contact state of each finger. The model reads the memory after sensor dropout, so the next reading of a dropped finger has to be inferred from the arm and the other finger. The forward model and its targets exist only in training: prediction costs nothing at inference, and the gradients of predicting touch flow into the arm encoder. We refer to this path, command → arm force → fingertip deformation, as the action–force–touch chain; “causal” is used in the temporal sense that no token can depend on a later measurement.

### 3.4. Balanced fusion

A counterfactual analysis of the trained shared-fusion policy (Section 4.3) shows that its actions follow the *level* of the arm reading and are almost invariant to the hand stream. Without a proprioceptive state, the level of the joint torques, which in free space is dominated by the gravity load of the posture, is a convenient substitute for the pose: a shortcut in the sense of causal confusion [@dehaan]. We therefore separate what the arm reading says about posture from what it says about interaction, and remove the competition between streams (Figure 2).

**Load–interaction decomposition.** Let *x*_{ref} be the oldest valid arm reading of the window, one second before the chunk start once the history is full. It is embedded by an MLP into a single *load token*, visible under the same latency rule as the row it comes from. The arm GRU reads the rows relative to it, *x*_{j} − *x*_{ref}, the *interaction force* accumulated within the window, and the forward model predicts this interaction force. A constant offset of the readings (posture, payload, sensor bias) now reaches the policy through one token only.

**Stream-wise attention.** With a single softmax the keys of all streams compete for the same attention mass. We instead normalise within each stream *s* ∈ {load, interaction, touch}, giving weights α^{s}_{ij} with the softmax of Eq. (4) taken over the rows *j* ∈ *s* only, and sum the results, with shared projections and the same mask:

$$ r_{i} ← r_{i} + W_{o} Σ_{s} Σ_{j∈s} α^{s}_{ij}W_{v}m_{j} $$

A stream none of whose rows is visible contributes zero. As an option, the interaction stream of a training sample is dropped with probability *p*_{arm}, which removes the arm shortcut for that sample and forces the action to be explained by touch and vision.

!figure column musclevlta_balanced_fusion
caption: Figure 2. Balanced fusion. Left: the arm reading of the sensor window is split into the load *x*_{ref}, its first valid reading, and the interaction force *x* − *x*_{ref}. Right: the reactive expert keeps one attention distribution per stream and sums the results. Inputs, losses and the robot client are those of Figure 1; the complete network is shown in the supplementary material.

### 3.5. Training and reactive inference

**Training.** Both experts are trained by conditional flow matching. With a flow time *s* ~ Beta(1.5, 1), ε ~ 𝒩(0, *I*) and *A*^{s} = *s*ε + (1 − *s*)*A*, each expert regresses the velocity ε − *A* from its own noise sample, which gives the losses ℒ_{nom} of the nominal and ℒ_{rea} of the reactive expert. The total loss adds the supervision of the sensor encoders:

$$ ℒ = ℒ_{nom} + ℒ_{rea} + λ_{c}ℒ_{gate} + λ_{f}ℒ_{fwd} $$

ℒ_{gate} is the binary cross-entropy between the contact probabilities and the labels max *D* > δ, ℒ_{fwd} a Huber loss on the predicted forces and patch features plus a cross-entropy on the predicted contact, λ_{c} = 0.05 and λ_{f} = 0.1. Algorithm 1 summarises one training step.

!algorithm Algorithm 1. One training step of muscleVLTA
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

**Reactive inference.** At deployment (Algorithm 2) the vision-language prefix is computed once per chunk and cached, and the noise of both experts is drawn once per chunk, so that repeated integrations of the flow are consistent. After every control step the client appends the newest arm and hand rows and requests the actions that have become computable; because action *i* reads rows up to *i* − *L* only, the answer equals that of a policy that waited for all rows, while up to *L* control periods remain for communication and inference. Pose increments are converted to absolute TCP targets and streamed at 100 Hz to the impedance controller. The sensor history is kept across chunks, and a finger that stops delivering samples is replaced by *e*_{miss}.

!algorithm Algorithm 2. Reactive rollout of one action chunk (10 Hz)
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
