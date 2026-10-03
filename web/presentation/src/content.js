const DATA = [
  {
    n: 1,
    speaker: "Noah",
    budget: 18.933333333333334,
    section: "CSI 5341 · Paper presentation",
    title: "VLA-JEPA",
    purpose: "Introduce the central research question and the speakers.",
    chunks: [
      {
        text: "Hi everyone, we're Noah and Fernando. We're presenting VLA-JEPA by Sun and colleagues. The paper asks whether human video can improve a robot policy without providing robot action labels. Its method combines future-feature prediction with robot action supervision, and its strongest reported improvement is under visual changes.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Sun et al., VLA-JEPA, arXiv:2602.10098v1 (2026), abstract and introduction.",
    wordCount: 47,
    pauseSeconds: 0.35,
    start: 0,
    end: 18.933333333333334,
    measuredSyntheticWithPauses: 18.933333333333334,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "Hi everyone, we're Noah and Fernando. We're presenting VLA-JEPA by Sun and colleagues. The paper asks whether human video can improve a robot policy without providing robot action labels. Its method combines future-feature prediction with robot action supervision, and its strongest reported improvement is under visual changes.",
    backup: false,
    id: "chapter-1",
    audioAsset: "narration-01.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 21.2,
    displayNumber: 1,
  },
  {
    hidden: true,
    id: "reserved-retired-intro",
  },
  {
    n: 2,
    speaker: "Noah",
    budget: 32.53333333333333,
    section: "Problem and insight",
    title: "What can video teach a robot?",
    purpose:
      "Make the task and representation problem concrete before introducing components.",
    chunks: [
      {
        text: "Look at these two clips. A person puts a jar in a box, and a robot puts a block in a bowl. We can see what happened in both cases. But the robot recording also tells us which commands made it happen. The human video doesn't.",
        pause: 0.35,
        focus: "Explain what supervision the two recordings provide.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "There are plenty of videos of people handling objects, but they don't tell this robot which commands to execute. The paper uses those videos to learn how a scene changes, while robot demonstrations directly connect that visual knowledge to the specific controls used by the robot being trained.",
        pause: 0.35,
        focus: "Explain what supervision the two recordings provide.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Human footage: Something-Something-v2 validation sample 174198.webm, label putting jar into box, template Putting [something] into [something]. Retrieved from the public morpheushoc/something-something-v2 archive mirror and checked against its validation.json annotation; native 427x240, 12 fps, 5.916 seconds. Cropped to 426x192; output duplicates source frames at 60 fps. Before/during/after stills are from this same clip. This is a verified V2 dataset example, not an identified VLA-JEPA training example. Dataset: https://www.qualcomm.com/developer/software/something-something-v-2-dataset . Mirror: https://huggingface.co/datasets/morpheushoc/something-something-v2 . Robot footage: real DROID episode AUTOLab+0d4edc83+2023-10-27-19h-52m-50s, exterior camera 24400334. Cropped; approximate timing restored from the 5.901-second recorded control-timestamp span, with an end hold. Paired trajectory.h5 contains recorded controls. Both videos play once silently, then hold. This is dataset footage, not a VLA-JEPA policy rollout. https://droid-dataset.github.io/visualizer/",
    wordCount: 94,
    pauseSeconds: 0.7,
    start: 18.933333333333334,
    end: 51.46666666666667,
    measuredSyntheticWithPauses: 32.53333333333333,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "Look at these two clips. A person puts a jar in a box, and a robot puts a block in a bowl. We can see what happened in both cases. But the robot recording also tells us which commands made it happen. The human video doesn't.\n\nThere are plenty of videos of people handling objects, but they don't tell this robot which commands to execute. The paper uses those videos to learn how a scene changes, while robot demonstrations directly connect that visual knowledge to the specific controls used by the robot being trained.",
    id: "chapter-2",
    audioAsset: "narration-03.mp3",
    scriptRevision: "related-work-bridge-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 42.5,
    displayNumber: 2,
  },
  {
    n: 3,
    speaker: "Noah",
    budget: 76.16666666666667,
    section: "Related work",
    title: "Feature prediction is shared; the pipeline differs",
    purpose:
      "Explain the contribution relative to robot-supervised VLAs, latent-action learning, and JEPA.",
    chunks: [
      {
        text: "LAPA and UniVLA also try to learn from unlabelled human video. Their approach is to infer latent actions from frame pairs, then train a policy to predict them and adapt that policy to recorded robot controls.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "LAPA learns latent actions from frame pairs by reconstructing the future image, then adapts the policy to robot controls. UniVLA follows a similar idea using DINO features instead of pixels.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "VLA-JEPA also predicts features, so that alone isn't its contribution. The difference is where the prediction loss enters: tokens produced from the current observation and instruction predict future features, training the policy's representation alongside robot action supervision.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The next diagram shows this connection: one representation supports both future-state prediction and control. Fernando will walk through how those two branches are trained.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "LAPA (2024), §3; UniVLA (RSS 2025), §III; VLA-JEPA (2026), §3. Different targets and pipelines; feature prediction is not unique to VLA-JEPA.",
    wordCount: 127,
    pauseSeconds: 1.4,
    start: 51.46666666666667,
    end: 127.63333333333334,
    measuredSyntheticWithPauses: 76.16666666666667,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
      {
        time: 0,
        instruction: "After this chapter, hand over to Fernando.",
        handoff: true,
      },
    ],
    notes:
      "LAPA and UniVLA also try to learn from unlabelled human video. Their approach is to infer latent actions from frame pairs, then train a policy to predict them and adapt that policy to recorded robot controls.\n\nLAPA learns latent actions from frame pairs by reconstructing the future image, then adapts the policy to robot controls. UniVLA follows a similar idea using DINO features instead of pixels.\n\nVLA-JEPA also predicts features, so that alone isn't its contribution. The difference is where the prediction loss enters: tokens produced from the current observation and instruction predict future features, training the policy's representation alongside robot action supervision.\n\nThe next diagram shows this connection: one representation supports both future-state prediction and control. Fernando will walk through how those two branches are trained.",
    backup: false,
    id: "chapter-3",
    audioAsset: "narration-04.mp3",
    scriptRevision: "direct-spoken-style-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 57.8,
    displayNumber: 3,
  },
  {
    n: 4,
    speaker: "Fernando",
    budget: 46.9,
    section: "Method 1 of 5 · Human video supervision",
    title: "Predict a future state in feature space",
    purpose:
      "Explain the JEPA target/predictor distinction using the original paper figure.",
    chunks: [
      {
        text: "Take the jar-and-box example. The useful prediction is the resulting state, rather than every detail of the future image.",
        pause: 0.35,
        focus: "Explain one idea at a time.",
        voice:
          "Speak to classmates in a natural voice. Use short breaths between ideas; do not read names or numbers with an announcer cadence.",
      },
      {
        text: "VLA-JEPA predicts the next state's embedding. Its latent action tokens condition that prediction, alongside encoded state history. We compare the output with features extracted from the actual future video by a frozen encoder.",
        pause: 0.35,
        focus: "Explain one idea at a time.",
        voice:
          "Speak to classmates in a natural voice. Use short breaths between ideas; do not read names or numbers with an announcer cadence.",
      },
      {
        text: "So the tokens represent the transition, while the predicted embedding represents the future state. Neither is a robot command. The action head handles that separately.",
        pause: 0.35,
        focus: "Explain one idea at a time.",
        voice:
          "Speak to classmates in a natural voice. Use short breaths between ideas; do not read names or numbers with an announcer cadence.",
      },
    ],
    source:
      "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    wordCount: 77,
    pauseSeconds: 1.05,
    start: 127.63333333333334,
    end: 174.53333333333333,
    measuredSyntheticWithPauses: 46.9,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Speak naturally; take a short breath between paragraphs. Use the slide to point out the example rather than reading every label.",
      },
      {
        time: 0,
        instruction:
          "After this chapter, hand over to Noah. This is a delivery note, not spoken text.",
        handoff: true,
      },
    ],
    notes:
      "Take the jar-and-box example. The useful prediction is the resulting state, rather than every detail of the future image.\n\nVLA-JEPA predicts the next state's embedding. Its latent action tokens condition that prediction, alongside encoded state history. We compare the output with features extracted from the actual future video by a frozen encoder.\n\nSo the tokens represent the transition, while the predicted embedding represents the future state. Neither is a robot command. The action head handles that separately.",
    backup: false,
    id: "chapter-4",
    audioAsset: "narration-05.mp3",
    scriptRevision: "conversational-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 35.3,
    hidden: true,
    retiredReason:
      "Redundant future-state overview; explanation merged into predictor chapter.",
  },
  {
    n: 5,
    speaker: "Fernando",
    budget: 22.666666666666668,
    section: "Method 2 of 5 · Architecture",
    title: "First, encode the observation and instruction",
    purpose:
      "Introduce the first architecture path before revealing the predictor.",
    chunks: [
      {
        text: "First, Qwen, the pretrained vision-language backbone, processes the current image and instruction. It produces the latent action tokens.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "These tokens condition both the world predictor and the action head. Their meaning is learned through those objectives; they aren't discrete motor commands that we execute directly.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    wordCount: 45,
    pauseSeconds: 0.7,
    start: 174.53333333333333,
    end: 197.2,
    measuredSyntheticWithPauses: 22.666666666666668,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "First, Qwen, the pretrained vision-language backbone, processes the current image and instruction. It produces the latent action tokens.\n\nThese tokens condition both the world predictor and the action head. Their meaning is learned through those objectives; they aren't discrete motor commands that we execute directly.",
    backup: false,
    id: "chapter-5",
    audioAsset: "narration-06.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 20.7,
    displayNumber: 4,
  },
  {
    n: 6,
    speaker: "Fernando",
    budget: 23.416666666666668,
    section: "Method 2 of 5 · Architecture",
    title: "Predict the next state’s embedding",
    purpose:
      "Define state embeddings and latent actions, then give implementation detail without a dense architecture dump.",
    chunks: [
      {
        text: "The world predictor combines the policy's action tokens with encoded state history and predicts the next state's embedding. For the jar-and-box example, that's a representation of the resulting scene, rather than a reconstruction of every pixel.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The action tokens and the predicted state are different objects. The tokens describe the transition requested by the instruction. The predictor uses those tokens to estimate its visual outcome. There's no image decoder in this branch, and the output isn't a set of robot commands.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The future frames are withheld from this prediction path. They are encoded separately to build the training target, so the prediction pathway cannot read the answer from its input.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    wordCount: 110,
    pauseSeconds: 1.05,
    start: 197.2,
    end: 220.61666666666665,
    measuredSyntheticWithPauses: 23.416666666666668,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
      {
        time: 0,
        instruction: "After this chapter, hand over to Noah.",
        handoff: true,
      },
    ],
    notes:
      "The world predictor combines the policy's action tokens with encoded state history and predicts the next state's embedding. For the jar-and-box example, that's a representation of the resulting scene, rather than a reconstruction of every pixel.\n\nThe action tokens and the predicted state are different objects. The tokens describe the transition requested by the instruction. The predictor uses those tokens to estimate its visual outcome. There's no image decoder in this branch, and the output isn't a set of robot commands.\n\nThe future frames are withheld from this prediction path. They are encoded separately to build the training target, so the prediction pathway cannot read the answer from its input.",
    backup: false,
    id: "chapter-6",
    audioAsset: "narration-07.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 49.9,
    displayNumber: 5,
  },
  {
    n: 7,
    speaker: "Noah",
    budget: 24.416666666666668,
    section: "Method 2 of 5 · Architecture",
    title: "Compare the prediction with the observed future",
    purpose:
      "Define state embeddings and latent actions, then give implementation detail without a dense architecture dump.",
    chunks: [
      {
        text: "The frozen V-JEPA 2 video encoder gives us the target features from the actual future frames. The prediction loss measures the mismatch with that target.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "Training updates the prediction pathway, including the vision-language backbone, while the target encoder stays fixed. Human video can therefore train the policy's representation without robot action labels.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The gradient updates the predictor: the target remains fixed, while the trainable prediction path is corrected. The bars illustrate feature mismatch, not a physical trajectory or a literal visualization of the learned feature space.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    wordCount: 86,
    pauseSeconds: 1.05,
    start: 220.61666666666665,
    end: 245.0333333333333,
    measuredSyntheticWithPauses: 24.416666666666668,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "The frozen V-JEPA 2 video encoder gives us the target features from the actual future frames. The prediction loss measures the mismatch with that target.\n\nTraining updates the prediction pathway, including the vision-language backbone, while the target encoder stays fixed. Human video can therefore train the policy's representation without robot action labels.\n\nThe gradient updates the predictor: the target remains fixed, while the trainable prediction path is corrected. The bars illustrate feature mismatch, not a physical trajectory or a literal visualization of the learned feature space.",
    backup: false,
    id: "chapter-7",
    audioAsset: "narration-08.mp3",
    scriptRevision: "direct-spoken-style-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 39.3,
    displayNumber: 6,
  },
  {
    n: 8,
    speaker: "Noah",
    budget: 44.96666666666667,
    section: "Method 3 of 5 · Information flow",
    title: "The predictor sees history, while future states are targets",
    purpose:
      "Resolve the subtle difference between target supervision and teacher-forced world-model history.",
    chunks: [
      {
        text: "Future frames appear only on the target side during training. The policy sees the current image and the instruction. The predictor also gets information about earlier states. It doesn't get the future frames it's supposed to predict.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "We only use those frames to make the target and check the prediction. That prevents the model from simply reading the answer from its input.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "When the robot is actually running, there's no future video to check against. We use the policy's tokens and an action head to generate commands. Fernando will explain how we train that part.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    wordCount: 95,
    pauseSeconds: 1.05,
    start: 245.0333333333333,
    end: 290.0,
    measuredSyntheticWithPauses: 44.96666666666667,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
      {
        time: 0,
        instruction: "After this chapter, hand over to Fernando.",
        handoff: true,
      },
    ],
    notes:
      "Future frames appear only on the target side during training. The policy sees the current image and the instruction. The predictor also gets information about earlier states. It doesn't get the future frames it's supposed to predict.\n\nWe only use those frames to make the target and check the prediction. That prevents the model from simply reading the answer from its input.\n\nWhen the robot is actually running, there's no future video to check against. We use the policy's tokens and an action head to generate commands. Fernando will explain how we train that part.",
    backup: false,
    id: "chapter-8",
    audioAsset: "narration-09.mp3",
    scriptRevision: "direct-spoken-style-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 43.3,
    displayNumber: 7,
  },
  {
    n: 9,
    speaker: "Fernando",
    budget: 60.15,
    section: "Method 4 of 5 · Action generation",
    title: "Learn to update a whole sequence of robot controls",
    purpose:
      "Explain interpolation, velocity supervision, and inference as separate operations.",
    chunks: [
      {
        text: "Thanks, Noah. The action head uses flow matching to generate a chunk of robot controls.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "Each panel here represents a complete control sequence, rather than a trajectory through physical space. On the right is a demonstrated action chunk. On the left is a Gaussian noise sample with the same dimensions.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "During training, we interpolate between them at a randomly sampled mixing time. The network receives that mixed sample, the mixing time and the policy tokens. It predicts the velocity from noise toward the demonstrated chunk.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The orange arrow is the target velocity; the blue arrow is the prediction. We minimize their squared difference across examples and mixing times.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The mixing time belongs to the generation process. It isn't the time at which the robot executes a command.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "For this straight interpolation, the target velocity is the difference between the demonstrated action chunk and the noise sample. Across many sampled pairs, the network learns a conditioned field, rather than copying the single arrow shown here.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper §3.3, Eqs. 7–8. Mixing time is generation time, not physical robot time. The target velocity is demonstration minus sampled noise; squared-error regression. Own 2D illustration inspired by Jia-Bin Huang and Julia Turc videos; not an empirical trajectory.",
    wordCount: 164,
    pauseSeconds: 2.1,
    start: 290.0,
    end: 350.15,
    measuredSyntheticWithPauses: 60.15,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "Thanks, Noah. The action head uses flow matching to generate a chunk of robot controls.\n\nEach panel here represents a complete control sequence, rather than a trajectory through physical space. On the right is a demonstrated action chunk. On the left is a Gaussian noise sample with the same dimensions.\n\nDuring training, we interpolate between them at a randomly sampled mixing time. The network receives that mixed sample, the mixing time and the policy tokens. It predicts the velocity from noise toward the demonstrated chunk.\n\nThe orange arrow is the target velocity; the blue arrow is the prediction. We minimize their squared difference across examples and mixing times.\n\nThe mixing time belongs to the generation process. It isn't the time at which the robot executes a command.\n\nFor this straight interpolation, the target velocity is the difference between the demonstrated action chunk and the noise sample. Across many sampled pairs, the network learns a conditioned field, rather than copying the single arrow shown here.",
    backup: false,
    id: "chapter-9",
    audioAsset: "narration-10.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 75.0,
    displayNumber: 8,
  },
  {
    n: 10,
    speaker: "Fernando",
    budget: 42.583333333333336,
    section: "Method 4 of 5 · Action generation",
    title: "Generate a control sequence with repeated learned updates",
    purpose:
      "Explain interpolation, velocity supervision, and inference as separate operations.",
    chunks: [
      {
        text: "At inference, we start from fresh noise, with no demonstrated action to mix in. The model predicts a velocity conditioned on the current observation and instruction.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "We take a small integration step, evaluate the velocity again, and repeat. That produces an action chunk for the controller. These four updates are a schematic illustration.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "Training supplies the target velocity through demonstrations. Inference follows the learned field. This action-generation process is separate from predicting future visual embeddings.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The field is evaluated again after each step, so the generated path can change direction. Straight training interpolations don't imply that every inference trajectory is one straight line.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper §3.3 and Appendix A: four denoising/integration steps, 7-dimensional actions, future action horizon 7. Own illustrative trajectory; sampler type is not asserted. Training and inference paths are different concepts.",
    wordCount: 103,
    pauseSeconds: 1.4,
    start: 350.15,
    end: 392.7333333333333,
    measuredSyntheticWithPauses: 42.583333333333336,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
      {
        time: 0,
        instruction: "After this chapter, hand over to Noah.",
        handoff: true,
      },
    ],
    notes:
      "At inference, we start from fresh noise, with no demonstrated action to mix in. The model predicts a velocity conditioned on the current observation and instruction.\n\nWe take a small integration step, evaluate the velocity again, and repeat. That produces an action chunk for the controller. These four updates are a schematic illustration.\n\nTraining supplies the target velocity through demonstrations. Inference follows the learned field. This action-generation process is separate from predicting future visual embeddings.\n\nThe field is evaluated again after each step, so the generated path can change direction. Straight training interpolations don't imply that every inference trajectory is one straight line.",
    backup: false,
    id: "chapter-10",
    audioAsset: "narration-11.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 47.2,
    displayNumber: 9,
  },
  {
    n: 11,
    speaker: "Noah",
    budget: 47.28333333333333,
    section: "Method 5 of 5 · Joint optimization",
    title: "Visual supervision transfers; robot controls need an interface",
    purpose:
      "Explain which loss applies to each data source and what remains frozen.",
    chunks: [
      {
        text: "Human video and robot demonstrations provide complementary supervision. Human videos provide future visual targets. Robot demonstrations provide those targets plus action labels, so they also train the control head.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The paper evaluates several robot setups, with post-training for the target setups. Sharing visual supervision doesn't make the policy independent of the robot: action conventions and controller timing still have to match.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "These clips illustrate the data sources. They do not demonstrate zero-shot transfer between arbitrary robots.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper, equation 9, sections 3.3 and 4.1; SSv2 220K videos, DROID 76K trajectories. Equation 5 is presented as an embedding discrepancy without an explicit norm; this presentation does not invent an L1 or Smooth-L1 choice. Robot embodiments and post-training: Paper §§4.1–4.2 and Appendix B. Controls require compatible frame, units, gripper convention and timing; arbitrary zero-shot embodiment transfer is not established.",
    wordCount: 76,
    pauseSeconds: 1.05,
    start: 392.7333333333333,
    end: 440.01666666666665,
    measuredSyntheticWithPauses: 47.28333333333333,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "Human video and robot demonstrations provide complementary supervision. Human videos provide future visual targets. Robot demonstrations provide those targets plus action labels, so they also train the control head.\n\nThe paper evaluates several robot setups, with post-training for the target setups. Sharing visual supervision doesn't make the policy independent of the robot: action conventions and controller timing still have to match.\n\nThese clips illustrate the data sources. They do not demonstrate zero-shot transfer between arbitrary robots.",
    backup: false,
    id: "chapter-11",
    audioAsset: "narration-12.mp3",
    scriptRevision: "direct-spoken-style-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 34.8,
    displayNumber: 10,
  },
  {
    n: 12,
    speaker: "Noah",
    budget: 13.5,
    section: "Paper results · Sun et al.",
    title: "Standard LIBERO: a near tie",
    purpose:
      "Compare matched benchmark numbers with correct units and a near-tie caveat.",
    chunks: [
      {
        text: "On standard LIBERO, VLA-JEPA reaches 97.2 percent success. OpenVLA-OFT, the robot-policy baseline shown here, reaches 97.1 percent.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "That's effectively a tie. LIBERO-Plus evaluates these policies under changes to the camera, lighting and object layout.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper, Tables 1 and 3. Differences: 97.2−97.1=0.1 percentage points, 79.5−69.6=9.9 percentage points. Selected baseline is OpenVLA-OFT; this is not a claim of superiority to every model on every task. Paper reports 50 episodes per task on standard LIBERO and does not provide confidence intervals for these headline averages.",
    wordCount: 34,
    pauseSeconds: 0.7,
    start: 440.01666666666665,
    end: 453.51666666666665,
    measuredSyntheticWithPauses: 13.5,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "On standard LIBERO, VLA-JEPA reaches 97.2 percent success. OpenVLA-OFT, the robot-policy baseline shown here, reaches 97.1 percent.\n\nThat's effectively a tie. LIBERO-Plus evaluates these policies under changes to the camera, lighting and object layout.",
    backup: false,
    id: "chapter-12",
    audioAsset: "narration-13.mp3",
    scriptRevision: "direct-spoken-style-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 15.8,
    displayNumber: 11,
  },
  {
    n: 13,
    speaker: "Noah",
    budget: 21.45,
    section: "Paper results · Sun et al.",
    title: "LIBERO-Plus: a larger gain under perturbations",
    purpose:
      "Compare matched benchmark numbers with correct units and a near-tie caveat.",
    chunks: [
      {
        text: "LIBERO-Plus changes the camera, lighting and object layout. Here, VLA-JEPA reaches 79.5 percent, versus 69.6 percent for OpenVLA-OFT.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "That's almost ten percentage points. This supports improved robustness to the changes tested here, rather than a general claim that the model has solved manipulation.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper, Tables 1 and 3. Differences: 97.2−97.1=0.1 percentage points, 79.5−69.6=9.9 percentage points. Selected baseline is OpenVLA-OFT; this is not a claim of superiority to every model on every task. Paper reports 50 episodes per task on standard LIBERO and does not provide confidence intervals for these headline averages.",
    wordCount: 43,
    pauseSeconds: 0.7,
    start: 453.51666666666665,
    end: 474.96666666666664,
    measuredSyntheticWithPauses: 21.45,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "LIBERO-Plus changes the camera, lighting and object layout. Here, VLA-JEPA reaches 79.5 percent, versus 69.6 percent for OpenVLA-OFT.\n\nThat's almost ten percentage points. This supports improved robustness to the changes tested here, rather than a general claim that the model has solved manipulation.",
    backup: false,
    id: "chapter-13",
    audioAsset: "narration-14.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 19.8,
    displayNumber: 12,
  },
  {
    n: 14,
    speaker: "Noah",
    budget: 18.75,
    section: "Paper results · Sun et al.",
    title: "Human video improves LIBERO-Plus robustness",
    purpose:
      "Use controlled within-method ablation and a counterexample to temper the main claim.",
    chunks: [
      {
        text: "The authors also remove human video from training. On LIBERO-Plus, success drops from 79.5 to 62.9 percent.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "That's a 16.6-point difference within the same method. This comparison is more direct evidence for the contribution of human-video supervision than the comparison with a different policy.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper, Tables 1–3 and sections 4.4–4.5. Within-method ablation deltas with human video minus without: LIBERO-Plus +16.6 pp; LIBERO +1.1 pp; SimplerEnv Google −13.2 pp. Real-world repeated grasping is an author-reported qualitative observation, with a proposed attribution to human videos, not an isolated causal proof.",
    wordCount: 44,
    pauseSeconds: 0.7,
    start: 474.96666666666664,
    end: 493.71666666666664,
    measuredSyntheticWithPauses: 18.75,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
      {
        time: 0,
        instruction: "After this chapter, hand over to Fernando.",
        handoff: true,
      },
    ],
    notes:
      "The authors also remove human video from training. On LIBERO-Plus, success drops from 79.5 to 62.9 percent.\n\nThat's a 16.6-point difference within the same method. This comparison is more direct evidence for the contribution of human-video supervision than the comparison with a different policy.",
    backup: false,
    id: "chapter-14",
    audioAsset: "narration-15.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 20.3,
    displayNumber: 13,
  },
  {
    n: 15,
    speaker: "Fernando",
    budget: 17.983333333333334,
    section: "Paper results · Sun et al.",
    title: "Human video does not improve every benchmark",
    purpose:
      "Use controlled within-method ablation and a counterexample to temper the main claim.",
    chunks: [
      {
        text: "The benefit isn't consistent across benchmarks. In one task group in SimplerEnv, another simulation benchmark, human-video training lowers success from 78.4 to 65.2 percent.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "So the strongest evidence is for robustness on LIBERO-Plus. We shouldn't assume the same gain on every robot or task.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper, Tables 1–3 and sections 4.4–4.5. Within-method ablation deltas with human video minus without: LIBERO-Plus +16.6 pp; LIBERO +1.1 pp; SimplerEnv Google −13.2 pp. Real-world repeated grasping is an author-reported qualitative observation, with a proposed attribution to human videos, not an isolated causal proof.",
    wordCount: 44,
    pauseSeconds: 0.7,
    start: 493.71666666666664,
    end: 511.7,
    measuredSyntheticWithPauses: 17.983333333333334,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "The benefit isn't consistent across benchmarks. In one task group in SimplerEnv, another simulation benchmark, human-video training lowers success from 78.4 to 65.2 percent.\n\nSo the strongest evidence is for robustness on LIBERO-Plus. We shouldn't assume the same gain on every robot or task.",
    backup: false,
    id: "chapter-15",
    audioAsset: "narration-16.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 20.3,
    displayNumber: 14,
  },
  {
    speaker: "Noah",
    budget: 27.25,
    section: "Our project · Proposal and work so far",
    title: "Our proposal: evaluate efficiency without assuming success",
    purpose:
      "Explain the planned comparison and current preparation without announcing results.",
    chunks: [
      {
        text: "Our project asks whether we can reduce the cost of running this model while preserving task performance.",
        pause: 0.35,
        focus: "Explain one idea at a time.",
        voice:
          "Speak to classmates in a natural voice. Use short breaths between ideas; do not read names or numbers with an announcer cadence.",
      },
      {
        text: "We'll compare the original pretrained model, a quantized inference variant, and a version with SmolVLM as the smaller vision-language backbone. We'll measure task success, latency and peak memory under consistent conditions.",
        pause: 0.35,
        focus: "Explain one idea at a time.",
        voice:
          "Speak to classmates in a natural voice. Use short breaths between ideas; do not read names or numbers with an announcer cadence.",
      },
      {
        text: "We have the code, pretrained models and personal compute for inference and limited experiments. Full training from scratch is outside our scope. If time permits, Dream-RSI will help us search for further architecture changes.",
        pause: 0.35,
        focus: "Explain one idea at a time.",
        voice:
          "Speak to classmates in a natural voice. Use short breaths between ideas; do not read names or numbers with an announcer cadence.",
      },
      {
        text: "That's the proposal; we're not presenting project performance results yet.",
        pause: 0.35,
        focus: "Explain one idea at a time.",
        voice:
          "Speak to classmates in a natural voice. Use short breaths between ideas; do not read names or numbers with an announcer cadence.",
      },
    ],
    source:
      "Presenters’ project proposal and work in progress. No project performance results, speedups, success retention or memory reductions are announced. Planned comparisons require matched evaluation; outcomes remain open.",
    n: 16,
    start: 511.7,
    end: 538.95,
    wordCount: 92,
    pauseSeconds: 1.4,
    measuredSyntheticWithPauses: 27.25,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Speak naturally; take a short breath between paragraphs. Use the slide to point out the example rather than reading every label.",
      },
      {
        time: 0,
        instruction:
          "After this chapter, hand over to Fernando. This is a delivery note, not spoken text.",
        handoff: true,
      },
    ],
    notes:
      "Our project asks whether we can reduce the cost of running this model while preserving task performance.\n\nWe'll compare the original pretrained model, a quantized inference variant, and a version with SmolVLM as the smaller vision-language backbone. We'll measure task success, latency and peak memory under consistent conditions.\n\nWe have the code, pretrained models and personal compute for inference and limited experiments. Full training from scratch is outside our scope. If time permits, Dream-RSI will help us search for further architecture changes.\n\nThat's the proposal; we're not presenting project performance results yet.",
    backup: false,
    id: "chapter-16",
    audioAsset: "narration-17.mp3",
    scriptRevision: "conversational-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 42.3,
    hidden: true,
    retiredReason: "Project proposal removed from assessed paper presentation.",
  },
  {
    n: 17,
    speaker: "Fernando",
    budget: 23.966666666666665,
    section: "Critical analysis",
    title: "Deployment gaps remain beyond headline averages",
    purpose:
      "Identify benchmark, statistical, semantic and hardware limits without inventing experiments.",
    chunks: [
      {
        text: "The paper has limits. Performance is weaker in the sensor-noise comparison, and the real-robot evaluation has only ten trials per task, including some wrong-object selections.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "Training uses eight A100 GPUs. The reported success rates therefore need to be considered alongside the training resources, as well as the limitations of the evaluation.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "With ten trials per task, a single additional failure changes the measured success rate by ten percentage points. That makes the real-world evidence much less precise than the headline percentages suggest. The reported averages also lack confidence intervals, so small differences should be interpreted cautiously.",
        pause: 0.35,
        focus: "Explain the paper-specific distinction.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Sun et al., Table 3, §4.1, §4.4 and Appendix B. Ten trials per task describes the authors’ real-world study; eight A100 GPUs describes their training. Our single-GPU work is a proposed efficiency comparison, with no performance claims.",
    wordCount: 96,
    pauseSeconds: 1.05,
    start: 538.95,
    end: 562.9166666666667,
    measuredSyntheticWithPauses: 23.966666666666665,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "The paper has limits. Performance is weaker in the sensor-noise comparison, and the real-robot evaluation has only ten trials per task, including some wrong-object selections.\n\nTraining uses eight A100 GPUs. The reported success rates therefore need to be considered alongside the training resources, as well as the limitations of the evaluation.\n\nWith ten trials per task, a single additional failure changes the measured success rate by ten percentage points. That makes the real-world evidence much less precise than the headline percentages suggest. The reported averages also lack confidence intervals, so small differences should be interpreted cautiously.",
    backup: false,
    id: "chapter-17",
    audioAsset: "narration-18.mp3",
    scriptRevision: "paper-focus-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 43.7,
    displayNumber: 15,
  },
  {
    n: 18,
    speaker: "Fernando",
    budget: 17.1,
    section: "Takeaway and discussion",
    title: "What VLA-JEPA contributes",
    purpose:
      "Summarize Sun et al.'s contribution and the limits of the paper's evidence.",
    chunks: [
      {
        text: "Sun and colleagues' contribution is to train a robot policy with future-feature prediction alongside action supervision. Human video supplies the predictive objective, while robot demonstrations also teach the action head to generate controls.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The strongest evidence is the LIBERO-Plus result and human-video ablation. The weaker results on other tasks limit how broadly we can interpret that improvement.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
      {
        text: "The paper supports a way to incorporate human manipulation video into robot-policy training. It doesn't establish reliable performance across arbitrary visual conditions or robot setups. Thank you.",
        pause: 0.35,
        focus: "State the technical relationship directly.",
        voice: "Conversational delivery for graduate computer-vision students.",
      },
    ],
    source:
      "Paper, sections 3 and 4.5. Suggested discussion experiment: controlled appearance changes versus changed dynamics, under matched data and compute. This is our proposed test, not a result in the paper.",
    wordCount: 84,
    pauseSeconds: 1.05,
    start: 562.9166666666667,
    end: 580.0166666666668,
    measuredSyntheticWithPauses: 17.1,
    deliveryCues: [
      {
        time: 0,
        instruction:
          "Explain the paper, not our project. Use a short breath between paragraphs; point to the relevant visual.",
      },
    ],
    notes:
      "Sun and colleagues' contribution is to train a robot policy with future-feature prediction alongside action supervision. Human video supplies the predictive objective, while robot demonstrations also teach the action head to generate controls.\n\nThe strongest evidence is the LIBERO-Plus result and human-video ablation. The weaker results on other tasks limit how broadly we can interpret that improvement.\n\nThe paper supports a way to incorporate human manipulation video into robot-policy training. It doesn't establish reliable performance across arbitrary visual conditions or robot setups. Thank you.",
    backup: false,
    id: "chapter-18",
    audioAsset: "narration-19.mp3",
    scriptRevision: "direct-spoken-style-20261002",
    narrationMatchesScript: false,
    estimatedSpeakingSeconds: 38.4,
    displayNumber: 16,
  },
];
const ACTIVE_CHAPTERS = DATA.map((s, i) => i).filter((i) => !DATA[i].hidden);
function chapterNumber(scene) {
  return ACTIVE_CHAPTERS.indexOf(scene) + 1;
}
