const DATA=[
  {
    "n": 1,
    "speaker": "Noah",
    "budget": 22.957,
    "section": "CSI 5341 · Paper presentation",
    "title": "VLA-JEPA",
    "purpose": "Introduce the central research question and the speakers.",
    "chunks": [
      {
        "text": "Hi everyone. We're Noah and Fernando, and today we're looking at VLA JEPA. The idea starts with a simple question: if a robot watches someone move an object, what should it actually learn from that video? We'll explain how the paper turns that question into a training method, and then look at where the evidence is convincing and where it still falls short.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 21.957
      }
    ],
    "source": "Sun et al., VLA-JEPA, arXiv:2602.10098v1 (2026), abstract and introduction.",
    "wordCount": 63,
    "pauseSeconds": 1.0,
    "start": 0,
    "end": 22.957166666666666,
    "measuredSyntheticWithPauses": 22.957,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Introduce the speakers warmly, then place gentle emphasis on the question."
      },
      {
        "time": 11.5,
        "instruction": "Let the central question appear; use a slight pitch rise, then settle."
      }
    ],
    "notes": "Hi everyone. We're Noah and Fernando, and today we're looking at VLA JEPA. The idea starts with a simple question: if a robot watches someone move an object, what should it actually learn from that video? We'll explain how the paper turns that question into a training method, and then look at where the evidence is convincing and where it still falls short.",
    "backup": false,
    "id": "chapter-1",
    "audioAsset": "narration-01.mp3"
  },
  {
    "hidden": true,
    "id": "reserved-retired-intro"
  },
  {
    "n": 2,
    "speaker": "Noah",
    "budget": 37.9,
    "section": "Problem and insight",
    "title": "What can video teach a robot?",
    "purpose": "Make the task and representation problem concrete before introducing components.",
    "chunks": [
      {
        "text": "Imagine asking a robot to put an apple into a bowl. Just watch the movement for a moment. What information would help the robot do that itself?",
        "pause": 1.2,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 8.505
      },
      {
        "text": "The useful part is the transition: where the object starts, how it moves, and where it ends up. But a video also contains things we don't want the robot to depend on, like the background or the camera angle. Robot demonstrations give us actual control commands, but collecting them takes work. Human videos are much easier to find; they just don't tell us which commands a robot should execute. That's the gap this paper tries to bridge.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 26.795
      }
    ],
    "source": "Human footage: Something-Something-v2 validation sample 174198.webm, label putting jar into box, template Putting [something] into [something]. Retrieved from the public morpheushoc/something-something-v2 archive mirror and checked against its validation.json annotation; native 427x240, 12 fps, 5.916 seconds. Cropped to 426x192; output duplicates source frames at 60 fps. Before/during/after stills are from this same clip. This is a verified V2 dataset example, not an identified VLA-JEPA training example. Dataset: https://www.qualcomm.com/developer/software/something-something-v-2-dataset . Mirror: https://huggingface.co/datasets/morpheushoc/something-something-v2 . Robot footage: real DROID episode AUTOLab+0d4edc83+2023-10-27-19h-52m-50s, exterior camera 24400334. Cropped; approximate timing restored from the 5.901-second recorded control-timestamp span, with an end hold. Paired trajectory.h5 contains recorded controls. Both videos play once silently, then hold. This is dataset footage, not a VLA-JEPA policy rollout. https://droid-dataset.github.io/visualizer/",
    "wordCount": 104,
    "pauseSeconds": 2.6,
    "start": 22.957,
    "end": 60.85679166666666,
    "measuredSyntheticWithPauses": 27.795,
    "deliveryCues": [
      {
        "time": 0,
        "instruction": "Ask the opening question. Let both real clips play; keep the cursor still."
      },
      {
        "time": 6.8,
        "instruction": "The human clip resolves into before, during and after frames. Both clips now hold."
      },
      {
        "time": 10.105,
        "instruction": "Explain the useful transition before the data gap."
      },
      {
        "time": 24.005000000000003,
        "instruction": "Stress “control commands” versus “observable transitions.”"
      }
    ],
    "notes": "Imagine asking a robot to put an apple into a bowl. Just watch the movement for a moment. What information would help the robot do that itself?\n\nThe useful part is the transition: where the object starts, how it moves, and where it ends up. But a video also contains things we don't want the robot to depend on, like the background or the camera angle. Robot demonstrations give us actual control commands, but collecting them takes work. Human videos are much easier to find; they just don't tell us which commands a robot should execute. That's the gap this paper tries to bridge.",
    "id": "chapter-2",
    "audioAsset": "narration-03.mp3"
  },
  {
    "n": 3,
    "speaker": "Noah",
    "budget": 43.798,
    "section": "Related work",
    "title": "Feature prediction is shared; the pipeline differs",
    "purpose": "Explain the contribution relative to robot-supervised VLAs, latent-action learning, and JEPA.",
    "chunks": [
      {
        "text": "It's tempting to say that earlier methods predict pixels while JEPA predicts features. But that would miss an important detail. LAPA learns discrete latent actions using pixel reconstruction. UniVLA already reconstructs DINO features, conditioned on language, and then trains a policy to predict discrete action codes. VLA JEPA instead obtains continuous latent tokens directly from the policy's current observation and instruction. Those tokens help a world predictor estimate future state features, so the alignment loss trains the policy representation itself. The distinction is the information pathway and training pipeline, not simply pixels versus embeddings. Robot demonstrations then teach the action head how to turn that representation into control.",
        "pause": 0.6,
        "focus": "Distinguish transition tokens, state targets, and controls",
        "voice": "Warm connected conversational explanation. Emphasize the precise distinction without reading as a list."
      }
    ],
    "source": "LAPA (2024), §3; UniVLA (RSS 2025), §III; VLA-JEPA (2026), §3. Different targets and pipelines; feature prediction is not unique to VLA-JEPA.",
    "wordCount": 108,
    "pauseSeconds": 1.0,
    "start": 60.857,
    "end": 104.65520833333335,
    "measuredSyntheticWithPauses": 43.798,
    "deliveryCues": [
      {
        "time": 0,
        "instruction": "Begin conversationally; allow the example or shared objects to establish context."
      },
      {
        "time": 17.5192,
        "instruction": "Stress the distinction between transition tokens and the future-state target."
      },
      {
        "time": 35.0384,
        "instruction": "Settle on the output, then make the handoff without a long pause."
      }
    ],
    "notes": "It's tempting to say that earlier methods predict pixels while JEPA predicts features. But that would miss an important detail. LAPA learns discrete latent actions using pixel reconstruction. UniVLA already reconstructs DINO features, conditioned on language, and then trains a policy to predict discrete action codes. VLA JEPA instead obtains continuous latent tokens directly from the policy's current observation and instruction. Those tokens help a world predictor estimate future state features, so the alignment loss trains the policy representation itself. The distinction is the information pathway and training pipeline, not simply pixels versus embeddings. Robot demonstrations then teach the action head how to turn that representation into control.",
    "backup": false,
    "id": "chapter-3",
    "audioAsset": "narration-04.mp3"
  },
  {
    "n": 4,
    "speaker": "Noah",
    "budget": 45.201,
    "section": "Method 1 of 5 · Human video supervision",
    "title": "Predict a future state in feature space",
    "purpose": "Explain the JEPA target/predictor distinction using the original paper figure.",
    "chunks": [
      {
        "text": "Think of watching someone put a jar into a box. You can anticipate the outcome without drawing the next frame in your head. That's a useful analogy, although the model's features aren't literal labels like jar inside box. Here's the precise distinction: the latent action tokens represent the intended transition. The world predictor uses those tokens and encoded state history to predict the next state's embedding. A frozen V JEPA two encoder processes the actual future video to provide the target. We compare those two feature representations. We don't reconstruct a future image, and these features aren't robot joint coordinates. Robot control comes from a separate action head. Let's follow the same pieces through the model.",
        "pause": 0.6,
        "focus": "Distinguish transition tokens, state targets, and controls",
        "voice": "Warm connected conversational explanation. Emphasize the precise distinction without reading as a list."
      }
    ],
    "source": "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    "wordCount": 116,
    "pauseSeconds": 1.0,
    "start": 104.655,
    "end": 149.856375,
    "measuredSyntheticWithPauses": 45.201,
    "deliveryCues": [
      {
        "time": 0,
        "instruction": "Begin conversationally; allow the example or shared objects to establish context."
      },
      {
        "time": 18.0804,
        "instruction": "Stress the distinction between transition tokens and the future-state target."
      },
      {
        "time": 36.1608,
        "instruction": "Settle on the output, then make the handoff without a long pause."
      }
    ],
    "notes": "Think of watching someone put a jar into a box. You can anticipate the outcome without drawing the next frame in your head. That's a useful analogy, although the model's features aren't literal labels like jar inside box. Here's the precise distinction: the latent action tokens represent the intended transition. The world predictor uses those tokens and encoded state history to predict the next state's embedding. A frozen V JEPA two encoder processes the actual future video to provide the target. We compare those two feature representations. We don't reconstruct a future image, and these features aren't robot joint coordinates. Robot control comes from a separate action head. Let's follow the same pieces through the model.",
    "backup": false,
    "id": "chapter-4",
    "audioAsset": "narration-05.mp3"
  },
  {
    "n": 5,
    "speaker": "Noah",
    "budget": 22.965,
    "section": "Method 2 of 5 · Architecture",
    "title": "First, encode the observation and instruction",
    "purpose": "Introduce the first architecture path before revealing the predictor.",
    "chunks": [
      {
        "text": "First, the policy reads the current observation and the language instruction. Its vision language backbone is Qwen three V L, with two billion parameters. The output we're focusing on is a set of latent action tokens. Think of these as an internal description of a transition. They will help predict what changes, but they aren't yet commands we can send to a robot.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 21.965
      }
    ],
    "source": "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    "wordCount": 63,
    "pauseSeconds": 1.0,
    "start": 149.856,
    "end": 172.820625,
    "measuredSyntheticWithPauses": 22.965,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Focus on observation and instruction."
      },
      {
        "time": 7.7,
        "instruction": "The VLM processes both inputs."
      },
      {
        "time": 15.3,
        "instruction": "Stress “internal representation”; lower pitch on “not motor commands.”"
      }
    ],
    "notes": "First, the policy reads the current observation and the language instruction. Its vision language backbone is Qwen three V L, with two billion parameters. The output we're focusing on is a set of latent action tokens. Think of these as an internal description of a transition. They will help predict what changes, but they aren't yet commands we can send to a robot.",
    "backup": false,
    "id": "chapter-5",
    "audioAsset": "narration-06.mp3"
  },
  {
    "n": 6,
    "speaker": "Noah",
    "budget": 22.973,
    "section": "Method 2 of 5 · Architecture",
    "title": "Predict the next state’s embedding",
    "purpose": "Define state embeddings and latent actions, then give implementation detail without a dense architecture dump.",
    "chunks": [
      {
        "text": "Now carry those tokens into the world predictor. The frozen video encoder represents the observed state history. The predictor combines that history with the latent action tokens and estimates the next state's embedding. So the tokens describe a transition, while the output describes the future state. There's no image decoder here. The actual future supplies a training target; it doesn't enter the policy branch.",
        "pause": 0.6,
        "focus": "Distinguish transition tokens, state targets, and controls",
        "voice": "Warm connected conversational explanation. Emphasize the precise distinction without reading as a list."
      }
    ],
    "source": "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    "wordCount": 64,
    "pauseSeconds": 1.0,
    "start": 172.821,
    "end": 195.79441666666668,
    "measuredSyntheticWithPauses": 22.973,
    "deliveryCues": [
      {
        "time": 0,
        "instruction": "Begin conversationally; allow the example or shared objects to establish context."
      },
      {
        "time": 9.1892,
        "instruction": "Stress the distinction between transition tokens and the future-state target."
      },
      {
        "time": 18.3784,
        "instruction": "Settle on the output, then make the handoff without a long pause."
      }
    ],
    "notes": "Now carry those tokens into the world predictor. The frozen video encoder represents the observed state history. The predictor combines that history with the latent action tokens and estimates the next state's embedding. So the tokens describe a transition, while the output describes the future state. There's no image decoder here. The actual future supplies a training target; it doesn't enter the policy branch.",
    "backup": false,
    "id": "chapter-6",
    "audioAsset": "narration-07.mp3"
  },
  {
    "n": 7,
    "speaker": "Noah",
    "budget": 25.905,
    "section": "Method 2 of 5 · Architecture",
    "title": "Compare the prediction with the observed future",
    "purpose": "Define state embeddings and latent actions, then give implementation detail without a dense architecture dump.",
    "chunks": [
      {
        "text": "Finally, we need something to check that prediction against. The same frozen encoder processes the actual future frames and produces the target embedding. The alignment loss compares the prediction with that target. Training updates the prediction path, including the vision language model, while the target encoder stays fixed. That's how video supervision shapes the policy's internal representation.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 24.905
      }
    ],
    "source": "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    "wordCount": 57,
    "pauseSeconds": 1.0,
    "start": 195.79399999999998,
    "end": 221.69858333333332,
    "measuredSyntheticWithPauses": 25.905,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Focus on the actual future and frozen encoder."
      },
      {
        "time": 8.6,
        "instruction": "Identify target and prediction by both labels and colors."
      },
      {
        "time": 17.3,
        "instruction": "Slow on “stays fixed”; explain that the prediction path learns."
      }
    ],
    "notes": "Finally, we need something to check that prediction against. The same frozen encoder processes the actual future frames and produces the target embedding. The alignment loss compares the prediction with that target. Training updates the prediction path, including the vision language model, while the target encoder stays fixed. That's how video supervision shapes the policy's internal representation.",
    "backup": false,
    "id": "chapter-7",
    "audioAsset": "narration-08.mp3"
  },
  {
    "n": 8,
    "speaker": "Noah",
    "budget": 41.017,
    "section": "Method 3 of 5 · Information flow",
    "title": "The predictor sees history, while future states are targets",
    "purpose": "Resolve the subtle difference between target supervision and teacher-forced world-model history.",
    "chunks": [
      {
        "text": "There's an important boundary here: the policy doesn't get to see the answer. Its latent action tokens come from the initial observations and the instruction. The world model can use state history during training, through teacher forcing, but its attention stays causal across time. In other words, it can use earlier states, not the future state it's supposed to predict. The future embeddings sit on the target side of the comparison. That separation matters because otherwise a good prediction could just be a shortcut, without learning a useful transition. So far we've learned a representation. Fernando will explain how that becomes an action.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 40.017
      }
    ],
    "source": "VLA-JEPA, §3.2. Human frames: SSV2 validation example #174198. Diagram features are illustrative, not measured embeddings or explicit robot coordinates.",
    "wordCount": 103,
    "pauseSeconds": 1.0,
    "start": 221.69899999999998,
    "end": 262.7159583333333,
    "measuredSyntheticWithPauses": 41.017,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Clearly separate policy inputs from world-model history."
      },
      {
        "time": 13.7,
        "instruction": "Slow on teacher forcing and causal attention."
      },
      {
        "time": 27.3,
        "instruction": "Emphasize the future is a comparison target, then hand off to Fernando."
      }
    ],
    "notes": "There's an important boundary here: the policy doesn't get to see the answer. Its latent action tokens come from the initial observations and the instruction. The world model can use state history during training, through teacher forcing, but its attention stays causal across time. In other words, it can use earlier states, not the future state it's supposed to predict. The future embeddings sit on the target side of the comparison. That separation matters because otherwise a good prediction could just be a shortcut, without learning a useful transition. So far we've learned a representation. Fernando will explain how that becomes an action.",
    "backup": false,
    "id": "chapter-8",
    "audioAsset": "narration-09.mp3"
  },
  {
    "n": 9,
    "speaker": "Fernando",
    "budget": 35.317,
    "section": "Method 4 of 5 · Action generation",
    "title": "Learn the velocity from one training pair",
    "purpose": "Explain interpolation, velocity supervision, and inference as separate operations.",
    "chunks": [
      {
        "text": "Thanks, Noah. To understand the action head, let's start with one training example. The blue point represents a demonstrated action chunk. We also sample a noise point, shown in gray. During training, we choose a point somewhere between them. The orange arrow is the direction from that noise sample toward the demonstration. The network sees the mixed point, the mixing time, and the latent action tokens, and predicts a velocity. We train that prediction to match the orange arrow. Then we repeat this with many pairs and many mixing times. This drawing is a two dimensional illustration; the real points represent higher dimensional action chunks.",
        "pause": 0.6,
        "focus": "Training pair; mixed sample; target and predicted arrows",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 34.317
      }
    ],
    "source": "Paper §3.3, Eqs. 7–8. Mixing time is generation time, not physical robot time. The target velocity is demonstration minus sampled noise; squared-error regression. Own 2D illustration inspired by Jia-Bin Huang and Julia Turc videos; not an empirical trajectory.",
    "wordCount": 105,
    "pauseSeconds": 1.0,
    "start": 262.716,
    "end": 298.0328333333333,
    "measuredSyntheticWithPauses": 35.317,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Show the blue demonstration and gray noise before introducing mixing."
      },
      {
        "time": 8.8,
        "instruction": "Follow the orange mixed sample, without reading symbols."
      },
      {
        "time": 17.7,
        "instruction": "Explain target direction, then the predicted arrow."
      },
      {
        "time": 26.5,
        "instruction": "As the predicted arrow aligns, stress “match”; qualify the 2D illustration."
      }
    ],
    "notes": "Thanks, Noah. To understand the action head, let's start with one training example. The blue point represents a demonstrated action chunk. We also sample a noise point, shown in gray. During training, we choose a point somewhere between them. The orange arrow is the direction from that noise sample toward the demonstration. The network sees the mixed point, the mixing time, and the latent action tokens, and predicts a velocity. We train that prediction to match the orange arrow. Then we repeat this with many pairs and many mixing times. This drawing is a two dimensional illustration; the real points represent higher dimensional action chunks.",
    "backup": false,
    "id": "chapter-9",
    "audioAsset": "narration-10.mp3"
  },
  {
    "n": 10,
    "speaker": "Fernando",
    "budget": 29.41,
    "section": "Method 4 of 5 · Action generation",
    "title": "Generate actions by following the learned field",
    "purpose": "Explain interpolation, velocity supervision, and inference as separate operations.",
    "chunks": [
      {
        "text": "Generating an action is a different process. This time we don't have a demonstration to aim at. We start with fresh noise and ask the learned velocity field which way to move. After each small update, we ask again at the new position. Following that field gradually gives us an action chunk, conditioned on the latent tokens. The paper uses four integration steps in its reported setup. The bend in this sketch is illustrative: the straight training pair doesn't mean every generated path has to be straight.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 28.41
      }
    ],
    "source": "Paper §3.3 and Appendix A: four denoising/integration steps, 7-dimensional actions, future action horizon 7. Own illustrative trajectory; sampler type is not asserted. Training and inference paths are different concepts.",
    "wordCount": 87,
    "pauseSeconds": 1.0,
    "start": 298.033,
    "end": 327.44291666666663,
    "measuredSyntheticWithPauses": 29.41,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Start with the distinction: no demonstration is provided at inference."
      },
      {
        "time": 9.8,
        "instruction": "Follow the successive updates; do not move the pointer continuously."
      },
      {
        "time": 19.6,
        "instruction": "Explain four steps as a paper setting; the bent path is illustrative."
      }
    ],
    "notes": "Generating an action is a different process. This time we don't have a demonstration to aim at. We start with fresh noise and ask the learned velocity field which way to move. After each small update, we ask again at the new position. Following that field gradually gives us an action chunk, conditioned on the latent tokens. The paper uses four integration steps in its reported setup. The bend in this sketch is illustrative: the straight training pair doesn't mean every generated path has to be straight.",
    "backup": false,
    "id": "chapter-10",
    "audioAsset": "narration-11.mp3"
  },
  {
    "n": 11,
    "speaker": "Fernando",
    "budget": 35.894,
    "section": "Method 5 of 5 · Joint optimization",
    "title": "Human and robot data provide different supervision",
    "purpose": "Explain which loss applies to each data source and what remains frozen.",
    "chunks": [
      {
        "text": "Now the two sources of training data fit together. Human videos provide the predictive alignment objective, because they tell us what happened next. Robot demonstrations provide that objective too, plus the action labels needed for flow matching. The robot loss combines the action objective with a weighted world model objective; beta controls that weighting. Pretraining uses roughly two hundred and twenty thousand human videos and seventy six thousand robot trajectories. The important distinction is that human video helps shape the representation, while robot data still anchors it to physical control.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 34.894
      }
    ],
    "source": "Paper, equation 9, sections 3.3 and 4.1; SSv2 220K videos, DROID 76K trajectories. Equation 5 is presented as an embedding discrepancy without an explicit norm; this presentation does not invent an L1 or Smooth-L1 choice.",
    "wordCount": 90,
    "pauseSeconds": 1.0,
    "start": 327.44300000000004,
    "end": 363.33695833333337,
    "measuredSyntheticWithPauses": 35.894,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Separate human-video prediction supervision from robot action supervision."
      },
      {
        "time": 12.0,
        "instruction": "Slow on the weighting of the two objectives."
      },
      {
        "time": 23.9,
        "instruction": "End on the role of robot data in physical control."
      }
    ],
    "notes": "Now the two sources of training data fit together. Human videos provide the predictive alignment objective, because they tell us what happened next. Robot demonstrations provide that objective too, plus the action labels needed for flow matching. The robot loss combines the action objective with a weighted world model objective; beta controls that weighting. Pretraining uses roughly two hundred and twenty thousand human videos and seventy six thousand robot trajectories. The important distinction is that human video helps shape the representation, while robot data still anchors it to physical control.",
    "backup": false,
    "id": "chapter-11",
    "audioAsset": "narration-12.mp3"
  },
  {
    "n": 12,
    "speaker": "Fernando",
    "budget": 14.926,
    "section": "Results · Benchmark comparison",
    "title": "Standard LIBERO: a near tie",
    "purpose": "Compare matched benchmark numbers with correct units and a near-tie caveat.",
    "chunks": [
      {
        "text": "On standard LIBERO, the headline result is basically a tie: ninety seven point two percent for VLA JEPA, versus ninety seven point one for OpenVLA O F T. So this comparison alone isn't a strong argument for the method.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 13.926
      }
    ],
    "source": "Paper, Tables 1 and 3. Differences: 97.2−97.1=0.1 percentage points, 79.5−69.6=9.9 percentage points. Selected baseline is OpenVLA-OFT; this is not a claim of superiority to every model on every task. Paper reports 50 episodes per task on standard LIBERO and does not provide confidence intervals for these headline averages.",
    "wordCount": 39,
    "pauseSeconds": 1.0,
    "start": 363.33700000000005,
    "end": 378.26254166666666,
    "measuredSyntheticWithPauses": 14.926,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Read the two percentages naturally, then stress “basically a tie.”"
      },
      {
        "time": 7.5,
        "instruction": "Keep the conclusion modest; do not celebrate 0.1 pp."
      }
    ],
    "notes": "On standard LIBERO, the headline result is basically a tie: ninety seven point two percent for VLA JEPA, versus ninety seven point one for OpenVLA O F T. So this comparison alone isn't a strong argument for the method.",
    "backup": false,
    "id": "chapter-12",
    "audioAsset": "narration-13.mp3"
  },
  {
    "n": 13,
    "speaker": "Fernando",
    "budget": 24.775,
    "section": "Results · Benchmark comparison",
    "title": "LIBERO-Plus: a larger gain under perturbations",
    "purpose": "Compare matched benchmark numbers with correct units and a near-tie caveat.",
    "chunks": [
      {
        "text": "The more interesting comparison is LIBERO Plus, which changes things such as the camera, lighting, and object layout. Here VLA JEPA reaches seventy nine point five percent, compared with sixty nine point six for OpenVLA O F T. That's a gain of nine point nine percentage points. This supports the robustness argument under these particular shifts, although an average doesn't tell us which failures remain.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 23.775
      }
    ],
    "source": "Paper, Tables 1 and 3. Differences: 97.2−97.1=0.1 percentage points, 79.5−69.6=9.9 percentage points. Selected baseline is OpenVLA-OFT; this is not a claim of superiority to every model on every task. Paper reports 50 episodes per task on standard LIBERO and does not provide confidence intervals for these headline averages.",
    "wordCount": 65,
    "pauseSeconds": 1.0,
    "start": 378.26300000000003,
    "end": 403.037625,
    "measuredSyntheticWithPauses": 24.775,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Name the perturbations before comparing numbers."
      },
      {
        "time": 8.3,
        "instruction": "Stress “nine point nine percentage points”, not percent improvement."
      },
      {
        "time": 16.5,
        "instruction": "Settle on the limit: an average hides remaining failures."
      }
    ],
    "notes": "The more interesting comparison is LIBERO Plus, which changes things such as the camera, lighting, and object layout. Here VLA JEPA reaches seventy nine point five percent, compared with sixty nine point six for OpenVLA O F T. That's a gain of nine point nine percentage points. This supports the robustness argument under these particular shifts, although an average doesn't tell us which failures remain.",
    "backup": false,
    "id": "chapter-13",
    "audioAsset": "narration-14.mp3"
  },
  {
    "n": 14,
    "speaker": "Fernando",
    "budget": 21.091,
    "section": "Results · Human-video ablation",
    "title": "Human video improves LIBERO-Plus robustness",
    "purpose": "Use controlled within-method ablation and a counterexample to temper the main claim.",
    "chunks": [
      {
        "text": "What happens if we remove the human video training? On LIBERO Plus, success drops from seventy nine point five to sixty two point nine percent. That's sixteen point six percentage points within the same method. This is stronger evidence for the contribution of human video than just comparing two different models, because the ablation changes that part of training directly.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 20.091
      }
    ],
    "source": "Paper, Tables 1–3 and sections 4.4–4.5. Within-method ablation deltas with human video minus without: LIBERO-Plus +16.6 pp; LIBERO +1.1 pp; SimplerEnv Google −13.2 pp. Real-world repeated grasping is an author-reported qualitative observation, with a proposed attribution to human videos, not an isolated causal proof.",
    "wordCount": 60,
    "pauseSeconds": 1.0,
    "start": 403.038,
    "end": 424.12874999999997,
    "measuredSyntheticWithPauses": 21.091,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Frame this as a within-method ablation."
      },
      {
        "time": 7.0,
        "instruction": "Stress the 16.6 percentage-point difference."
      },
      {
        "time": 14.1,
        "instruction": "Explain why this comparison isolates the human-video contribution more directly."
      }
    ],
    "notes": "What happens if we remove the human video training? On LIBERO Plus, success drops from seventy nine point five to sixty two point nine percent. That's sixteen point six percentage points within the same method. This is stronger evidence for the contribution of human video than just comparing two different models, because the ablation changes that part of training directly.",
    "backup": false,
    "id": "chapter-14",
    "audioAsset": "narration-15.mp3"
  },
  {
    "n": 15,
    "speaker": "Fernando",
    "budget": 19.233,
    "section": "Results · Human-video ablation",
    "title": "Human video does not improve every benchmark",
    "purpose": "Use controlled within-method ablation and a counterexample to temper the main claim.",
    "chunks": [
      {
        "text": "But the benefit isn't universal. On the Google robot tasks in SimplerEnv, adding human video reduces success from seventy eight point four to sixty five point two percent. So we should say the evidence is strongest for robustness on LIBERO Plus, rather than claim that human video improves every task.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 18.233
      }
    ],
    "source": "Paper, Tables 1–3 and sections 4.4–4.5. Within-method ablation deltas with human video minus without: LIBERO-Plus +16.6 pp; LIBERO +1.1 pp; SimplerEnv Google −13.2 pp. Real-world repeated grasping is an author-reported qualitative observation, with a proposed attribution to human videos, not an isolated causal proof.",
    "wordCount": 50,
    "pauseSeconds": 1.0,
    "start": 424.129,
    "end": 443.36179166666665,
    "measuredSyntheticWithPauses": 19.233,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Use a neutral pitch to introduce the counterexample."
      },
      {
        "time": 9.6,
        "instruction": "Stress the drop, then narrow the overall claim."
      }
    ],
    "notes": "But the benefit isn't universal. On the Google robot tasks in SimplerEnv, adding human video reduces success from seventy eight point four to sixty five point two percent. So we should say the evidence is strongest for robustness on LIBERO Plus, rather than claim that human video improves every task.",
    "backup": false,
    "id": "chapter-15",
    "audioAsset": "narration-16.mp3"
  },
  {
    "speaker": "Fernando",
    "budget": 34.265,
    "section": "Our experiments · Preliminary results",
    "title": "Single GPU Quantization Trade-offs",
    "purpose": "Preserve the preliminary memory and performance comparison with the original metric distinction.",
    "chunks": [
      {
        "text": "We also have our own preliminary single GPU results. On an R T X thirty ninety, the full precision setup peaks at nineteen point four gigabytes. Eight bit quantization brings that down to eleven point eight, while retaining ninety six point eight percent of baseline success. That's roughly forty percent less memory. Four bit quantization reaches seven point two gigabytes, but task accuracy falls below sixty percent and we saw action drift. So eight bit looks like the more promising compromise in this pilot. These are our preliminary deployment findings, separate from the paper's benchmark results.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 33.265
      }
    ],
    "source": "Presenters’ original slides, “Single GPU Quantization Trade-offs,” confirmed as preliminary runs. FP16/B16: 19.4 GB peak VRAM and 100% relative success; INT8/Q8: 11.8 GB and 96.8% baseline-success retention; INT4/Q4: 7.2 GB with severe action drift and <60% task accuracy. These performance labels are kept distinct. INT8 memory reduction is (19.4−11.8)/19.4 = 39.2%. RTX3090 has 24 GB. These are our preliminary results, not results of Sun et al. The original does not specify trial count, checkpoint or uncertainty. “Promising” avoids claiming a demonstrated optimum.",
    "n": 16,
    "start": 443.362,
    "end": 477.62733333333335,
    "wordCount": 96,
    "pauseSeconds": 1.0,
    "measuredSyntheticWithPauses": 34.265,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Explicitly identify these as our preliminary results."
      },
      {
        "time": 11.4,
        "instruction": "Contrast memory and retention at INT8."
      },
      {
        "time": 22.8,
        "instruction": "Slow on the INT4 accuracy loss and action drift; conclude cautiously."
      }
    ],
    "notes": "We also have our own preliminary single GPU results. On an R T X thirty ninety, the full precision setup peaks at nineteen point four gigabytes. Eight bit quantization brings that down to eleven point eight, while retaining ninety six point eight percent of baseline success. That's roughly forty percent less memory. Four bit quantization reaches seven point two gigabytes, but task accuracy falls below sixty percent and we saw action drift. So eight bit looks like the more promising compromise in this pilot. These are our preliminary deployment findings, separate from the paper's benchmark results.",
    "backup": false,
    "id": "chapter-16",
    "audioAsset": "narration-17.mp3"
  },
  {
    "n": 17,
    "speaker": "Fernando",
    "budget": 28.52,
    "section": "Critical analysis",
    "title": "Deployment gaps remain beyond headline averages",
    "purpose": "Identify benchmark, statistical, semantic and hardware limits without inventing experiments.",
    "chunks": [
      {
        "text": "There are still gaps behind those averages. Under sensor noise, VLA JEPA trails pi zero in the reported comparison.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 7.182
      },
      {
        "text": "The real robot evaluation also has only ten trials per task, with some wrong object selections. That limits how much we can conclude about reliability.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "action": "Reveal the real-world evidence column.",
        "measuredSyntheticSpeechSeconds": 9.07
      },
      {
        "text": "And the paper trains on eight A one hundred GPUs. Our single GPU pilot is about deployment, so it doesn't establish that the same training is cheap.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "action": "Reveal training versus deployment.",
        "measuredSyntheticSpeechSeconds": 10.068
      }
    ],
    "source": "Paper, Table 3; sections 4.1 and 4.4; Appendix B. Ten trials per task refers to the real-world study. Eight A100 GPUs describes paper training, not a minimum inference requirement. Our preliminary deployment results are sourced from the presenters’ original slides.",
    "wordCount": 71,
    "pauseSeconds": 2.1999999999999997,
    "start": 477.627,
    "end": 506.1467083333333,
    "measuredSyntheticWithPauses": 28.52,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Explain the sensor-noise weakness."
      },
      {
        "time": 9.5,
        "instruction": "Reveal the ten-trial evidence with the second spoken segment."
      },
      {
        "time": 19.0,
        "instruction": "Reveal training versus deployment with the third segment."
      }
    ],
    "notes": "There are still gaps behind those averages. Under sensor noise, VLA JEPA trails pi zero in the reported comparison.\n\nThe real robot evaluation also has only ten trials per task, with some wrong object selections. That limits how much we can conclude about reliability.\n\nAnd the paper trains on eight A one hundred GPUs. Our single GPU pilot is about deployment, so it doesn't establish that the same training is cheap.",
    "backup": false,
    "id": "chapter-17",
    "audioAsset": "narration-18.mp3"
  },
  {
    "n": 18,
    "speaker": "Fernando",
    "budget": 19.4,
    "section": "Takeaway and discussion",
    "title": "Predictive supervision strengthens the policy",
    "purpose": "Close with a defensible takeaway and one answerable discussion question.",
    "chunks": [
      {
        "text": "The takeaway is that predicting future embeddings can give a robot policy useful supervision, with the clearest evidence here coming from robustness under perturbations. The next question we'd ask is: which experiment would separate visual robustness from actual physical reasoning? That's a useful starting point for discussion. Thank you.",
        "pause": 0.6,
        "focus": "Follow each visual stage",
        "voice": "Explain this to classmates naturally. Use connected phrasing, gentle emphasis on the key contrast, and relaxed sentence endings. Avoid an announcer cadence.",
        "measuredSyntheticSpeechSeconds": 18.4
      }
    ],
    "source": "Paper, sections 3 and 4.5. Suggested discussion experiment: controlled appearance changes versus changed dynamics, under matched data and compute. This is our proposed test, not a result in the paper.",
    "wordCount": 49,
    "pauseSeconds": 1.0,
    "start": 506.147,
    "end": 525.5472916666666,
    "measuredSyntheticWithPauses": 19.4,
    "deliveryCues": [
      {
        "time": 0.0,
        "instruction": "Deliver the takeaway with a falling pitch."
      },
      {
        "time": 9.7,
        "instruction": "Lift on the discussion question, then thank the audience."
      }
    ],
    "notes": "The takeaway is that predicting future embeddings can give a robot policy useful supervision, with the clearest evidence here coming from robustness under perturbations. The next question we'd ask is: which experiment would separate visual robustness from actual physical reasoning? That's a useful starting point for discussion. Thank you.",
    "backup": false,
    "id": "chapter-18",
    "audioAsset": "narration-19.mp3"
  },
  {
    "n": "B1",
    "speaker": "Noah",
    "budget": 35,
    "section": "Backup · Our preliminary experiments",
    "title": "SmolVLM Architectural Adaptation",
    "purpose": "Preserve the preliminary lightweight architecture work outside the ten-minute presentation.",
    "chunks": [
      {
        "text": "In our preliminary adaptation, SmolVLM replaces Qwen three V L, with integration through LeRobot. We retain the frozen V JEPA encoder and flow matching action head.",
        "pause": 2,
        "focus": "Adaptation and components",
        "voice": "Explain the component change with a neutral tone.",
        "measuredSyntheticSpeechSeconds": 11.949
      },
      {
        "text": "The original experiments report over sixty percent less backbone memory. State and action tokens use cross attention fusion. We retain pretrained feature alignment, but this does not establish equivalent task performance.",
        "pause": 2,
        "focus": "Memory finding and interpretation",
        "voice": "Stress backbone memory; slow on the limitation.",
        "measuredSyntheticSpeechSeconds": 14.695
      }
    ],
    "source": "Presenters’ original slides, confirmed as preliminary experiments. Do not attribute these findings to the VLA-JEPA paper. Exact evaluation details and uncertainty are not specified in the source slides. Original slide cites Marafioti et al. (2025). The reported >60% is backbone memory reduction, not necessarily total peak VRAM. Retained feature alignment does not prove equal downstream success.",
    "wordCount": 57,
    "pauseSeconds": 6,
    "measuredSyntheticWithPauses": 32.643,
    "notes": "In our preliminary adaptation, SmolVLM replaces Qwen three V L, with integration through LeRobot. We retain the frozen V JEPA encoder and flow matching action head.\n\nThe original experiments report over sixty percent less backbone memory. State and action tokens use cross attention fusion. We retain pretrained feature alignment, but this does not establish equivalent task performance.",
    "deliveryCues": [
      {
        "time": 0,
        "instruction": "Optional reference outside the narrated main talk."
      }
    ],
    "backup": true,
    "id": "chapter-b1",
    "audioAsset": null
  },
  {
    "n": "B2",
    "speaker": "Fernando",
    "budget": 35,
    "section": "Backup · Our preliminary experiments",
    "title": "Dream-RSI Self-Improving Search",
    "purpose": "Retain the exploratory search result and identify the selection criterion.",
    "chunks": [
      {
        "text": "Our preliminary Dream R S I harness explores attention layers and projection tokens overnight on LIBERO. Candidates are compared by latency, parameter count, and task success.",
        "pause": 2,
        "focus": "Search and selection",
        "voice": "Measured explanatory tone; pause after the three criteria.",
        "measuredSyntheticSpeechSeconds": 12.471
      },
      {
        "text": "Candidate n zero zero zero eight reports stable policy retention with fewer parameters and a fourteen point two percent inference speedup from pruned attention. This is a preliminary candidate result.",
        "pause": 2,
        "focus": "Candidate result",
        "voice": "Stress preliminary; state speedup without overstating generality.",
        "measuredSyntheticSpeechSeconds": 12.915
      }
    ],
    "source": "Presenters’ original slides, confirmed as preliminary experiments. Do not attribute these findings to the VLA-JEPA paper. Exact evaluation details and uncertainty are not specified in the source slides. Original slide cites Zheng et al. (2026) and reports LIBERO overnight search, candidate n0008 and +14.2% inference speedup via pruned attention. The source does not specify exact baseline, trial count or error bars; stable retention has no numeric metric in the source.",
    "wordCount": 56,
    "pauseSeconds": 6,
    "measuredSyntheticWithPauses": 31.386,
    "notes": "Our preliminary Dream R S I harness explores attention layers and projection tokens overnight on LIBERO. Candidates are compared by latency, parameter count, and task success.\n\nCandidate n zero zero zero eight reports stable policy retention with fewer parameters and a fourteen point two percent inference speedup from pruned attention. This is a preliminary candidate result.",
    "deliveryCues": [
      {
        "time": 0,
        "instruction": "Optional reference outside the narrated main talk."
      }
    ],
    "backup": true,
    "id": "chapter-b2",
    "audioAsset": null
  },
  {
    "n": "B3",
    "speaker": "Noah",
    "budget": 40,
    "section": "Method 2 of 5 · Architecture",
    "title": "Architecture overview",
    "purpose": "Define state embeddings and latent actions, then give implementation detail without a dense architecture dump.",
    "chunks": [
      {
        "text": "This is our redraw of the architecture for technical discussion. The frozen encoder represents observed states and supplies future targets. The vision-language model produces latent action tokens. Those tokens condition both the world-model predictor and the action head. Human video supervises future-state prediction; robot demonstrations also supply action labels.",
        "pause": 2,
        "focus": "Original Figure 1",
        "voice": "Trace encoder, latent action, predictor and target in that order.",
        "measuredSyntheticSpeechSeconds": 27.036
      }
    ],
    "source": "Own redraw of the paper architecture after the component explanation; paper Figure 1 remains the authoritative reference.",
    "wordCount": 62,
    "pauseSeconds": 4,
    "start": 170,
    "end": 240,
    "measuredSyntheticWithPauses": 31.036,
    "notes": "This is our redraw of the architecture for technical discussion. The frozen encoder represents observed states and supplies future targets. The vision-language model produces latent action tokens. Those tokens condition both the world-model predictor and the action head. Human video supervises future-state prediction; robot demonstrations also supply action labels.",
    "deliveryCues": [
      {
        "time": 0,
        "instruction": "Optional reference outside the narrated main talk."
      }
    ],
    "backup": true,
    "id": "chapter-b3",
    "audioAsset": null
  },
  {
    "n": "B4",
    "speaker": "Fernando",
    "title": "Flow-matching equations",
    "source": "Sun et al. (2026), equations 7–8. Mixing time is not physical robot time.",
    "chunks": [
      {
        "text": "For reference, training interpolates between sampled Gaussian noise and a demonstrated action chunk. The learned velocity is regressed onto the difference between the demonstration and that noise sample."
      }
    ],
    "notes": "For reference, training interpolates between sampled Gaussian noise and a demonstrated action chunk. The learned velocity is regressed onto the difference between the demonstration and that noise sample.",
    "backup": true,
    "id": "chapter-b4",
    "audioAsset": null
  }
];const stage=document.getElementById('stage'),audio=document.getElementById('audio');let index=0,p=0,playing=false,animation=false,last=0,auto=false;
const ink='#142033',blue='#0284c7',gray='#94a3b8',muted='#64748b',orange='#bf531c',light='#e8f5fb';const clamp=x=>Math.max(0,Math.min(1,x));const ease=x=>{x=clamp(x);return x*x*(3-2*x)};const phase=(a,b)=>ease((p-a)/(b-a));
