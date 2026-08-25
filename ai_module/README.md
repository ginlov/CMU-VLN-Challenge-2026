# AI module — Team Xiao Hei

One entry point, two stacks, chosen by what the question asks for.

```
ros2 launch dummy_vlm dummy_vlm.launch
        │
        └── dummyVLM  (router)  — waits for /challenge_question, classifies it
              │
              ├── instruction-following ──▶ drives the route in this process
              │      VLM grounds the phrase in the 360 camera, the lidar says
              │      how far, the autonomy stack does the driving.
              │      → geometry_msgs/Pose2D on /way_point_with_heading
              │
              └── numerical / object-reference ──▶ exec scene_gemini.launch
                     Frontier exploration builds a perception scene graph
                     (YOLO-World v2 + SAM 2.1), then Gemini answers.
                     → std_msgs/Int32 on /numerical_response
                     → visualization_msgs/Marker on /selected_object_marker
```

The hand-off replaces the process rather than starting a second one. The
explorer publishes waypoints as soon as it comes up, so exactly one of the two
stacks may ever be alive. The question is not forwarded — the evaluation node
repeats it at 1 Hz, so the pipeline that takes over receives it on its own.

## Requirements

| | |
|---|---|
| **`ANTHROPIC_API_KEY`** | required, passed in at `docker run`. Instruction-following answers by calling the Claude API. |
| Gemini key | already baked into the image (see `docker/Dockerfile.full`); nothing to pass. |
| Network | outbound HTTPS to `api.anthropic.com` and the Gemini endpoint. |
| GPU | needed by the perception sidecar; the instruction-following stack is CPU-only. |

## Building and running

```bash
cd ai_module
docker/build.sh                       # adds our layers to oel20/cmu-vln-ai-module:latest

docker exec -it iros2026_ai_module bash
export ANTHROPIC_API_KEY=...          # supplied separately
ros2 launch dummy_vlm dummy_vlm.launch
```

`docker/run.sh` forwards `ANTHROPIC_API_KEY` from the host shell.

### Check the plumbing without an API key

```bash
python3 /opt/xiao_hei/vlm/challenge_node.py --selftest
```

Captures one frame from every subscribed topic, prints their shapes, publishes
a waypoint two metres ahead and reports whether the vehicle moved. No key, no
model call, no cost.

## Topics

Only topics on the challenge README's allowed list are used.

**Subscribed** — `/challenge_question` (String), `/camera/image` (Image),
`/registered_scan`, `/terrain_map`, `/terrain_map_ext` (PointCloud2),
`/state_estimation` (Odometry).

**Published** — `/way_point_with_heading` (Pose2D),
`/numerical_response` (Int32), `/selected_object_marker` (Marker).

`/way_point_reached` is deliberately **not** subscribed by the
instruction-following stack: it is not on the allowed list, and arrival is
decided from `/state_estimation` alone.

## Configuration

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **required** |
| `XIAO_HEI_CLASSIFY` | `claude` | `stub` forces every question to instruction-following (no classification call) |
| `XIAO_HEI_BUDGET_S` | `540` | seconds for the whole question, measured from process start — the challenge clock starts at system launch, and the margin covers the drive still in flight |
| `XIAO_HEI_IMAGE_TOPIC` | `/camera/image` | `/camera/image/compressed` to use the compressed stream |
| `XIAO_HEI_MODEL` | `claude-opus-5` | |
| `XIAO_HEI_GOTO_STEPS` | `20` | cap on grounding calls per destination; the real governor is the budget |
| `XIAO_HEI_OUT` | `/tmp/xiao_hei_run` | per-step log destination |
| `XIAO_HEI_OTHER_LAUNCH` | `dummy_vlm scene_gemini.launch` | what the hand-off launches |

## Layout

```
ai_module/
├── docker/
│   ├── Dockerfile        our layers on top of the published base image
│   ├── Dockerfile.full   from-source recipe for that base
│   ├── run.sh            + forwards ANTHROPIC_API_KEY
│   └── run_dev.sh        bind-mounts vlm/ so edits need no rebuild
├── src/dummy_vlm/        launch + entry point, names kept from the template
│   ├── src/dummyVLM.py   installed as `dummyVLM` — the router
│   └── launch/
│       ├── dummy_vlm.launch    starts the router
│       └── scene_gemini.launch the previous pipeline, verbatim
├── vlm/                  instruction-following, at /opt/xiao_hei/vlm
│   ├── challenge_node.py question in, driven trajectory out
│   ├── robot_node.py     ROS I/O: one frame of everything, or one waypoint
│   ├── classify.py       which of the three question types arrived
│   └── scripts/ perception/ src/
├── xiao_hei_vln/         the perception + Gemini responder
└── perception/           the sidecar it talks to
```

Each step of an instruction run writes its four camera faces, the model's
reply, the chosen waypoint and the drive result to
`$XIAO_HEI_OUT/<timestamp>/steps.jsonl`.

## Method, in one paragraph

For instruction-following, the sentence is decomposed once into an ordered list
of clauses. Per clause the loop captures the panorama, cuts it into four
perspective faces, and asks the model which pixels are the thing — never how
far away it is. A ray through the box centre is intersected with the registered
scan to get metres, and that reading is refused unless it survives a size check
and agrees with the previous binding. The waypoint published is the one the
converter is predicted to settle *at*, not the one we would like it to chase.
Arrival is geometry's call, not the model's. Throughout: the model proposes
semantics, geometry decides metres, the autonomy stack decides motion.
