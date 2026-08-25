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
              ├── numerical ──▶ answers in this process
              │      → std_msgs/Int32 on /numerical_response
              │
              └── object reference ──▶ exec scene_claude.launch
                     The perception sidecar (YOLO-World v2 + SAM 2.1) builds
                     the scene graph while `nav_task1` drives toward the object
                     the question names; Claude then picks the referenced
                     object_id out of the built graph.
                     → visualization_msgs/Marker on /selected_object_marker
```

**One model API.** Every question type is answered by Claude — classification,
instruction following, numerical, and object reference. The Gemini pipeline
that used to answer object reference has been removed: the `google-genai` SDK
is not installed in the image, no Gemini key is baked into it, and
`scene_gemini.launch` is gone. `XIAO_HEI_RESPONDER=scene_gemini` raises with
that explanation rather than failing later on a missing key.

The hand-off replaces the process rather than starting a second one. The
explorer publishes waypoints as soon as it comes up, so exactly one of the two
stacks may ever be alive. The question is not forwarded — the evaluation node
repeats it at 1 Hz, so the pipeline that takes over receives it on its own.

## Requirements

| | |
|---|---|
| **`ANTHROPIC_API_KEY`** | required. Supplied with the submission and passed to `docker compose build --build-arg` — see [Supplying the API key](#supplying-the-api-key). It is not in this repository and not in any image we publish. The only key this module needs; every question type calls the Claude API. |
| Network | outbound HTTPS to `api.anthropic.com`. |
| GPU | needed by the perception sidecar; the instruction-following stack is CPU-only. |

## Building and running

This is the flow `docker/README.md` at the repository root documents, and the
one evaluation uses. Both containers come from the root `docker/` compose
files, not from `ai_module/docker/build.sh` — that script builds an image and
starts nothing.

**1. Bring both containers up**, from the repository root:

```bash
xhost +
cd docker
docker compose -f compose_gpu.yml build --build-arg ANTHROPIC_API_KEY=<the key>
docker compose -f compose_gpu.yml up -d          # compose.yml without a GPU
```

Build and up are two commands rather than `up --build`, because `up` takes no
`--build-arg`. Building is what picks up changes under `ai_module/`. Two
containers start: `iros2026_system` (simulator + autonomy stack) and
`iros2026_ai_module` (this module).

**2. Start the base autonomy system** — nothing moves without it:

```bash
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

**3. Start this module**, in a second terminal:

```bash
docker exec -it iros2026_ai_module bash
ros2 launch dummy_vlm dummy_vlm.launch
```

No export here: the key went in at build time above.

### Supplying the API key

**The key is not in this repository and not in any image we publish.** It is
sent with the submission, and goes in at build time:

```bash
cd docker
docker compose -f compose_gpu.yml build --build-arg ANTHROPIC_API_KEY=<the key>
docker compose -f compose_gpu.yml up -d
```

That is all. The key is now part of the image on your machine, so it is there
for every `up`, every `restart` and every `docker compose run` — including the
relaunch between questions — and nothing has to be exported before a launch.

To confirm it landed, before starting anything:

```bash
docker compose -f compose_gpu.yml run --rm --no-deps --entrypoint sh ai_module \
  -c 'test -n "$ANTHROPIC_API_KEY" && echo key present || echo KEY MISSING'
```

If it is missing, the module says so at boot and exits instead of driving
somewhere first:

```
ANTHROPIC_API_KEY is not set — this module answers by calling the Claude API
and can do nothing without it.
```

<details>
<summary>Why build time, and what to do if you would rather not rebuild</summary>

The `ai_module` service in `docker/compose*.yml` declares no `environment:`
block, so **nothing exported on the host reaches the container** — `export
ANTHROPIC_API_KEY=...` before `docker compose up` has no effect. That file sits
outside `ai_module/`, which the challenge README names as the only folder a
submission should change, so it is left alone.

A one-off session can pass the key straight to `docker exec`, which avoids a
rebuild but has to be repeated for every shell, and is lost when the container
restarts:

```bash
docker exec -it -e ANTHROPIC_API_KEY=<the key> iros2026_ai_module bash
ros2 launch dummy_vlm dummy_vlm.launch
```

If editing the compose file is acceptable after all, adding

```yaml
    environment:
      - ANTHROPIC_API_KEY
```

to the `ai_module` service lets the host environment through, and `--build-arg`
is then unnecessary. Any of the three works; the build-time one is the default
here only because it needs no repetition and survives a restart.

</details>

**4. Send a question** (either container — they share the ROS graph):

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the vase closest to the hookah'}"
```

### Standalone, without compose

`ai_module/docker/` has its own pair for working on this module alone. They
build and run `iros2026/ai_module:latest`, and `run.sh` forwards
`ANTHROPIC_API_KEY` from the host shell, so no export inside the container is
needed:

```bash
cd ai_module
docker/build.sh
export ANTHROPIC_API_KEY=...
docker/run.sh          # or run_dev.sh, which bind-mounts vlm/ so edits to the
                       # drive loop need no rebuild
```

Note these containers are unnamed, so `docker exec -it iros2026_ai_module` does
**not** reach them — that name belongs to the compose service. `run.sh` drops
you straight into the container instead.

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
| `XIAO_HEI_REFERENCE_LAUNCH` | `dummy_vlm scene_claude.launch` | which launch answers object reference |
| `XIAO_HEI_OTHER_LAUNCH` | — | the previous name for the above, still read |
| `XIAO_HEI_NAV_VLM_MODEL` | `claude-opus-5` | model for `scene_claude` navigation and answering |
| `XIAO_HEI_NAV_TASK1_COVERAGE_PLATEAU_S` | `45` | `scene_claude`: arrive once the scene graph stops gaining views for this long |
| `XIAO_HEI_NAV_TASK1_ANSWER_RESERVE_S` | `60` | `scene_claude`: seconds held back from navigation so the final answer call still lands inside the budget |

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
│       └── scene_claude.launch object reference via nav_task1 + Claude
├── vlm/                  instruction-following, at /opt/xiao_hei/vlm
│   ├── challenge_node.py question in, driven trajectory out
│   ├── robot_node.py     ROS I/O: one frame of everything, or one waypoint
│   ├── classify.py       which of the three question types arrived
│   └── scripts/ perception/ src/
├── xiao_hei_vln/         the perception + scene_claude responder
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
