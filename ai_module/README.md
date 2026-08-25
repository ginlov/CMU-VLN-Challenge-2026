# AI module — Team Xiao Hei

`dummy_vlm.launch` starts a router that reads `/challenge_question` and picks
the stack that answers it: instruction-following and numerical are answered in
that process; object reference `exec`s `scene_claude.launch`, which brings up
the perception sidecar (YOLO-World v2 + SAM 2.1) and answers with Claude.
Every question type calls the Claude API, so `ANTHROPIC_API_KEY` is required —
it is supplied with the submission, and is in neither this repository nor any
image we publish.

## Steps

**1. Build, with the key**, from the repository root. Two commands rather than
`up --build`, because `up` takes no `--build-arg`:

```bash
xhost +
cd docker
docker compose -f compose_gpu.yml build --build-arg ANTHROPIC_API_KEY=<the key>
docker compose -f compose_gpu.yml up -d          # compose.yml without a GPU
```

The key is now part of the image, so it is there for every `up`, `restart` and
`run` — including the relaunch between questions — and no launch needs an
export in front of it. It has to go in here because the `ai_module` service
declares no `environment:` block and `docker/compose*.yml` is outside
`ai_module/`, the only folder a submission should change. To check it landed:

```bash
docker compose -f compose_gpu.yml run --rm --no-deps --entrypoint sh \
  ai_module -c 'test -n "$ANTHROPIC_API_KEY" && echo key present || echo KEY MISSING'
```

For a one-off session without rebuilding, `docker exec -it -e
ANTHROPIC_API_KEY=<the key> iros2026_ai_module bash` also works, but has to be
repeated per shell and is lost on restart.

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

Without a key it logs the reason and exits here, rather than driving somewhere
first.

**4. Send a question** — either container, they share the ROS graph:

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the vase closest to the hookah'}"
```

## Layout

```
ai_module/
├── docker/
│   ├── Dockerfile        our layers on top of the published base image
│   ├── Dockerfile.full   from-source recipe for that base
│   ├── run.sh            standalone run, forwards ANTHROPIC_API_KEY
│   └── run_dev.sh        bind-mounts vlm/ so edits need no rebuild
├── src/dummy_vlm/        launch + entry point, names kept from the template
│   ├── src/dummyVLM.py   installed as `dummyVLM` — the router
│   └── launch/
│       ├── dummy_vlm.launch     starts the router
│       └── scene_claude.launch  object reference via nav_task1 + Claude
├── vlm/                  instruction-following, at /opt/xiao_hei/vlm
│   ├── challenge_node.py question in, driven trajectory out
│   ├── robot_node.py     ROS I/O: one frame of everything, or one waypoint
│   ├── classify.py       which of the three question types arrived
│   └── scripts/ perception/ src/
├── xiao_hei_vln/         the perception + scene_claude responder
└── perception/           the sidecar it talks to
```

Each step of a run writes its camera faces, the model's reply, the chosen
waypoint and the drive result to `$XIAO_HEI_OUT/<timestamp>/steps.jsonl`
(`XIAO_HEI_OUT` defaults to `/tmp/xiao_hei_run`).
