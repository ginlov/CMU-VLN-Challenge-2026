#!/usr/bin/env python3
"""The AI module: one question in, a driven trajectory out.

This is what `ros2 launch dummy_vlm dummy_vlm.launch` starts. Two facts from
the challenge README shape the whole file:

  "The system will be relaunched for each language command tested such that
   information collected from previously exploring the scene is not retained."

so there is exactly one question per process, no loop over questions and no
state to carry between them; and

  "Timing will begin immediately at system startup."

so the ten-minute budget is measured from *here*, at import, not from the first
waypoint. `execute_plan`'s CLI starts its clock after the plan comes back from
the model; that would quietly hand us back the decompose call and the discovery
wait, which we do not have.

Debugging, from inside the container:

    ros2 launch dummy_vlm dummy_vlm.launch                  # what the graders run
    python3 /opt/xiao_hei/vlm/challenge_node.py --selftest  # plumbing only, no API
    python3 /opt/xiao_hei/vlm/challenge_node.py --question "go to the ..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Wall clock at import: as close to "system startup" as this process can get.
T0 = time.time()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))       # PYTHONPATH does this in the image;
                                            # repeated so the file runs anywhere

import numpy as np                                                # noqa: E402
import rclpy                                                      # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import Int32, String                            # noqa: E402

from classify import NUMERICAL, REFERENCE, classify                # noqa: E402
from robot_node import RobotNode                                   # noqa: E402

from answer_numerical import answer_numerical                     # noqa: E402
from approach_loop import COST_PER_CALL, Ctx                       # noqa: E402
import decompose as decompose_mod                                  # noqa: E402
from decompose import decompose                                    # noqa: E402
from execute_plan import BUDGET_S, execute                         # noqa: E402
from instruction_plan import keepouts, steps                       # noqa: E402
from vlm_probe import DEFAULT_PROMPT_VER                           # noqa: E402

BUDGET_S = float(os.environ.get("XIAO_HEI_BUDGET_S", BUDGET_S))
MODEL = os.environ.get("XIAO_HEI_MODEL", "claude-opus-5")
GOTO_STEPS = int(os.environ.get("XIAO_HEI_GOTO_STEPS", "20"))
OUT_ROOT = Path(os.environ.get("XIAO_HEI_OUT", "/tmp/xiao_hei_run"))

# The question is published once per startup but repeated at 1 Hz, so it cannot
# be missed. RELIABLE anyway: it is the one message whose loss ends the run.
QUESTION_QOS = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST)
# The answer goes back the way the question came: RELIABLE, and published more
# than once, because there is exactly one of it and its loss is the whole mark.
ANSWER_QOS = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)


def wait_for_question(node, timeout: float) -> str | None:
    """Block until `/challenge_question` says something, or give up."""
    got: list[str] = []

    def on_q(m) -> None:
        text = str(m.data).strip()
        if text and not got:
            got.append(text)

    sub = node.create_subscription(String, "/challenge_question", on_q,
                                   QUESTION_QOS)
    deadline = time.time() + timeout
    while not got and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return got[0] if got else None


# Which launch an object-reference question is handed to.
#
# One pipeline now: `nav_task1` drives at the object the question names, then
# Claude picks the object_id out of the perception scene graph. The Gemini
# sweep that used to answer these is gone — the team decided to run on one
# model API, so the Gemini SDK is no longer installed in the image, no Gemini
# key is baked into it, and `scene_gemini.launch` no longer exists.
#
# Still a variable rather than a literal, because the hand-off is the seam
# where a future pipeline would be swapped in and `os.execvp` should not need
# editing to do it. Pointing it at a launch file that is not installed fails
# loudly at exec, which is the correct outcome.
REFERENCE_LAUNCH = os.environ.get("XIAO_HEI_REFERENCE_LAUNCH") or os.environ.get(
    "XIAO_HEI_OTHER_LAUNCH", "dummy_vlm scene_claude.launch")


def handle_numerical(node, bot: RobotNode, question: str) -> int:
    """Find what the question is about, look at it, count, publish one integer.

    Numerical used to hand over with object reference, and no longer does. The
    two are not the same problem: eleven of the fifteen released counting
    questions are *anchor-local* — one piece of furniture, and the count is of
    small things on or above it — so answering one is finding an object and
    framing it, which is what the drive loop below already does. Object
    reference stays with the perception + Claude stack; see `hand_over`.

    It publishes whatever it counted. Silence was right while nothing was wired
    in, because a wrong integer scores the same 0 and looks in the log like a
    working responder — but scoring is exact-match 0 or 1, so once a responder
    exists, abstaining can only lose the point it might have won.
    """
    out = OUT_ROOT / time.strftime("%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(robot=bot, out=out, log=(out / "steps.jsonl").open("w"),
              backend="claude", model=MODEL,
              prompt_version=DEFAULT_PROMPT_VER, deadline=T0 + BUDGET_S)
    ctx.note_settings()
    try:
        t = answer_numerical(ctx, question)
    finally:
        ctx.close()

    pub = node.create_publisher(Int32, "/numerical_response", ANSWER_QOS)
    msg = Int32()
    msg.data = int(t["count"])
    # Repeated for the same reason the question is: a subscriber that came up
    # late still hears it, and there is no second chance at this message.
    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.2)
    node.get_logger().info(
        f"published {msg.data} on /numerical_response "
        f"({t['clusters']} placed + {t['unplaced']} unplaced over "
        f"{t['looks']} looks, per view {t['per_view']}, "
        f"{'a view was sufficient' if t['sufficient'] else 'never sufficient'}) "
        f"in {ctx.calls} calls (${ctx.calls * COST_PER_CALL:.2f}), "
        f"{time.time() - T0:.0f}s of {BUDGET_S:.0f}   log: {out}/steps.jsonl")
    (out / "answer.json").write_text(json.dumps(t, indent=1, default=str))
    return 0


def hand_over(node, question: str, kind: str) -> None:
    """Give the question to a perception-backed pipeline and get out of the way.

    Object-reference questions are answered by the other half of the team's
    stack: a YOLO-World + SAM 2.1 sidecar building a scene graph while the robot
    moves, then a model answering from it. That is a different process tree with
    a different Python environment, so this is a hand-off, not a function call.

    Which launch is `REFERENCE_LAUNCH` (see above); there is one, and it is the
    question-directed `nav_task1` drive with Claude answering. Re-launching
    rather than re-spawning its two processes by hand means their startup
    semantics — ordering, logging, shutdown — are exactly what was tested, and
    nothing here has to know how a uvicorn sidecar wants to be started.

    Two things make the hand-off safe:

    - **`exec`, not `Popen`.** Their explorer publishes waypoints from the
      moment it starts, so it must never run at the same time as our drive
      loop; replacing this process guarantees only one of the two is ever
      alive. Everything after this call is unreachable.
    - **The question is re-delivered.** We consumed a message they now need,
      but the evaluation node "publishes a single question each startup ... at
      a rate of 1Hz" (README §Evaluation), so their subscriber receives it
      within a second of coming up. Nothing has to be forwarded.

    The cost is the few seconds we spent waiting for the question and
    classifying it, out of 600.
    """
    argv = ["ros2", "launch", *REFERENCE_LAUNCH.split()]
    node.get_logger().info(
        f"{kind} question — handing over to {' '.join(argv)} "
        f"({time.time() - T0:.0f}s into the budget): {question!r}")
    # Drop our subscriptions and publisher before the image is replaced, so the
    # graph does not briefly show two waypoint publishers during the swap.
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:                             # noqa: BLE001
        pass
    os.execvp(argv[0], argv)


def run_instruction(node, bot: RobotNode, question: str) -> int:
    """Decompose the sentence and drive the clauses in order."""
    out = OUT_ROOT / time.strftime("%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "steps.jsonl").open("w")

    # No decompose cache is *read*: the test scenes are held out, so every
    # question here is one we have never seen and a lookup could only miss.
    # The file is still pointed somewhere writable, because what lands in it is
    # the model's raw reply — the only record of how the sentence got split if
    # the legs come out wrong. Rebound on the module rather than through
    # XIAO_HEI_DECOMPOSE_CACHE because that is read at import, which happened
    # before this run had a directory.
    decompose_mod.CACHE = out / "decompose.json"
    plan, from_model = decompose(question, cache={})
    node.get_logger().info(
        f"plan ({'model' if from_model else 'REGEX FALLBACK'}):")
    for i, c in enumerate(steps(plan), 1):
        node.get_logger().info(f"  step {i}  {c}")
    for c in keepouts(plan):
        node.get_logger().info(f"  always  {c}")

    pre = bot.preflight()
    node.get_logger().info(f"preflight: {json.dumps(pre)}")
    if not pre.get("ok"):
        # Never fatal. See RobotNode.preflight.
        node.get_logger().warning(f"!! {pre.get('why')}")

    ctx = Ctx(robot=bot, out=out, log=log, backend="claude", model=MODEL,
              prompt_version=DEFAULT_PROMPT_VER,
              deadline=T0 + BUDGET_S)
    ctx.note_settings()
    results = execute(ctx, question, plan, goto_steps=GOTO_STEPS)
    ctx.leg_deadline = None
    ctx.close()

    done = sum(r["ok"] for r in results)
    for r in results:
        node.get_logger().info(
            f"  {'OK  ' if r['ok'] else 'FAIL'} {r['k']}. {r['clause']} "
            f"— {r['why']}")
    node.get_logger().info(
        f"{done}/{len(results)} constraints satisfied in {ctx.calls} calls "
        f"(${ctx.calls * COST_PER_CALL:.2f}), {time.time() - T0:.0f}s of "
        f"{BUDGET_S:.0f}   log: {out}/steps.jsonl")
    (out / "plan.json").write_text(json.dumps(
        {"question": question, "from_model": from_model,
         "plan": [str(c) for c in plan], "results": results}, indent=1))
    return 0 if done == len(results) else 1


def selftest(node, bot: RobotNode) -> int:
    """Prove the plumbing without spending a cent.

    Receive-question through publish-waypoint through the-vehicle-moved is the
    part that is new in a submission; the grounding loop above it has been
    driven for weeks. If this does not move the robot, nothing else matters, so
    it is worth being able to run on its own — no API key, no model, no faces.
    """
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.05)
        if bot.pose is not None:
            break
    if bot.pose is None:
        node.get_logger().error("no /state_estimation — is the system up?")
        return 1
    node.get_logger().info(f"preflight: {json.dumps(bot.preflight())}")

    eq, scan, terrain, pose = bot.capture()
    node.get_logger().info(
        f"capture: image {eq.shape} {eq.dtype}, scan {scan.shape}, "
        f"terrain {terrain.shape}, pose {pose['position']}")

    # Two metres straight ahead in the map frame. Deliberately not through the
    # planner's obstacle logic — if there is a wall there, "settled" is a pass
    # too. What is being tested is that the waypoint arrives at all.
    q = pose["orientation"]
    yaw = float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]),
                           1 - 2 * (q[1] * q[1] + q[2] * q[2])))
    x = pose["position"][0] + 2.0 * np.cos(yaw)
    y = pose["position"][1] + 2.0 * np.sin(yaw)
    node.get_logger().info(f"driving to ({x:.2f}, {y:.2f})")
    res = bot.drive_to(x, y, timeout=30.0)
    node.get_logger().info(f"drive: {json.dumps(res)[:400]}")
    if not res.get("ok"):
        node.get_logger().error(f"selftest FAILED: {res.get('why')}")
        return 1
    node.get_logger().info(
        f"selftest OK — moved {res['moved_m']:.2f} m, {res['why']}")
    return 0


def idle(node) -> None:
    """Spin until the container is stopped."""
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="capture once and drive 2 m ahead; no API key needed")
    ap.add_argument("--question", default=None,
                    help="skip /challenge_question and use this instead")
    ap.add_argument("--question-timeout", type=float, default=300.0)
    args, _ = ap.parse_known_args()          # ros2 launch appends its own args

    rclpy.init()
    node = rclpy.create_node("xiao_hei_vln")
    bot = RobotNode(node)
    node.get_logger().info(
        f"xiao_hei_vln up — budget {BUDGET_S:.0f}s from process start, "
        f"model {MODEL}, image topic {os.environ.get('XIAO_HEI_IMAGE_TOPIC', '/camera/image')}")
    try:
        if args.selftest:
            return selftest(node, bot)

        # Fail here rather than three minutes in, after the robot has driven
        # somewhere on a plan we cannot follow up. The key is passed in by the
        # graders; it is never in the repo or the image.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            node.get_logger().error(
                "ANTHROPIC_API_KEY is not set — this module answers by calling "
                "the Claude API and can do nothing without it. Pass it with "
                "`docker run -e ANTHROPIC_API_KEY=...`; see ai_module/README.md.")
            return 2

        question = args.question
        if question is None:
            node.get_logger().info("waiting for /challenge_question ...")
            question = wait_for_question(node, args.question_timeout)
        if not question:
            node.get_logger().error(
                f"no question on /challenge_question after "
                f"{args.question_timeout:.0f}s — is the evaluation node up?")
            return 3
        node.get_logger().info(f"question: {question!r}")

        kind = classify(question)
        node.get_logger().info(f"type: {kind}")
        # The hand-off never returns — see `hand_over`. It is outside the
        # try/except below on purpose: there is nothing left of this process to
        # recover into, and swallowing an exec failure would leave a node that
        # answers nothing while looking alive.
        #
        # Numerical used to come through here too. It answers in-process now,
        # so only object reference leaves.
        if kind == REFERENCE:
            hand_over(node, question, kind)
        try:
            rc = (handle_numerical(node, bot, question) if kind == NUMERICAL
                  else run_instruction(node, bot, question))
        except Exception:                         # noqa: BLE001 — see below
            # Scoring is per constraint with partial credit on the trajectory
            # actually driven, so the waypoints already published still count.
            # Letting the exception out would take the node down with them and
            # turn a partial score into a zero — which is what happened when a
            # TypeError reached the top of a live run two legs in. Log it whole,
            # then idle like any other finished question.
            import traceback
            node.get_logger().error(
                f"the run raised, {time.time() - T0:.0f}s in; the trajectory "
                f"driven so far stands:\n{traceback.format_exc()}")
            rc = 4

        # Park before idling. `drive_to` publishes a waypoint once and never
        # retracts it, so a module that stops thinking leaves the local planner
        # still driving at its last goal — and oscillating in place when that
        # goal was unreachable, which is what a drive timeout means. Those
        # metres are still being scored: instruction following is marked on the
        # driven trajectory, and a vehicle nobody is steering can wander into a
        # keep-out the run had respected.
        parked = bot.stop()
        if not parked.get("ok"):
            node.get_logger().warning(
                f"could not park the vehicle: {parked.get('why')} — it may keep "
                f"driving at its last waypoint")

        # Stay up once the question has been answered. The container is the
        # graders' to stop, and a process that exits the moment it finishes is
        # indistinguishable in their logs from one that crashed. Only this path
        # idles — a missing key or a question that never arrived should fall
        # straight through and report.
        node.get_logger().info("answered — idling until stopped")
        idle(node)
        return rc
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
