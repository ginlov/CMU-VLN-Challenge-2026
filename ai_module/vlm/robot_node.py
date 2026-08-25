#!/usr/bin/env python3
"""In-process replacement for `approach_loop.Robot`.

Development drives the loop from a laptop: `Robot` shells out over ssh and
`docker exec` into the system container, runs `robot_io.py` there, and copies
`/tmp/loop_*.npy` back. A submission cannot do that — the AI module *is* a node
inside `iros2026_ai_module`, started by the organizers, with no shell on the
other side and nothing to ssh to.

The seam that makes this a small change is that `Robot` is four methods used at
six call sites, and `capture()` already hands back exactly what the loop wants:
`(equirect BGR, scan Nx4 in the map frame, terrain Nx4, pose dict)`. So this
class implements the same four methods against live subscriptions and
`approach_loop`, `execute_plan`, `vlm_approach` and the rest are untouched.

`robot_io.py`'s `Capture` and `Driver` merge into one node here, because a
persistent node cannot create and destroy a subscription per call the way a
one-shot process could. Everything else is carried over deliberately, including
the constants — see `robot_io` for what each of them is protecting against.

Single-threaded on purpose. Callbacks are only needed while capturing and while
driving; the ~35 s Claude call in between wants no ROS work done at all, and
BEST_EFFORT + depth=1 means nothing queues up behind it.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from robot_io import (ARRIVE_TOL_M, STALL_DEG, STALL_M,  # noqa: E402
                      STALL_S, TRACK_STEP_M)

# README §System Outputs lists five topics and says "these are the only ones
# allowed to be used during test time". `/camera/image` is the one named there;
# development used `/camera/image/compressed`, which is the same sensor but not
# on the list. The override exists so the two can be A/B'd against a scene we
# have logs for without a rebuild — the compressed path costs a JPEG round trip
# the raw one does not, and it is the only difference the model can see.
IMAGE_TOPIC = os.environ.get("XIAO_HEI_IMAGE_TOPIC", "/camera/image")

# RELIABLE + depth=5 makes DDS replay stale samples in arrival order whenever
# the subscriber falls behind, and this one falls behind by 35 s every step. The
# reader would then hand back a frame up to five periods old, and every bearing
# in the loop rides on the image/pose pairing. depth=1 keeps only the newest,
# which is the whole point of a snapshot. Measured on the other pipeline at
# 1.2 s of lag on /camera/image before it was changed.
SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)


def _pc2_to_xyzi(msg) -> np.ndarray:
    """PointCloud2 -> (N, 4) float32, read by declared offset.

    Do not infer columns from `point_step`: PCL pads PointXYZI out to 32 bytes
    on `/terrain_map`, so intensity sits at offset 16 with three dead floats
    after it, and a contiguous reshape lands on padding. On terrain, intensity
    is height above the local ground — what `obstacleHeightThre` is compared
    against in `waypointConverter`, and so the only column that matters.
    """
    n, step = int(msg.width) * int(msg.height), int(msg.point_step)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, step)
    out = np.zeros((n, 4), dtype=np.float32)
    off = {f.name: int(f.offset) for f in msg.fields}
    for i, name in enumerate(("x", "y", "z", "intensity")):
        if name in off:
            out[:, i] = np.frombuffer(
                buf[:, off[name]:off[name] + 4].tobytes(), dtype=np.float32)
    return out


def _image_to_bgr(msg) -> np.ndarray:
    """A raw ROS Image as the BGR array every downstream stage expects.

    `faces_of` re-encodes to JPEG anyway, so this is the only place in the
    system where the switch from `/camera/image/compressed` to `/camera/image`
    is visible. Getting the channel order wrong here would not crash anything:
    it would silently swap red and blue in every image the model is shown, and
    the questions name colours.
    """
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = str(msg.encoding).lower()
    a = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, step)
    if enc in ("bgr8", "rgb8"):
        img = a[:, :w * 3].reshape(h, w, 3)
        return img if enc == "bgr8" else img[:, :, ::-1].copy()
    if enc in ("bgra8", "rgba8"):
        img = a[:, :w * 4].reshape(h, w, 4)[:, :, :3]
        return img.copy() if enc == "bgra8" else img[:, :, ::-1].copy()
    raise SystemExit(
        f"/camera/image encoding {enc!r} is not one this node knows how to "
        f"read as BGR. Add it to _image_to_bgr rather than guessing — a wrong "
        f"channel order does not crash, it just shows the model wrong colours.")


class RobotNode:
    """Same four methods as `approach_loop.Robot`, against live topics."""

    def __init__(self, node) -> None:
        self.node = node
        self._compressed = IMAGE_TOPIC.endswith("/compressed")

        # Snapshot slots, filled only while `capture()` is collecting.
        self._snapping = False
        self._img: np.ndarray | None = None
        self._scan: np.ndarray | None = None
        self._terrain: np.ndarray | None = None
        self._snap_pose: dict | None = None

        # Always-current pose, for driving.
        self.pose: list[float] | None = None
        self._yaw: float | None = None
        # `/terrain_map_ext` reaches 20 m against `/terrain_map`'s 5 m and is on
        # the allowed list. It is collected but not yet fed to `ConverterModel`,
        # which still models the 5 m map — swapping it is a behaviour change
        # that wants its own regression run.
        self.terrain_ext: np.ndarray | None = None

        self._driving = False
        self.goal = (0.0, 0.0)
        self.start: list[float] | None = None
        self.track: list[list[float]] = []
        self._last_move = 0.0
        self._last_xy: tuple[float, float] | None = None
        self._last_yaw: float | None = None

        img_type = CompressedImage if self._compressed else Image
        node.create_subscription(img_type, IMAGE_TOPIC, self._on_img, SENSOR_QOS)
        node.create_subscription(PointCloud2, "/registered_scan",
                                 self._on_scan, SENSOR_QOS)
        node.create_subscription(PointCloud2, "/terrain_map",
                                 self._on_terrain, SENSOR_QOS)
        node.create_subscription(PointCloud2, "/terrain_map_ext",
                                 self._on_terrain_ext, SENSOR_QOS)
        node.create_subscription(Odometry, "/state_estimation",
                                 self._on_pose, SENSOR_QOS)
        # `/way_point_reached` is NOT subscribed. It answers "did you reach the
        # waypoint the converter snapped ours to", it is not on the README's
        # allowed-topic list, and nothing in the loop branches on it — arrival
        # is decided from `/state_estimation` alone. See `robot_io.Driver`.
        self.pub = node.create_publisher(Pose2D, "/way_point_with_heading", 5)

    # ---- callbacks ---------------------------------------------------------

    def _on_img(self, m) -> None:
        if not self._snapping or self._img is not None:
            return
        import cv2
        self._img = (cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8),
                                  cv2.IMREAD_COLOR) if self._compressed
                     else _image_to_bgr(m))
        # Anything already held was sampled before this frame. Drop it and take
        # the next pose, which lands within a pose period (5-10 ms) rather than
        # the camera's (107 ms). The lift maps pixels through this pose; pairing
        # a frame with a pose from before it was taken is an error on every
        # bearing in the step. See `robot_io.Capture.__doc__`.
        self._snap_pose = None

    def _on_scan(self, m) -> None:
        if self._snapping and self._scan is None:
            self._scan = _pc2_to_xyzi(m)

    def _on_terrain(self, m) -> None:
        if self._snapping and self._terrain is None:
            self._terrain = _pc2_to_xyzi(m)

    def _on_terrain_ext(self, m) -> None:
        self.terrain_ext = _pc2_to_xyzi(m)

    def _on_pose(self, m) -> None:
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = [p.x, p.y, p.z]
        self._yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                     1 - 2 * (q.y * q.y + q.z * q.z)))
        if self._snapping and self._img is not None and self._snap_pose is None:
            self._snap_pose = {"position": [p.x, p.y, p.z],
                               "orientation": [q.x, q.y, q.z, q.w]}
        if not self._driving:
            return
        if self.start is None:
            self.start = list(self.pose)
        # Subsampled by distance: /state_estimation runs at 100-200 Hz and a
        # crossing test needs shape, not sample rate. README §175 scores "the
        # actual trajectory followed by the robot", and `local_planner` picks an
        # arc from its own path library, so the straight line between two
        # waypoints is not where the vehicle went. A passage constraint can only
        # be checked against this.
        if not self.track or np.hypot(p.x - self.track[-1][0],
                                      p.y - self.track[-1][1]) > TRACK_STEP_M:
            self.track.append([round(p.x, 3), round(p.y, 3)])
        moved = (self._last_xy is None
                 or np.hypot(p.x - self._last_xy[0],
                             p.y - self._last_xy[1]) > STALL_M)
        # A turn counts as progress. The platform points itself at the waypoint
        # before it translates, so a goal behind the vehicle produces seconds of
        # pure rotation that a position-only test reads as refusal to move.
        turned = (self._last_yaw is None
                  or abs(np.rad2deg(np.arctan2(np.sin(self._yaw - self._last_yaw),
                                               np.cos(self._yaw - self._last_yaw))))
                  > STALL_DEG)
        if moved or turned:
            self._last_xy, self._last_yaw = (p.x, p.y), self._yaw
            self._last_move = time.time()

    # ---- the Robot interface ----------------------------------------------

    def push(self) -> None:
        """No-op. There is no bridge script to copy into another container."""

    def preflight(self) -> dict:
        """Who is publishing what, before the loop spends money finding out.

        `/joy` is the one that bites: without a heartbeat the local planner
        silently discards every waypoint, and the only symptom is a robot that
        never moves. A rival publisher on `/way_point_with_heading` is the
        other — the C++ `dummyVLM` will fight us for control if it was left
        running. Reported, never fatal: a missing count here is often just
        discovery not having settled, and refusing to start the question over
        that would forfeit it outright.
        """
        for _ in range(40):                       # let discovery settle
            rclpy.spin_once(self.node, timeout_sec=0.05)
        want = {"/state_estimation": "pub", "/registered_scan": "pub",
                IMAGE_TOPIC: "pub", "/terrain_map": "pub", "/joy": "pub",
                "/way_point_with_heading": "sub"}
        counts = {t: (self.node.count_publishers(t) if k == "pub"
                      else self.node.count_subscribers(t))
                  for t, k in want.items()}
        # Name them rather than count them: RViz always holds one, which is the
        # manual Waypoint tool and emits only when a human clicks, and we are
        # ourselves a publisher here. An autonomous third party is the problem.
        mine = self.node.get_name()
        rivals = [e.node_name for e in
                  self.node.get_publishers_info_by_topic("/way_point_with_heading")
                  if e.node_name not in ("rviz", mine)]
        missing = [t for t, c in counts.items() if c == 0]
        why = None
        if "/joy" in missing:
            why = ("nothing is publishing /joy — the local planner will ignore "
                   "every waypoint; is the base autonomy system up?")
        elif missing:
            why = f"no traffic on {missing}"
        elif rivals:
            why = (f"{rivals} also publish(es) /way_point_with_heading and will "
                   f"fight the loop")
        return {"ok": why is None, "why": why, "counts": counts,
                "rival_waypoint_publishers": rivals}

    def capture(self, timeout: float = 20.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """One frame of everything, with the pose taken *after* the image."""
        # Drain whatever the readers are holding from before this call, so the
        # snapshot cannot open with a frame that predates the drive we just
        # finished.
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.0)
        self._img = self._scan = self._terrain = self._snap_pose = None
        self._snapping = True
        deadline = time.time() + timeout
        try:
            while time.time() < deadline and not self._have_snap():
                rclpy.spin_once(self.node, timeout_sec=0.05)
        finally:
            self._snapping = False
        if not self._have_snap():
            missing = [k for k, v in (("image", self._img), ("scan", self._scan),
                                      ("terrain", self._terrain),
                                      ("pose", self._snap_pose)) if v is None]
            raise SystemExit(f"capture failed, missing {missing} — is the base "
                             f"autonomy system running?")
        return self._img, self._scan, self._terrain, self._snap_pose

    def _have_snap(self) -> bool:
        return all(v is not None for v in
                   (self._img, self._scan, self._terrain, self._snap_pose))

    def stop(self) -> dict:
        """Park where we stand, so the stack stops chasing the last waypoint.

        `drive_to` publishes once and returns; nothing retracts the goal, so the
        local planner holds it after this process has finished with the
        question and keeps driving at it. When the goal is unreachable -- the
        case a drive timeout reports -- the vehicle oscillates in place until
        the stack is restarted.

        That is not cosmetic here. Instruction following is scored on the
        driven trajectory, so a vehicle still moving after the run has ended is
        still writing to the thing being marked, and it can wander into a
        keep-out the run had respected. Publishing the current pose is the
        retraction: a goal the vehicle is already at, which the planner settles
        on instead of hunting.

        Never raises. It runs on the way out, and a question that answered
        correctly must not fail because the parking brake did.
        """
        try:
            if self.pose is None:
                for _ in range(120):
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                    if self.pose is not None:
                        break
            if self.pose is None:
                return {"ok": False, "why": "no /state_estimation"}
            self.goal = (float(self.pose[0]), float(self.pose[1]))
            sent = self._send()
            for _ in range(40):
                rclpy.spin_once(self.node, timeout_sec=0.05)
            return {"ok": bool(sent), "why": "parked" if sent else "no subscriber",
                    "goal": list(self.goal)}
        except Exception as e:                    # noqa: BLE001 -- see docstring
            return {"ok": False, "why": repr(e)}

    def drive_to(self, x: float, y: float, timeout: float) -> dict:
        """Publish one waypoint and watch the vehicle until it settles.

        Arrival is our own distance to the waypoint we asked for, plus whether
        the vehicle has stopped making progress. `waypoint_converter` snaps our
        waypoint into the traversable area, so the vehicle usually never reaches
        the literal XY and a pure distance test would hang; the stall test is
        what covers that, and it needs nothing but pose.
        """
        self.goal = (float(x), float(y))
        self.start = None
        self.track = []
        self._last_xy = self._last_yaw = None
        self._last_move = time.time()

        for _ in range(60):                       # a pose to leave from
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.pose is not None:
                break
        self._driving = True
        try:
            if not self._send():
                return {"ok": False, "why": "no subscriber on "
                                            "/way_point_with_heading — is the "
                                            "autonomy stack up?"}
            deadline, why = time.time() + timeout, "timeout"
            while time.time() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.05)
                gap = self._gap()
                if gap is not None and gap <= ARRIVE_TOL_M:
                    why = "arrived"
                    break
                # Either the stack put us as close as its obstacle clearance
                # allows, or the waypoint is unreachable. `dist_to_requested_m`
                # tells them apart downstream; both mean stop driving here.
                if self._last_xy is not None and \
                        time.time() - self._last_move > STALL_S:
                    why = "settled"
                    break
        finally:
            self._driving = False

        moved = (float(np.linalg.norm(np.asarray(self.pose[:2])
                                      - np.asarray(self.start[:2])))
                 if self.pose and self.start else None)
        return {"ok": why in ("arrived", "settled"), "why": why,
                "goal": list(self.goal), "pose": self.pose, "moved_m": moved,
                "track": self.track, "dist_to_requested_m": self._gap()}

    def _gap(self) -> float | None:
        if self.pose is None:
            return None
        return float(np.hypot(self.pose[0] - self.goal[0],
                              self.pose[1] - self.goal[1]))

    def _send(self) -> bool:
        """Publish once, after a subscriber exists to receive it."""
        for _ in range(100):                      # ~5 s of discovery
            if self.pub.get_subscription_count() > 0:
                m = Pose2D()
                m.x, m.y, m.theta = self.goal[0], self.goal[1], 0.0
                self.pub.publish(m)
                return True
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return False
