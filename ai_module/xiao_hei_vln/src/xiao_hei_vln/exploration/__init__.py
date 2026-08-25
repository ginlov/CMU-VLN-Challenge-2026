"""Online exploration strategies for the CMU VLN challenge.

The module is organised around a single protocol (``ExplorationStrategy``):

    update(snapshot: VLMInput) -> Waypoint | None
    is_complete() -> bool
    reset() -> None

Any class implementing these three methods is a valid exploration strategy
and can be dropped into the main tick loop.  Register new strategies in
``_build_explorer`` (``app/main.py``) and select them at runtime with the
``XIAO_HEI_EXPLORATION_STRATEGY`` environment variable.

Currently implemented
---------------------
frontier (default)
    ``FrontierExplorer`` — builds a 2-D occupancy grid from
    ``terrain_map_ext`` snapshots and navigates toward the largest nearby
    frontier cluster.  Stops when ``max_waypoints`` have been visited.

nav_vlm
    ``NavVLMExplorer`` — asks a vision-language model for the next waypoint on
    each reach / cannot-reach event, snapping the pick to a grid-reachable free
    cell.  Select with ``XIAO_HEI_EXPLORATION_STRATEGY=nav_vlm`` (needs an
    Anthropic key).

nav_task1
    ``NavTask1Explorer`` — the question-directed variant of ``nav_vlm``: it
    drives toward the object named in an OBJECT_REFERENCE question (feeding the
    model the question plus the objects detected so far) and completes when the
    model declares arrival, at which point the ``scene_claude`` responder
    answers from the built scene graph.  Select with
    ``XIAO_HEI_EXPLORATION_STRATEGY=nav_task1``.

Visualisation
-------------
save_exploration_plot(visited_waypoints, grid, output_path)
    Saves a PNG debug plot showing the explored map and the robot path.
    Only supported by strategies that expose ``get_visited_waypoints()``
    and ``get_grid()``.  Requires matplotlib (install the ``[exploration]``
    extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from xiao_hei_vln.messages.inputs import VLMInput
    from xiao_hei_vln.messages.outputs import Waypoint


@runtime_checkable
class ExplorationStrategy(Protocol):
    """Duck-type protocol every exploration algorithm must satisfy."""

    def update(self, snapshot: "VLMInput") -> "Waypoint | None": ...
    def is_complete(self) -> bool: ...
    def reset(self) -> None: ...


from xiao_hei_vln.exploration._frontier import FrontierExplorer
from xiao_hei_vln.exploration._nav_task1 import NavTask1Explorer
from xiao_hei_vln.exploration._nav_vlm import NavVLMExplorer
from xiao_hei_vln.exploration._visualize import save_exploration_plot

__all__ = [
    "ExplorationStrategy",
    "FrontierExplorer",
    "NavTask1Explorer",
    "NavVLMExplorer",
    "save_exploration_plot",
]
