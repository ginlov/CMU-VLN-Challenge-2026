cmake_minimum_required(VERSION 3.5)
project(dummy_vlm)

find_package(ament_cmake REQUIRED)

# Launch-and-entrypoint package. Nothing is compiled: the module is Python.
#
# The package name, the launch file name and the executable are all kept from
# the challenge template so that the evaluation's documented
# `ros2 launch dummy_vlm dummy_vlm.launch` still starts us.
#
# `dummyVLM` is the router (see vlm/challenge_node.py). It reads the question,
# decides which of the team's two stacks should answer it, and either drives
# the instruction itself or hands the process over to one of the object-
# reference launches — scene_gemini.launch (default) or scene_claude.launch.
# Both are installed by the DIRECTORY rule below; neither is named here.
#
# Note there is no `data` directory to install any more — the canned waypoints
# and object list moved into the xiao_hei_vln package when the C++ node was
# retired, and naming it here would fail the build.
install(PROGRAMS src/dummyVLM.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dummyVLM
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
