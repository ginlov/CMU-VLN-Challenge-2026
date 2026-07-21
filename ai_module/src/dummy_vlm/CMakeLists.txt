cmake_minimum_required(VERSION 3.5)
project(dummy_vlm)

# Launch-only package. The actual AI module is a pip-installed Python
# package (xiao_hei_vln) plus a perception sidecar, both started by
# launch/dummy_vlm.launch. No C++ node is built here — we only install the
# launch directory so `ros2 launch dummy_vlm dummy_vlm.launch` resolves.
find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
