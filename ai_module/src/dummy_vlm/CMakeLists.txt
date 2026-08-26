cmake_minimum_required(VERSION 3.5)
project(dummy_vlm)

find_package(ament_cmake REQUIRED)

# Launch file only — the executable lives in the `vlm` package.
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
