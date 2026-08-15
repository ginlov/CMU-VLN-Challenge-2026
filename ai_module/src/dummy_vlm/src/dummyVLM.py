#!/usr/bin/env python3
"""What `ros2 run dummy_vlm dummyVLM` starts — Team Xiao Hei's AI module.

Everything about this package's *naming* is deliberate. The package is still
`dummy_vlm`, the launch file is still `dummy_vlm.launch`, and this executable
is still `dummyVLM`, even though not one line of the original C++ remains. The
challenge README says to integrate a model by modifying "the system startup
script", and that script is not in this repository — so if it invokes
`ros2 launch dummy_vlm dummy_vlm.launch`, as the docker README's instructions
do, we are still what starts. Renaming the package would be tidier and would
risk the module never being launched at all.

The implementation lives outside the ament package, at `/opt/xiao_hei/vlm`, so
that the drive loop stays a directory of plain, individually runnable Python
files rather than something that has to be reinstalled to be edited. Point
`XIAO_HEI_VLM_DIR` somewhere else to run a working copy — see
`ai_module/docker/run_dev.sh`, which bind-mounts one.
"""

import os
import sys

sys.path.insert(0, os.environ.get("XIAO_HEI_VLM_DIR", "/opt/xiao_hei/vlm"))

from challenge_node import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
