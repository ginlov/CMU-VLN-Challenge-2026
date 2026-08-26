#!/usr/bin/env python3
"""What `ros2 run vlm vlmNode` starts — Team Xiao Hei's AI module.

Not one line of the challenge template's C++ remains, so the package is named
for what it is. The template's name is not gone, though: the evaluation is
documented as running `ros2 launch dummy_vlm dummy_vlm.launch`, that command
is written down outside `ai_module/` where a submission may not edit it, and
the startup script the graders actually use is not in this repository at all.
A launch-only `dummy_vlm` package therefore sits beside this one and forwards
here, so both names start the same module and nothing depends on which one the
graders type.

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
