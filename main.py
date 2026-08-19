"""C2-A 快速盘中扫描入口。"""

import sys
from pathlib import Path


def _entrypoint() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from robot_quant.c2a_fast import main

    main()


if __name__ == "__main__":
    _entrypoint()
