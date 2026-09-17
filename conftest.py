# -*- coding: utf-8 -*-
"""
让 pytest 能够直接 `import core` / `import tokenrouter`。

仓库根目录下同时存在 simulation/、modules/（算力侧）和 core/、tokenrouter/
（本次场景外扩新增），所有包都以仓库根为导入起点。与其在每个测试文件里重复
sys.path 处理，统一在这里做一次。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
