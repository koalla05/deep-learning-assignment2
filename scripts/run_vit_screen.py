"""Run the frozen transformer screen (notebook 22) from a terminal: python scripts/run_vit_screen.py [keys...]"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2.vit_screen import SCREEN_ORDER, format_table, run_screen  # noqa: E402

warnings.filterwarnings("ignore")
keys = tuple(sys.argv[1:]) or SCREEN_ORDER
rows = run_screen(Path(__file__).resolve().parents[1], keys=keys, device="mps", half=True)
print(format_table(rows))
