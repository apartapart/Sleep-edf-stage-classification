"""scripts/ 에서 src 패키지를 import 할 수 있게 repo root 를 sys.path 에 추가."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
