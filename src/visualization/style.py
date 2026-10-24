"""공통 스타일 (dataviz 레퍼런스 팔레트).

- 범주형 6색 고정 순서 = 수면 단계 W,N1,N2,N3,N4,R
- 순차형 = 단일 hue(blue) light->dark : confusion matrix
- 텍스트는 텍스트 토큰, 시리즈 색 사용 안 함
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # slots 1-6
MISMATCH = "#e34948"  # slot 8 red: 불일치 표시 전용
STAGE_COLOR = dict(zip(["W", "N1", "N2", "N3", "N4", "R"], CAT))
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("ref_blues", ["#fcfcfb"] + SEQ_STEPS)


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
    })
