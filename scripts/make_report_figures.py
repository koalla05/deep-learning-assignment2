"""Draw the architecture, training and ensemble diagrams used in README.md.

Plain SVG, no dependencies. Run from anywhere: `python scripts/make_report_figures.py`.
Every number in the figures is taken from timm (layer shapes at 384 px) or from the run records.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

TEXT, MUTED, LINE = "#1f2937", "#4b5563", "#6b7280"
STYLE = {  # (fill, stroke)
    "io": ("#ffffff", "#6b7280"),
    "bb": ("#dbeafe", "#2563eb"),
    "head": ("#ede9fe", "#7c3aed"),
    "block": ("#dcfce7", "#16a34a"),
    "frozen": ("#f3f4f6", "#9ca3af"),
    "conv": ("#dbeafe", "#2563eb"),
    "norm": ("#fef3c7", "#d97706"),
    "act": ("#dcfce7", "#16a34a"),
    "se": ("#fce7f3", "#db2777"),
    "add": ("#ffffff", "#374151"),
    "note": ("#ffffff", "#9ca3af"),
}
MEMBER = {  # ensemble member colours: (fill, stroke)
    "Exp1": ("#dbeafe", "#2563eb"),
    "Exp2": ("#e0f2fe", "#0284c7"),
    "Exp3": ("#ffedd5", "#ea580c"),
    "Exp5": ("#dcfce7", "#16a34a"),
}
CHAR = 6.1  # average glyph width at 11 px, for sizing boxes


def svg(width: float, height: float, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="Helvetica, Arial, sans-serif">\n'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>\n'
        '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="#f3f4f6"/><line x1="0" y1="0" x2="0" y2="6" stroke="#d1d5db" stroke-width="2"/></pattern></defs>\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def text(x, y, s, size=11, weight="normal", fill=MUTED, anchor="start", italic=False) -> str:
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{style}>{escape(s)}</text>')


def rect(x, y, w, h, kind, dash=False, rx=6, fill=None) -> str:
    f, s = STYLE[kind]
    dashed = ' stroke-dasharray="5 3"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill or f}" stroke="{s}" stroke-width="1.4"{dashed}/>'


def arrow(x1, y1, x2, y2, dash=False) -> str:
    dashed = ' stroke-dasharray="4 3"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{LINE}" stroke-width="1.4" marker-end="url(#arrow)"{dashed}/>'


def box(x, y, w, h, kind, title, lines=(), dash=False, fill=None, title_size=12) -> list[str]:
    out = [rect(x, y, w, h, kind, dash, fill=fill)]
    rows = [title, *lines]
    top = y + h / 2 - (len(rows) - 1) * 14 / 2 + 4
    for i, row in enumerate(rows):
        out.append(text(x + w / 2, top + 14 * i, row, size=title_size if i == 0 else 11,
                        weight="bold" if i == 0 else "normal", fill=TEXT if i == 0 else MUTED, anchor="middle"))
    return out


def width_for(title, lines, minimum=70) -> float:
    return max(minimum, max(len(s) for s in (title, *lines)) * CHAR + 18)


# --------------------------------------------------------------------------- Figure 1

FAMILIES = [
    dict(
        name="ConvNeXt-base", used="Exp0, Exp1, Exp2",
        facts="87.6M parameters · LayerNorm + GELU · pretrained ImageNet-22k → 1k at 384 px",
        pipeline=[
            ("io", "Input", ["3 × 384²"]),
            ("bb", "Stem", ["4×4 conv, s4", "128 × 96²"]),
            ("bb", "Stage 1", ["3 blocks", "128 × 96²"]),
            ("bb", "Stage 2", ["3 blocks", "256 × 48²"]),
            ("bb", "Stage 3", ["27 blocks", "512 × 24²"]),
            ("bb", "Stage 4", ["3 blocks", "1024 × 12²"]),
            ("head", "Head", ["avg pool, LayerNorm", "Linear 1024 → 5"]),
        ],
        block_name="ConvNeXt block",
        block=[("conv", "7×7 depthwise conv"), ("norm", "LayerNorm"), ("conv", "1×1 conv C → 4C"),
               ("act", "GELU"), ("conv", "1×1 conv 4C → C")],
        note="Between stages: LayerNorm + 2×2 conv with stride 2 halves the resolution and doubles the channels.",
    ),
    dict(
        name="EfficientNetV2-S", used="Exp3",
        facts="20.2M parameters · BatchNorm + SiLU · pretrained ImageNet-21k → 1k",
        pipeline=[
            ("io", "Input", ["3 × 384²"]),
            ("bb", "Stem", ["3×3 conv, s2", "24 × 192²"]),
            ("bb", "Stage 1", ["2× conv", "24 × 192²"]),
            ("bb", "Stage 2", ["4× Fused", "48 × 96²"]),
            ("bb", "Stage 3", ["4× Fused", "64 × 48²"]),
            ("bb", "Stage 4", ["6× MBConv", "128 × 24²"]),
            ("bb", "Stage 5", ["9× MBConv", "160 × 24²"]),
            ("bb", "Stage 6", ["15× MBConv", "256 × 12²"]),
            ("head", "Head", ["1×1 conv → 1280", "avg pool", "Linear 1280 → 5"]),
        ],
        block_name="MBConv block",
        block=[("conv", "1×1 conv, expand ×4/×6"), ("norm", "BN + SiLU"), ("conv", "3×3 depthwise conv"),
               ("norm", "BN + SiLU"), ("se", "Squeeze-Excite"), ("conv", "1×1 conv, project"), ("norm", "BN")],
        note="Fused-MBConv (stages 2–3) replaces the 1×1 expand + 3×3 depthwise pair with one full 3×3 conv and has no SE.",
    ),
    dict(
        name="BiT ResNetV2-50x1", used="Exp5",
        facts="23.5M parameters · GroupNorm + weight-standardized conv · pretrained ImageNet-21k → 1k at 448 px",
        pipeline=[
            ("io", "Input", ["3 × 384²"]),
            ("bb", "Stem", ["7×7 conv, s2", "+ max pool, s2", "64 × 96²"]),
            ("bb", "Stage 1", ["3 bottlenecks", "256 × 96²"]),
            ("bb", "Stage 2", ["4 bottlenecks", "512 × 48²"]),
            ("bb", "Stage 3", ["6 bottlenecks", "1024 × 24²"]),
            ("bb", "Stage 4", ["3 bottlenecks", "2048 × 12²"]),
            ("head", "Head", ["GN + ReLU, avg pool", "1×1 conv 2048 → 5"]),
        ],
        block_name="Pre-activation bottleneck",
        block=[("norm", "GN + ReLU"), ("conv", "1×1 conv ↓"), ("norm", "GN + ReLU"), ("conv", "3×3 conv"),
               ("norm", "GN + ReLU"), ("conv", "1×1 conv ↑ ×4")],
        note="Normalization comes before each conv (pre-activation). GroupNorm does not depend on the batch, so no batch statistics are kept.",
    ),
]


def pipeline_row(x0, y, items, gap=16, h=74) -> tuple[list[str], float]:
    out, x = [], x0
    for i, (kind, title, lines) in enumerate(items):
        w = width_for(title, lines)
        out += box(x, y, w, h, kind, title, lines)
        x += w
        if i < len(items) - 1:
            out.append(arrow(x + 2, y + h / 2, x + gap - 2, y + h / 2))
            x += gap
    return out, x


def chain_row(x0, y, items, h=26, gap=14, skip=True) -> tuple[list[str], float]:
    out, x = [], x0
    if skip:
        out.append(arrow(x, y + h / 2, x + gap - 2, y + h / 2))
        start = x
        x += gap
    for kind, label in items:
        w = len(label) * CHAR + 18
        out.append(rect(x, y, w, h, kind, rx=13))
        out.append(text(x + w / 2, y + h / 2 + 4, label, size=11, fill=TEXT, anchor="middle"))
        x += w
        out.append(arrow(x + 2, y + h / 2, x + gap - 2, y + h / 2))
        x += gap
    if skip:  # residual addition, with the identity path drawn above the chain
        cx = x + 11
        out.append(f'<circle cx="{cx:.1f}" cy="{y + h / 2:.1f}" r="11" fill="#ffffff" stroke="#374151" stroke-width="1.4"/>')
        out.append(text(cx, y + h / 2 + 5, "+", size=15, weight="bold", fill=TEXT, anchor="middle"))
        top = y - 14
        out.append(f'<path d="M{start + 4:.1f},{y + h / 2:.1f} L{start + 4:.1f},{top:.1f} L{cx:.1f},{top:.1f} L{cx:.1f},{y + h / 2 - 13:.1f}" '
                    f'fill="none" stroke="{LINE}" stroke-width="1.4" marker-end="url(#arrow)"/>')
        out.append(text((start + cx) / 2, top - 4, "identity (skip connection)", size=10, anchor="middle", italic=True))
        x = cx + 11
    return out, x


def figure_backbones() -> str:
    body, y, widest, separators = [], 30, 0, []
    body.append(text(20, 22, "Three pretrained backbones (shapes at our 384 px input: channels × height²)", size=15, weight="bold", fill=TEXT))
    legend = [("bb", "backbone (pretrained)"), ("head", "head (new, 5 outputs)"), ("conv", "conv"), ("norm", "normalization"),
              ("act", "activation"), ("se", "squeeze-excite")]
    lx = 20
    for kind, label in legend:
        body.append(rect(lx, y + 4, 14, 14, kind, rx=3))
        body.append(text(lx + 20, y + 15, label))
        lx += 20 + len(label) * CHAR + 22
    y += 34
    for family in FAMILIES:
        body.append(text(20, y + 16, family["name"], size=14, weight="bold", fill=TEXT))
        body.append(text(20 + len(family["name"]) * 8.2 + 10, y + 16, f'used in {family["used"]}  ·  {family["facts"]}'))
        row, right = pipeline_row(20, y + 28, family["pipeline"])
        body += row
        widest = max(widest, right)
        body.append(text(20, y + 152, family["block_name"] + ":", size=11, weight="bold", fill=TEXT))
        chain, right = chain_row(20 + len(family["block_name"]) * 6.6 + 14, y + 134, family["block"])
        body += chain
        widest = max(widest, right)
        body.append(text(20, y + 186, family["note"], italic=True))
        y += 214
        separators.append(y - 12)
    body += [f'<line x1="20" y1="{s}" x2="{widest:.0f}" y2="{s}" stroke="#e5e7eb"/>' for s in separators]
    body.append(text(20, y + 8, "All three end with a 12 × 12 feature map (stride 32). The head pools it to one vector and maps it to the 5 label logits."))
    return svg(widest + 20, y + 24, body)


# --------------------------------------------------------------------------- Figure 2

def segment(x, y, w, h, kind, title, sub, dash=False) -> list[str]:
    fill = "url(#hatch)" if kind == "frozen" else None
    return box(x, y, w, h, kind, title, [sub], dash=dash, fill=fill, title_size=11)


def figure_training() -> str:
    P1, P2, NOTES = 210, 520, 880
    BB, BLK, HD, H = 170, 74, 84, 40
    rows = [
        dict(name="Exp0", model="ConvNeXt-base",
             p1=[("frozen", "Backbone", "frozen", BB), ("head", "Head", "LR 1e-3", HD)], p1_note="10 epochs, constant LR",
             p2=None, notes=["head-only baseline", "LB 0.93365"]),
        dict(name="Exp1", model="ConvNeXt-base", p1_ref="starts from Exp0 head",
             p2=[("bb", "Backbone", "LR 1e-5", BB), ("head", "Head", "LR 1e-4", HD)], p2_note="early stop after 5 of 8 epochs",
             notes=["standard fine-tune", "LB 0.97003"]),
        dict(name="Exp2", model="ConvNeXt-base + block", p1_ref="starts from Exp0 head",
             p2=[("bb", "Backbone", "LR 1e-5", BB - BLK + 8), ("block", "New block", "LR 1e-5", BLK), ("head", "Head", "LR 1e-4", HD)],
             p2_note="8 epochs; block starts at zero", notes=["own building block", "LB 0.96043"]),
        dict(name="Exp3", model="EfficientNetV2-S",
             p1=[("frozen", "Backbone", "frozen", BB), ("head", "New head", "LR 1e-3", HD)], p1_note="8 epochs",
             p2=[("bb", "Backbone", "LR 1e-4, BN stats frozen", BB), ("head", "Head", "LR 1e-3", HD)], p2_note="8 epochs",
             notes=["own head warm-up", "LB 0.97015"]),
        dict(name="Exp5", model="BiT ResNetV2-50x1",
             p1=[("frozen", "Backbone", "frozen", BB), ("head", "New head", "LR 1e-3", HD)], p1_note="8 epochs",
             p2=[("bb", "Backbone", "LR 3e-5", BB), ("head", "Head", "LR 3e-4", HD)], p2_note="8 epochs",
             notes=["own head warm-up", "LB 0.94035"]),
    ]
    body = [text(20, 22, "What was trained in each single-model experiment (all on folds 1–4, selected on fold 0)", size=15, weight="bold", fill=TEXT)]
    legend = [("frozen", "frozen (no gradient)"), ("bb", "trainable backbone"), ("block", "new block (Exp2)"), ("head", "trainable head")]
    lx = 20
    for kind, label in legend:
        body.append(rect(lx, 34, 14, 14, kind, rx=3, fill="url(#hatch)" if kind == "frozen" else None))
        body.append(text(lx + 20, 45, label))
        lx += 20 + len(label) * CHAR + 22
    y = 80
    body.append(text(P1, y, "Phase 1: head warm-up", size=12, weight="bold", fill=TEXT))
    body.append(text(P2, y, "Phase 2: fine-tune the whole network", size=12, weight="bold", fill=TEXT))
    body.append(text(NOTES, y, "Role · public LB", size=12, weight="bold", fill=TEXT))
    y += 14
    for row in rows:
        body.append(text(20, y + 20, row["name"], size=14, weight="bold", fill=TEXT))
        body.append(text(20, y + 36, row["model"]))
        if "p1" in row:
            x = P1
            for kind, title, sub, w in row["p1"]:
                body += segment(x, y + 4, w, H, kind, title, sub)
                x += w + 4
            body.append(text(P1, y + H + 18, row["p1_note"], size=10, italic=True))
        else:
            body += box(P1, y + 4, BB + HD + 4, H, "note", row["p1_ref"], ["no own phase 1 (head trained in Exp0)"], dash=True, title_size=11)
        if row["p2"]:
            x = P2
            for kind, title, sub, w in row["p2"]:
                width = max(w, len(sub) * 5.6 + 12)
                body += segment(x, y + 4, width, H, kind, title, sub)
                x += width + 4
            body.append(text(P2, y + H + 18, row["p2_note"], size=10, italic=True))
        else:
            body.append(text(P2, y + 28, "—"))
        body.append(text(NOTES, y + 20, row["notes"][0], fill=TEXT))
        body.append(text(NOTES, y + 36, row["notes"][1], weight="bold", fill=TEXT))
        y += 78
    footer = [
        "\"Head\" is only the final Linear layer (1×1 conv for BiT). ConvNeXt's pooling LayerNorm and EfficientNet's 1×1 conv → 1280 belong to the backbone.",
        "Phase 2 for all: AdamW, weight decay 0.01 (not on biases and norms), 1 warm-up epoch then cosine to 10% of the peak, gradient clip 1.0,",
        "effective batch 16, BCE loss, checkpoint with the best fold-0 macro F1 at 0.25. Frozen BN statistics: running mean/var fixed, scale and shift still trained.",
    ]
    for i, line in enumerate(footer):
        body.append(text(20, y + 6 + 16 * i, line))
    return svg(1060, y + 6 + 16 * len(footer) + 8, body)


# --------------------------------------------------------------------------- Figure 3

def figure_custom_block() -> str:
    body = [text(20, 22, "Exp2: residual spatial block inserted between ConvNeXt stage 4 and the head", size=15, weight="bold", fill=TEXT)]
    y, h = 92, 52
    body += box(20, y, 120, h, "bb", "Stage 4 output", ["1024 × 12²"])
    branch = [("conv", "1×1 conv", "1024 → 256"), ("act", "GELU", ""), ("conv", "3×3 depthwise", "256 ch"),
              ("act", "GELU", ""), ("conv", "1×1 conv", "256 → 1024, zero init")]
    x = 180
    body.append(arrow(140, y + h / 2, x - 2, y + h / 2))
    block_start = x - 10
    for i, (kind, title, sub) in enumerate(branch):
        w = width_for(title, [sub] if sub else [], minimum=64)
        body += box(x, y, w, h, kind, title, [sub] if sub else [])
        x += w
        body.append(arrow(x + 2, y + h / 2, x + 22, y + h / 2))
        x += 24
    cx = x + 12
    f, s = STYLE["block"]
    body.insert(1, f'<rect x="{block_start:.1f}" y="{y - 12}" width="{cx + 24 - block_start:.1f}" height="{h + 24}" rx="10" '
                   f'fill="{f}" fill-opacity="0.45" stroke="{s}" stroke-width="1.4" stroke-dasharray="6 4"/>')
    body.append(text(block_start + 8, y + h + 26, "new block (trainable, LR 1e-5)", size=11, weight="bold", fill=s))
    body.append(f'<circle cx="{cx:.1f}" cy="{y + h / 2:.1f}" r="12" fill="#ffffff" stroke="#374151" stroke-width="1.4"/>')
    body.append(text(cx, y + h / 2 + 5, "+", size=16, weight="bold", fill=TEXT, anchor="middle"))
    top = y - 30
    body.append(f'<path d="M160,{y + h / 2:.1f} L160,{top} L{cx:.1f},{top} L{cx:.1f},{y + h / 2 - 14:.1f}" fill="none" stroke="{LINE}" stroke-width="1.4" marker-end="url(#arrow)"/>')
    body.append(text((160 + cx) / 2, top - 6, "identity: the original stage-4 features pass unchanged", size=11, anchor="middle", italic=True))
    body.append(arrow(cx + 14, y + h / 2, cx + 40, y + h / 2))
    body += box(cx + 42, y, 150, h, "head", "Head (as in Exp1)", ["pool, LN, Linear → 5"])
    width = cx + 42 + 150 + 20
    notes = [
        "• Extra parameters: 528,128 (0.6% of ConvNeXt-base). Trained at the backbone LR (1e-5); everything else matches Exp1.",
        "• The last 1×1 conv starts at zero, so at step 0 the block outputs exactly its input and the network equals Exp1.",
        "• Idea: let the model re-mix spatial context of the final feature map for small pets. Limit: at 12 × 12 the backbone has already",
        "  downsampled 32×, so the block cannot recover detail that earlier stages lost.",
    ]
    for i, line in enumerate(notes):
        body.append(text(20, y + h + 56 + 17 * i, line, fill=TEXT if i == 0 else MUTED))
    return svg(width, y + h + 56 + 17 * len(notes) + 4, body)


# --------------------------------------------------------------------------- Figure 4

def chip(x, y, name, star=False) -> tuple[list[str], float]:
    fill, stroke = MEMBER[name]
    label = name + ("*" if star else "")
    w = 52
    return ([f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
             text(x + w / 2, y + 16, label, size=11, weight="bold", fill=TEXT, anchor="middle")], w)


def figure_ensembles() -> str:
    members = [("Exp1", "ConvNeXt-base", "0.97003"), ("Exp2", "ConvNeXt + block", "0.96043"),
               ("Exp3", "EfficientNetV2-S", "0.97015"), ("Exp5", "BiT ResNetV2-50x1", "0.94035")]
    body = [text(20, 22, "How the ensembles were built from saved predictions (no retraining, threshold 0.25)", size=15, weight="bold", fill=TEXT)]
    # Step 1: members
    body.append(text(20, 50, "1. Each member is trained once on folds 1–4 and saves two probability tables:", size=12, weight="bold", fill=TEXT))
    body.append(text(20, 66, "fold 0 (2,452 images × 5 labels) for local scoring, and test (18,673 × 5) for submissions."))
    x = 20
    for name, model, lb in members:
        fill, stroke = MEMBER[name]
        body.append(f'<rect x="{x}" y="80" width="190" height="44" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        body.append(text(x + 10, 98, f"{name}  {model}", size=12, weight="bold", fill=TEXT))
        body.append(text(x + 10, 115, f"alone: LB {lb}"))
        x += 202
    # Step 2: recipes
    body.append(text(20, 156, "2. Recipes: combine member probabilities, threshold at 0.25, submit.", size=12, weight="bold", fill=TEXT))
    recipes = [
        (["Exp1", "Exp2"], False, "mean of probabilities", "same backbone and start → same mistakes", 0.96824, 0.97528),
        (["Exp1", "Exp3"], False, "mean of probabilities", "two families", 0.97761, 0.98646),
        (["Exp1", "Exp3"], False, "per model: mean of original + mirrored image; then mean", "flip TTA", 0.97420, 0.98636),
        (["Exp1", "Exp3", "Exp5"], False, "mean of probabilities", "three families — best", 0.98047, 0.98656),
        (["Exp1", "Exp3", "Exp5"], False, "logits → one LogReg per label, fitted on fold 0", "stacking", 0.96685, 0.99878),
        (["Exp1", "Exp3", "Exp5"], True, "mean of probabilities (members retrained on all 5 folds)", "Exp6, fixed epochs", 0.97386, None),
    ]
    COMB, BAR, BAR_W, LO, HI = 220, 720, 230, 0.960, 0.985
    y = 172
    body.append(text(20, y + 10, "members", size=11, weight="bold", fill=TEXT))
    body.append(text(COMB, y + 10, "combination", size=11, weight="bold", fill=TEXT))
    body.append(text(BAR, y + 10, f"public LB (axis {LO:.3f} – {HI:.3f})", size=11, weight="bold", fill=TEXT))
    y += 22
    top = y
    for names, star, method, why, lb, local in recipes:
        x = 20
        for name in names:
            parts, w = chip(x, y + 6, name, star)
            body += parts
            x += w + 6
        body.append(arrow(x + 2, y + 18, COMB - 6, y + 18))
        best = lb == 0.98047
        body.append(rect(COMB, y, 440, 36, "head" if best else "note", rx=6))
        body.append(text(COMB + 10, y + 15, method, size=11, weight="bold", fill=TEXT))
        local_text = f"fold-0 F1 {local:.5f}" if local else "no held-out data"
        body.append(text(COMB + 10, y + 29, f"{why} · {local_text}", size=10))
        body.append(arrow(COMB + 442, y + 18, BAR - 6, y + 18))
        w = (lb - LO) / (HI - LO) * BAR_W
        colour = "#7c3aed" if best else "#9ca3af"
        body.append(f'<rect x="{BAR}" y="{y + 8}" width="{w:.1f}" height="20" rx="3" fill="{colour}"/>')
        body.append(text(BAR + (0.981 - LO) / (HI - LO) * BAR_W, y + 23, f"{lb:.5f}", size=12, weight="bold" if best else "normal", fill=TEXT))
        y += 46
    ref = BAR + (0.97003 - LO) / (HI - LO) * BAR_W
    body.append(f'<line x1="{ref:.1f}" y1="{top - 4}" x2="{ref:.1f}" y2="{y - 6}" stroke="#2563eb" stroke-width="1.2" stroke-dasharray="4 3"/>')
    body.append(text(ref, y + 8, "best single model (Exp1 0.97003)", size=10, fill="#2563eb", anchor="middle"))
    body.append(text(20, y + 30, "* Exp6 members: same recipes as Exp1/Exp3/Exp5 (Exp3 at half LR), trained on all 12,260 images for a fixed number of epochs (2 / 8 / 6)."))
    return svg(1040, y + 44, body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "backbones.svg": figure_backbones(),
        "training_phases.svg": figure_training(),
        "custom_block.svg": figure_custom_block(),
        "ensembles.svg": figure_ensembles(),
    }
    for name, content in figures.items():
        (OUT / name).write_text(content)
        print("wrote", (OUT / name).relative_to(OUT.parents[1]))


if __name__ == "__main__":
    main()
