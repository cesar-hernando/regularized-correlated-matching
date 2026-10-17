"""
Decoding-graph figure of one CM failure: the true error, then the MWPM first pass leading to CM's
second pass (alpha = 1). Each panel is the whole (x, y, t) decoding graph, every round.

    matching panels      the selected edges, red (X graph) and blue (Z graph)
    first-pass panel     also every edge the first pass discounts for the second pass, thin and soft, in
                         its graph's colour
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures"))
from lcd_figures import PAGE_WIDTH, paper_style  # noqa: E402  (PRL \textwidth, 7.08 in)

BASIS = {0: "#d62728", 1: "#1f77b4"}           # X-stabiliser detectors red, Z blue
OK, BAD = "#1e8449", "#c0392b"
TICK = "#f28e00"                                 # marks selected edges that flip the logical observable
HEIGHT, TITLE_Y, ZL_Y = 0.42, 0.97, 0.035                    # figure height / PAGE_WIDTH; title position in the panel
ZOOM = {3: 1.4, 5: 1.22, 7: 1.08}                # graph size per distance, as large as the titles and labels allow
LINE = 0.06 * 0.56 / HEIGHT                      # spacing of the text lines under a panel (axes units)
DISCOUNT_LW = 0.7                                # discounted edges (w_post < w_prior): thin, in their graph's colour,
DISCOUNT_ALPHA = 0.45                            # and soft, so the first-pass matching stands out


def _qubit(x, y):
    """Role of a qubit: data qubits sit at odd (x, y), ancillas elsewhere."""
    x, y = int(float(x)), int(float(y))
    return "data" if x % 2 == 1 and y % 2 == 1 else "ancilla"


def fault_label(m):
    """Short circuit-level name of a mechanism, from its representative fault, e.g.
    'CZ: X ancilla, Y data' or 'idle: X data'."""
    text = m["fault"]
    q = re.findall(r"([XYZ]) on qubit \d+ \(([\d.]+), ([\d.]+)\)", text)
    if text.startswith("DEPOLARIZE2"):
        q.sort(key=lambda t: _qubit(t[1], t[2]).startswith("data"))         # ancilla first
        return "CZ: " + ", ".join(f"{P} {_qubit(x, y)}" for P, x, y in q)
    if text.startswith("DEPOLARIZE1"):
        kind = "idle" if "idle" in m["label"] else "1q gate"
        return f"{kind}: " + ", ".join(f"{P} {_qubit(x, y)}" for P, x, y in q)
    if text.startswith("X_ERROR"):
        return "reset flip: " + ", ".join(_qubit(x, y) for _, x, y in q)
    if text.startswith("M "):
        x, y = re.search(r"\(([\d.]+), ([\d.]+)\)", text).groups()
        return f"meas. flip: {_qubit(x, y)}"
    return m["label"]


def _t(e):
    return (int(e[0]), int(e[1]))


def draw(c, ex, path, view=(20, -60), legend_y=0.0, zoom=None, width=0.78,
         cm_title=r"CM second pass ($\alpha = 1$)", captions=True, flip_mark="tick", fontscale=1.0, zl_labels=True, crop=False, legend=True, titles=True, inline_faults=False, faults_fontscale=None,
         column_gap=0.0, dark_graph=False, height=HEIGHT, faults_drop=0.05,
         mark_y=TITLE_Y, clip=True, faults_left=0.0, mark_x=0.5, width_ratios=None, arrow_scale=1.0,
         detector_scale=1.0, edge_scale=1.0):
    """`captions=False` drops the circuit-fault lines and the reweighted-edge legend (Z_L labels stay);
    `legend=False` drops only the legend, `titles=False` the panel titles (the verdict marks stay, alone and larger).
    `inline_faults=True` writes the circuit faults on one line after "Circuit faults:" (two or more if it is
    wider than the figure), scaled by `faults_fontscale` (default `fontscale`). `crop=True` crops to what is
    drawn, `crop="vertical"` only top and bottom, so that figures of one width line up. `column_gap` is the
    space between the three panels, as a fraction of a panel's width (matplotlib's wspace). `dark_graph=True`
    draws the decoding graph's edges and nodes in black instead of light grey, for print; `height` is the
    figure height as a fraction of PAGE_WIDTH, for room above and below zoomed-in graphs; `faults_drop` how far
    (axes units) the inline circuit faults sit below where the Z_L labels would be, when there are none;
    `mark_y` and `mark_x` the position (axes units) of the verdict marks when there are no titles: centred
    over the graph by default, beside it past `mark_x` = 1. `width_ratios`, the
    widths of the three columns (matplotlib's), equal by default; `arrow_scale` scales the arrow's head and line,
    `detector_scale` the fired detectors (their diameter) and `edge_scale` the width of the true and selected edges
    and of their Z_L ticks. `clip=False` lets the
    graphs reach past their (square) 3D axes into the free space beside them, so they can be zoomed in further.
    `faults_left` > 0 reserves that fraction of the figure width, left of the true error, for the inline circuit
    faults, written there one per line under "Circuit faults:" and centred on the graph, instead of under it (the
    graphs then take the full height, and the true error's sits at the left of its column, next to them)."""
    with paper_style(use_latex=True):
        _draw(c, ex, path, view, legend_y, zoom, width, cm_title, captions, flip_mark, fontscale, zl_labels, crop,
              legend, titles, inline_faults, faults_fontscale or fontscale, column_gap, dark_graph, height,
              faults_drop, mark_y, clip, faults_left, mark_x, width_ratios, arrow_scale,
              detector_scale, edge_scale)


def _draw(c, ex, path, view, legend_y, zoom, width, cm_title, captions, flip_mark, fontscale, zl_labels, crop,
          legend, titles, inline_faults, faults_fontscale, column_gap, dark_graph, height, faults_drop,
          mark_y, clip, faults_left, mark_x, width_ratios, arrow_scale, detector_scale, edge_scale):
    xyz = {k: np.array(v[:3], float) for k, v in c["coords"].items()}
    bases = c["bases"]
    allp = np.array(list(xyz.values()))
    lows, highs = allp.min(axis=0) - (1.0, 1.0, 0.3), allp.max(axis=0) + (1.0, 1.0, 0.3)
    centre = allp.mean(axis=0)
    span = highs - lows
    fired = set()
    for m in ex["members"]:
        fired ^= set(int(k) for k in np.flatnonzero(c["det"][m]))
    first = [_t(r["edge"]) for r in ex["first_pass"]]
    # edges that flip the logical observable: the "wall" a correction must cross an even number of
    # times, together with the error, to be correct
    obs_edges = {_t((-1, u) if v is None else (min(u, v), max(u, v)))
                 for u, v, data in c["mwpm"].edges() if data.get("fault_ids")}

    def outward(e):
        """Direction of a boundary edge, towards the boundary it ends on.

        Z-graph boundary edges end on the left boundary (-x) if they flip the logical observable,
        which is Z on the left column of data qubits, and on the right boundary (+x) otherwise;
        X-graph boundary edges end on the top or bottom boundary (+-y), whichever is nearer.
        """
        b = e[1]
        if bases[b] == 1:
            return np.array([-1.0 if e in obs_edges else 1.0, 0.0, 0.0])
        return np.array([0.0, 1.0 if xyz[b][1] >= centre[1] else -1.0, 0.0])

    def seg(ax, e, **kw):
        a, b = e
        if a == -1:
            ax.plot(*zip(xyz[b], xyz[b] + outward(e) * 1.2), **kw)
        else:
            ax.plot(*zip(xyz[a], xyz[b]), **kw)

    def node(e):
        return e[1] if e[0] == -1 else e[0]

    def midpoint(e):
        a, b = e
        if a == -1:
            return xyz[b] + outward(e) * 0.6
        return (xyz[a] + xyz[b]) / 2

    def base(ax, title, colour="black"):
        ax.set_box_aspect((span[0], span[1], span[2] * 1.6), zoom=zoom or ZOOM.get(ex["d"], 1.1))
        # fixed layers instead of depth sorting, so no highlighted edge is hidden part-way: graph,
        # nodes, detector halos, matched edges, discounted edges, detectors, ticks
        ax.computed_zorder = False
        for a, b in c["priors"]:
            if a != -1:
                ax.plot(*zip(xyz[a], xyz[b]), color="black" if dark_graph else "0.87", lw=0.1 if dark_graph else 0.18,
                        zorder=1)
        for kk in xyz:
            ax.scatter(*xyz[kk], s=0.8 if dark_graph else 0.6, color="black" if dark_graph else "0.65",
                       depthshade=False, zorder=2)
        for kk in fired:
            ax.scatter(*xyz[kk], s=26 * detector_scale ** 2, color=BASIS[bases[kk]], alpha=0.25, depthshade=False,
                       zorder=3)
            ax.scatter(*xyz[kk], s=6 * detector_scale ** 2, color=BASIS[bases[kk]], depthshade=False, zorder=7)
        ax.set_xlim(lows[0], highs[0]); ax.set_ylim(lows[1], highs[1]); ax.set_zlim(lows[2], highs[2])
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_axis_off()
        if titles:
            ax.set_title(title, fontsize=8 * fontscale, y=TITLE_Y, color=colour)

    def matching(ax, edges, weights):
        for e in edges:
            seg(ax, e, color=BASIS[bases[node(e)]], lw=1.6 * edge_scale, zorder=5)
        crossing = [e for e in edges if e in obs_edges]
        for e in crossing:                       # a short orange tick across each selected edge that flips Z_L
            a, b = e
            along = outward(e) if a == -1 else xyz[b] - xyz[a]
            perp = np.array([-along[1], along[0], 0.0])
            perp = perp / (np.linalg.norm(perp) + 1e-9) * 0.45
            mid = midpoint(e)
            if flip_mark == "star":              # an orange star on the edge instead of the tick
                ax.scatter(*mid, marker="*", s=55, color=TICK, edgecolor="white", lw=0.3, depthshade=False, zorder=8)
            else:
                ax.plot(*zip(mid - perp, mid + perp), color=TICK, lw=1.4 * edge_scale, solid_capstyle="butt", zorder=8)
        flips = len(crossing) % 2 == 1
        if not zl_labels:
            return
        ax.text2D(0.53, ZL_Y, "$Z_L$ flip:", transform=ax.transAxes, ha="right", fontsize=6.5 * fontscale, color="0.2")
        ax.text2D(0.55, ZL_Y, "True" if flips else "False", transform=ax.transAxes, ha="left", fontsize=6.5 * fontscale,
                  color=TICK if flips else "0.2")

    def reweighting(ax):
        for r in ex["reweights"]:
            if r["w_prior"] - r["w_cm"] > 1e-9:
                e = _t(r["edge"])
                seg(ax, e, color=BASIS[bases[node(e)]], lw=DISCOUNT_LW, alpha=DISCOUNT_ALPHA, zorder=6)
        if not (captions and legend):
            return
        handles =[Line2D([], [], color=BASIS[0], lw=DISCOUNT_LW, alpha=DISCOUNT_ALPHA, label="X-type reweighted edges"),
                   Line2D([], [], color=BASIS[1], lw=DISCOUNT_LW, alpha=DISCOUNT_ALPHA, label="Z-type reweighted edges")]
        key = ax.legend(handles=handles, loc="upper center",
                        bbox_to_anchor=(0.5, -0.04 if legend_y == "faults" else legend_y - 0.04), ncol=1, frameon=False,
                        fontsize=6.5 * fontscale, handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)
        legends.append(key)

    verdicts = []
    legends = []

    def verdict(ax, name, wrong):
        """Title, followed by a green tick (correct) or red cross (wrong), placed once laid out."""
        base(ax, name)
        verdicts.append((ax, wrong))

    def place_verdicts():
        # usetex text cannot change colour part-way, so the mark is its own text after the title
        fig.canvas.draw()                         # titles are positioned at draw time
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        for ax, wrong in verdicts:
            colour, sym = (BAD, r"$\times$") if wrong else (OK, r"$\checkmark$")
            if not titles:                        # the decoder is named elsewhere: the mark alone, larger
                ax.text2D(mark_x, mark_y, sym, transform=ax.transAxes, ha="center", va="center",
                          fontsize=13 * fontscale, color=colour)
                continue
            box = ax.title.get_window_extent(renderer)
            x, y = to_fig.transform((box.x1, (box.y0 + box.y1) / 2))
            fig.text(x + 0.006, y, sym, ha="left", va="center", fontsize=8 * fontscale, color=colour)

    fig = plt.figure(figsize=(PAGE_WIDTH * width, PAGE_WIDTH * height))
    bottom = 0.02 if faults_left else 0.11                 # no faults line under the graphs if they are beside them
    gs = fig.add_gridspec(1, 3, left=faults_left, right=1.0, bottom=bottom, top=0.98, wspace=column_gap,
                          width_ratios=width_ratios)
    prior = {e: math.log((1 - q) / q) for e, q in c["priors"].items()}       # MWPM's edge weights
    ax_true = fig.add_subplot(gs[0, 0], projection="3d")
    if faults_left:
        ax_true.set_anchor("W")                            # the graph at the left of its column, by its faults
    base(ax_true, "True error")
    matching(ax_true, [_t(e) for e in ex["true_edges"]], prior)
    lines = (["Circuit faults:"] + [fault_label(m) for m in ex["mechanisms"]]) if captions and not inline_faults else []
    fault_texts = [ax_true.text2D(0.5, -0.075 - LINE * i, line, transform=ax_true.transAxes, ha="center",
                                  fontsize=6.5 * fontscale, color="black" if i == 0 else "0.2") for i, line in enumerate(lines)]
    ax_mwpm = fig.add_subplot(gs[0, 1], projection="3d")
    verdict(ax_mwpm, "MWPM (first pass)", ex["wrong"]["MWPM"])
    reweighting(ax_mwpm)                                   # under the matching
    matching(ax_mwpm, first, prior)
    ax_cm = fig.add_subplot(gs[0, 2], projection="3d")
    verdict(ax_cm, cm_title, ex["wrong"]["CM"])
    matching(ax_cm, [_t(r["edge"]) for r in ex["second_pass"]], {_t(r["edge"]): r["w_cm"] for r in ex["second_pass"]})

    # the arrow from the first pass to the second, in figure coordinates, just right of the gap between them
    mid = (gs[0, 1].get_position(fig).x1 + gs[0, 2].get_position(fig).x0) / 2 + 0.008
    y_arrow = (bottom + 0.98) / 2 + 0.005
    half = 0.02 * min(arrow_scale, 1.15)                  # a little longer at most, so that it clears the graphs
    fig.add_artist(FancyArrowPatch((mid - half, y_arrow), (mid + half, y_arrow), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=8 * arrow_scale, color="0.35",
                                   lw=0.8 * arrow_scale))

    place_verdicts()
    if captions and inline_faults and faults_left:
        # "Circuit faults:" over the faults, one per line, from the left edge, centred on the true error's graph
        size = 6.5 * faults_fontscale
        lines = ["Circuit faults:"] + [fault_label(m) for m in ex["mechanisms"]]
        step = 1.3 * size / 72 / fig.get_figheight()                         # line spacing, figure units
        box = ax_true.get_position()
        top = (box.y0 + box.y1) / 2 + (len(lines) - 1) * step / 2
        for i, line in enumerate(lines):
            fig.text(0.0, top - i * step, line, ha="left", va="center", fontsize=size,
                     color="black")
    elif captions and inline_faults:
        # "Circuit faults:" and, after it, the faults, in their own colours: one line centred under the true
        # error, or from the left edge of the figure if it is wider than the panel, and over as many lines as
        # it takes, under each other, if it is wider than the figure. In place of the Z_L labels, a little
        # lower, if there are none
        to_fig = fig.transFigure.inverted()
        renderer = fig.canvas.get_renderer()
        y = to_fig.transform(ax_true.transAxes.transform((0.0, -0.075 if zl_labels else ZL_Y - faults_drop)))[1]
        size = 6.5 * faults_fontscale
        head = fig.text(0.0, y, "Circuit faults:", ha="left", va="baseline", fontsize=size)
        gap, margin = 0.008, 0.01

        def width(t):
            return to_fig.transform_bbox(t.get_window_extent(renderer)).width

        room = 1.0 - margin - width(head) - gap
        rows, probe = [[]], fig.text(0.0, 0.0, "", fontsize=size)
        for label in (fault_label(m) for m in ex["mechanisms"]):
            probe.set_text("; ".join(rows[-1] + [label]) + ";")
            if rows[-1] and width(probe) > room:
                rows.append([])
            rows[-1].append(label)
        probe.remove()
        step = 1.25 * size / 72 / fig.get_figheight()                        # line spacing, figure units
        body = [fig.text(0.0, y - i * step, "; ".join(r) + (";" if i < len(rows) - 1 else ""), ha="left",
                         va="baseline", fontsize=size, color="0.2") for i, r in enumerate(rows)]
        column = gs[0, 0].get_position(fig)
        x0 = max(0.0, (column.x0 + column.x1) / 2 - (width(head) + gap + max(map(width, body))) / 2)
        head.set_x(x0)
        for t in body:
            t.set_x(x0 + width(head) + gap)
    if legend_y == "faults" and legends:
        # centre the reweighted-edge legend on the circuit-fault lines under the true error
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in fault_texts]
        target = (min(b.y0 for b in boxes) + max(b.y1 for b in boxes)) / 2
        box = legends[0].get_window_extent(renderer)
        y = ax_mwpm.transAxes.inverted().transform((0.0, target))[1]
        legends[0].set_bbox_to_anchor((0.5, y), transform=ax_mwpm.transAxes)
        legends[0].set_loc("center") if hasattr(legends[0], "set_loc") else setattr(legends[0], "_loc", 10)
    if not clip:
        for ax in fig.axes:
            for a in [*ax.lines, *ax.collections, *ax.patches]:
                a.set_clip_on(False)
    if crop:
        # crop to what is drawn (graphs, labels, arrow) rather than the 3D axes boxes, which leave
        # empty space below the graphs
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for ax in fig.axes:
            for a in [*ax.lines, *ax.collections, *ax.texts, ax.title]:
                if a.get_visible() and (not hasattr(a, "get_text") or a.get_text()):
                    b = a.get_window_extent(renderer)
                    if np.all(np.isfinite(b.get_points())) and b.width < 1e5:
                        boxes.append(b)
        boxes += [t.get_window_extent(renderer) for t in fig.texts]
        boxes += [a.get_window_extent(renderer) for a in fig.patches]
        box = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(0.03)
        if crop == "vertical":
            box = Bbox([[0.0, box.y0], [fig.get_figwidth(), box.y1]])
        fig.savefig(path, bbox_inches=box)
    else:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
