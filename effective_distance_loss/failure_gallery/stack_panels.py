"""
The panels of fig_mechanisms_stack.tex, redrawn from examples.json as stack_<figure>.pdf, so the
gallery's own figures stay as they are: larger text (FONTSCALE, TEXTSCALE), no titles, legend or Z_L labels, the circuit faults
beside the true error (FAULTS_LEFT), and all of one width (WIDTH) so that their columns line up. The column titles, shared by all
panels, are stack_header.pdf, and the legend stack_legend.pdf.

    python effective_distance_loss/failure_gallery/stack_panels.py
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import gallery_figure
import make_gallery as mg
from gallery_figure import BASIS, DISCOUNT_ALPHA, DISCOUNT_LW, HEIGHT, PAGE_WIDTH, TICK, paper_style

HERE = Path(__file__).resolve().parent
FONTSCALE = 2.4                                  # the column titles and verdict marks
TEXTSCALE = 3.0                                  # the legend and the circuit faults
GLYPHSCALE = 2.6                                 # the legend's lines and markers, against the panels'
MIN_WIDTH = 2.48                                 # of every panel, as a fraction of PAGE_WIDTH, unless the
LETTER_WIDTH = 1.15 / 2.54                       # legend is wider than the panels and their letters (\letterw, in):
                                                 # wide enough that the figure (about 18 x 21.6 in) has the
                                                 # proportions of a page's text block less its caption, 7.08 x 8.5 in
FAULTS_LEFT = 0.15                               # of every panel's width, left of the true error, for its faults
WIDTH_RATIOS = (0.75, 1, 1)                      # the true error's column narrower: no mark, and its graph at its left
COLUMN_GAP = 0.15                                # between the three graphs of a panel, as a fraction of one
ABOVE = dict(view=(40, 30), zoom=1.8)            # the "above" view of variations.py, zoomed in, unclipped
ZOOM_D5 = 1.6                                    # the d = 5 graphs, zoomed in (gallery_figure: 1.22), unclipped
PANEL_HEIGHT = 1.29 * HEIGHT                     # taller panels for larger graphs (the marks are beside them)
DETECTOR_SCALE = 1.6                             # the fired detectors' diameter in the panels, against gallery_figure
EDGE_SCALE = 2.0                                 # the true and selected edges' width in the panels, against gallery_figure
ARROW_SCALE = 2.2                                # the first-to-second-pass arrows: head and line, against gallery_figure
MARK = (1.06, 0.9)                               # the verdict marks, beside the graphs. top right corner (axes units)
# (figure, distance, members, extra draw arguments)
PANELS = [("fig_d3_reuse_195_view_above", 3, [195], ABOVE),
          ("fig_d5_abandoned", 5, None, dict(zoom=ZOOM_D5)),
          ("fig_d5_phantom_257_364", 5, [257, 364], dict(zoom=ZOOM_D5)),
          ("fig_d5_reuse_17_227", 5, [17, 227], dict(zoom=ZOOM_D5)),
          ("fig_d5_reuse", 5, None, dict(zoom=ZOOM_D5))]


class Glyph:
    """A legend key drawn by `draw(x0, y0, width, height)`, which returns its artists."""
    def __init__(self, draw):
        self.draw = draw


class GlyphHandler:
    def legend_artist(self, legend, orig, fontsize, box):
        artists = orig.draw(box.xdescent, box.ydescent, box.width, box.height)
        for a in artists:
            a.set_transform(box.get_transform())
            box.add_artist(a)
        return artists[0]


def detector(x, y, basis):
    """A fired detector, as in the panels (a soft halo, scatter s = 26, around a dot, s = 6), GLYPHSCALE larger."""
    k = GLYPHSCALE
    return [Line2D([x], [y], ls="", marker="o", ms=math.sqrt(26) * k, mew=0, color=BASIS[basis], alpha=0.25),
            Line2D([x], [y], ls="", marker="o", ms=math.sqrt(6) * k, mew=0, color=BASIS[basis])]


def pair(lw, **kw):
    """An X-graph (red) edge above a Z-graph (blue) one, of the panels' width `lw`, GLYPHSCALE wider."""
    return Glyph(lambda x, y, w, h: [Line2D([x, x + w], [y + h * f] * 2, color=BASIS[b], lw=lw * GLYPHSCALE, **kw)
                                     for b, f in ((0, 0.85), (1, 0.15))])


def legend():
    """The keys, in three columns of two: the detectors, the edges, the reweighted and logical edges."""
    keys = [(Glyph(lambda x, y, w, h: detector(x + w / 2, y + h / 2, 0)), "Triggered X-type detectors"),
            (Glyph(lambda x, y, w, h: detector(x + w / 2, y + h / 2, 1)), "Triggered Z-type detectors"),
            (pair(lw=1.6), "Underlying / predicted edges"),
            (Glyph(lambda x, y, w, h: [Line2D([x + 0.15 * w, x + w], [y + h / 2] * 2, color=BASIS[1],
                                              lw=1.6 * GLYPHSCALE),
                                       *detector(x + 0.15 * w, y + h / 2, 1)]),
             "Boundary edges (a single node)"),
            (pair(lw=DISCOUNT_LW, alpha=DISCOUNT_ALPHA), "Reweighted edges"),
            (Glyph(lambda x, y, w, h: [Line2D([x, x + w], [y + h / 2] * 2, color=BASIS[1], lw=1.6 * GLYPHSCALE),
                                       Line2D([x + w / 2] * 2, [y - 0.15 * h, y + 1.15 * h], color=TICK,
                                              lw=1.4 * GLYPHSCALE, solid_capstyle="butt")]),
             r"Anticommutes with the reference $Z_L$")]
    with paper_style(use_latex=True):
        fig = plt.figure(figsize=(PAGE_WIDTH, 0.5))
        key = fig.legend(*zip(*keys), handler_map={Glyph: GlyphHandler()}, loc="center", ncol=3, frameon=True,
                         fancybox=False, edgecolor="0.45", borderpad=0.7,
                         fontsize=6.5 * TEXTSCALE, handlelength=1.9, handleheight=0.9, columnspacing=1.0,
                         labelspacing=0.6)
        key.get_frame().set_linewidth(0.6)
        fig.savefig(HERE / "stack_legend.pdf", bbox_inches="tight", pad_inches=0.02)
        width = key.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi + 0.04      # as saved, in inches
        plt.close(fig)
    print("wrote stack_legend.pdf")
    return width


def header(width):
    """The titles of the three columns, in bold, each on one line and centred on its column, and if need be
    smaller than the panels' titles were so that each fits in its column; the verdict marks are in the panels."""
    with paper_style(use_latex=True):
        fig = plt.figure(figsize=(PAGE_WIDTH * width, 0.018 * 8 * FONTSCALE))  # one line of title
        columns = [fig.add_gridspec(1, 3, left=FAULTS_LEFT, right=1.0, wspace=COLUMN_GAP,
                                     width_ratios=WIDTH_RATIOS)[0, i].get_position(fig)
                   for i in range(3)]                    # as in gallery_figure
        # "True error" centred on its faults and graph together: from the left edge to the right of the graph,
        # which sits at the left of its column and is as wide as it is high (gallery_figure: axes 0.96 high)
        side = 0.96 * PANEL_HEIGHT / width
        centres = [(FAULTS_LEFT + side) / 2] + [(col.x0 + col.x1) / 2 for col in columns[1:]]
        titles = [fig.text(x, 0.5,
                           r"{\boldmath\bfseries\begin{tabular}{c}%s\end{tabular}}" % r"\\".join(lines),
                           ha="center", va="center", fontsize=8 * FONTSCALE)
                  for x, lines in zip(centres, (["True error"], ["MWPM (first pass)"],
                                                  [r"CM second pass ($\alpha = 1$)"]))]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        scale = min(1.0, *(0.95 * col.width / fig.transFigure.inverted().transform_bbox(
            t.get_window_extent(renderer)).width for t, col in zip(titles[1:], columns[1:])))   # each in its column
        for t in titles:
            t.set_fontsize(8 * FONTSCALE * scale)
        fig.savefig(HERE / "stack_header.pdf")          # the full width, as the panels
        plt.close(fig)
    print("wrote stack_header.pdf")


# the panels as wide as the legend, so that the circuit faults have room and the columns fill the figure
WIDTH = max(MIN_WIDTH, (legend() - LETTER_WIDTH) / PAGE_WIDTH)
header(WIDTH)
saved = json.load(open(HERE / "examples.json"))
contexts = {}
for name, d, members, kw in PANELS:
    if members is None:                          # a main example, by its failure mode
        mode = name.split("_")[-1]
        ex = next(x for x in saved["main"] if x["d"] == d and x["mode"] == mode)
    else:
        ex = next(x for x in saved["main"] + saved["additional"] if x["d"] == d and x["members"] == members)
    c = contexts.setdefault(d, mg.context(d))
    gallery_figure.draw(c, ex, HERE / f"stack_{name}.pdf", width=WIDTH, fontscale=FONTSCALE, legend=False,
                        titles=False, zl_labels=False, inline_faults=True, faults_fontscale=TEXTSCALE,
                        column_gap=COLUMN_GAP, height=PANEL_HEIGHT, faults_drop=0.09, mark_y=MARK[1], mark_x=MARK[0],
                        crop="vertical", clip=False, faults_left=FAULTS_LEFT, width_ratios=WIDTH_RATIOS, arrow_scale=ARROW_SCALE,
                        detector_scale=DETECTOR_SCALE, edge_scale=EDGE_SCALE, **kw)
    print("wrote", f"stack_{name}.pdf")
