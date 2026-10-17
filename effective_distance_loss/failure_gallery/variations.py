"""
The two d = 3 single-fault examples (fig_d3_reuse, fig_d3_reuse_195) seen from other viewpoints:

    top     straight down the time axis (elev 90): the spatial layout of the discounts, rounds overlaid
    side    low and from the other side (elev 8, azimuth -115): the rounds as layers
    front   level, facing the x-t plane (elev 0, azimuth -90): rounds stacked vertically, y collapsed
    above   high, from behind (elev 40, azimuth 30): the reverse of the default view

    python effective_distance_loss/failure_gallery/variations.py
"""
import json
from pathlib import Path

import gallery_figure
import make_gallery as mg

HERE = Path(__file__).resolve().parent
VIEWS = {"top": (90, -90), "side": (8, -115), "front": (0, -90), "above": (40, 30)}
LEGEND_Y = {"above": "faults"}                    # legend centred on the circuit-fault lines
COMPACT = {"above": dict(zoom=1.45, width=0.72)}   # narrower figure, larger graph: less free space

c = mg.context(3)
saved = json.load(open(HERE / "examples.json"))
for stem, members in (("fig_d3_reuse", [61]), ("fig_d3_reuse_195", [195])):
    ex = next(x for x in saved["main"] + saved["additional"] if x["d"] == 3 and x["members"] == members)
    for tag, view in VIEWS.items():
        gallery_figure.draw(c, ex, HERE / f"{stem}_view_{tag}.pdf", view=view, legend_y=LEGEND_Y.get(tag, 0.0),
                            **COMPACT.get(tag, {}))
        print("wrote", f"{stem}_view_{tag}.pdf")
    if stem == "fig_d3_reuse":                       # the "above" view without the alpha in the CM title
        gallery_figure.draw(c, ex, HERE / f"{stem}_view_above_noalpha.pdf", view=VIEWS["above"],
                            legend_y=LEGEND_Y["above"], cm_title="CM second pass", **COMPACT["above"])
        print("wrote", f"{stem}_view_above_noalpha.pdf")
        gallery_figure.draw(c, ex, HERE / f"{stem}_view_above_noalpha_nolegend.pdf", view=VIEWS["above"],
                            cm_title="CM second pass", captions=False, fontscale=1.3, **COMPACT["above"])
        print("wrote", f"{stem}_view_above_noalpha_nolegend.pdf")
        gallery_figure.draw(c, ex, HERE / f"{stem}_view_above_noalpha_nolegend_nozl.pdf", view=VIEWS["above"],
                            cm_title="CM second pass", captions=False, fontscale=1.3, zl_labels=False, crop=True,
                            **COMPACT["above"])
        print("wrote", f"{stem}_view_above_noalpha_nolegend_nozl.pdf")
        gallery_figure.draw(c, ex, HERE / f"{stem}_view_above_noalpha_nolegend_star.pdf", view=VIEWS["above"],
                            cm_title="CM second pass", captions=False, flip_mark="star", **COMPACT["above"])
        print("wrote", f"{stem}_view_above_noalpha_nolegend_star.pdf")
