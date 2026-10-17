"""
Fig. 4 and Fig. S3 (posterior_and_failing_probability_merged_v2_{contributions,weights}.pdf): d = 3 and 5 from
fault_counts.csv
(exhaustive enumeration, unchanged), d = 7 from this folder's uniform-sampling run (`run.sh`), merged into
fault_counts_v2.csv here and drawn by combined_figure.py.

    python effective_distance_loss/uniform_d7/make_v2.py
"""
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDL = HERE.parent


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


base = [r for r in rows(EDL / "fault_counts.csv") if r["distance"] in ("3", "5")]
uniform = rows(HERE / "fault_counts.csv")
merged = HERE / "fault_counts_v2.csv"
with open(merged, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(base[0]))
    writer.writeheader()
    writer.writerows(base + uniform)
print(f"wrote {merged}: {len(base)} rows at d = 3, 5 and {len(uniform)} uniform rows at d = 7")
subprocess.run([sys.executable, str(EDL / "combined_figure.py"), "--counts", str(merged),
                "--out", "posterior_and_failing_probability_merged_v2.pdf", "--split"], check=True)
