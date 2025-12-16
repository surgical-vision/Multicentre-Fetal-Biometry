#!/usr/bin/env python3
"""
Generate updated LaTeX cross-validation table from test results.
Author: Chiara Di Vece
Date: 2025-12-16
"""
import re
import sys
from collections import defaultdict

# Usage:
#   python make_table_from_log.py logfile.log > cross_data_evaluation_table.tex

LOG = sys.argv[1] if len(sys.argv) > 1 else None
if not LOG:
    raise SystemExit("Usage: python make_table_from_log.py <logfile> > cross_data_evaluation_table.tex")

TRAINS = ["FP", "HC18", "UCL", "MULTICENTRE"]
TESTS  = ["FP", "HC18", "UCL", "MULTICENTRE"]
MEAS   = ["BPD", "OFD", "APAD", "TAD", "FL"]  # columns in your table

# --- Parse log ---
hdr_re = re.compile(
    r"Testing:\s*TRAINED ON=(?P<trained>[\w\-]+),\s*TESTED ON=(?P<tested>[\w\-]+),\s*STRUCTURE=(?P<structure>\w+),\s*METRIC=(?P<metric>\w+)"
)
res_re = re.compile(
    r"Test Results.*?\bnme mean:(?P<mean>[0-9]*\.?[0-9]+)\s+nme std:(?P<std>[0-9]*\.?[0-9]+)"
)

# data[train][test][metric] = (mean, std)
data = defaultdict(lambda: defaultdict(dict))
cur = None

with open(LOG, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = hdr_re.search(line)
        if m:
            cur = m.groupdict()
            continue
        if cur:
            r = res_re.search(line)
            if r:
                tr = cur["trained"]
                te = cur["tested"]
                metric = cur["metric"]
                mean = float(r.group("mean"))
                std = float(r.group("std"))
                data[tr][te][metric] = (mean, std)
                cur = None

def fmt(mean, std, decimals=2):
    return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"

def rank_marks(train, metric):
    """Return dict test->('best'|'second'|None) for this train+metric."""
    vals = []
    for te in TESTS:
        if metric in data.get(train, {}).get(te, {}):
            mean, std = data[train][te][metric]
            vals.append((mean, te))
    vals.sort(key=lambda x: x[0])
    mark = {te: None for te in TESTS}
    if len(vals) >= 1:
        mark[vals[0][1]] = "best"
    if len(vals) >= 2:
        mark[vals[1][1]] = "second"
    return mark

# Precompute best/second marks per training block and metric
marks = {tr: {m: rank_marks(tr, m) for m in MEAS} for tr in TRAINS}

def cell(train, test, metric, decimals=2, missing=""):
    if metric not in data.get(train, {}).get(test, {}):
        return missing
    mean, std = data[train][test][metric]
    s = fmt(mean, std, decimals=decimals)
    tag = marks[train][metric][test]
    if tag == "best":
        return f"\\textbf{{{s}}}"
    if tag == "second":
        return f"\\underline{{{s}}}"
    return s

# --- Emit LaTeX table in your exact structure ---
print(r"\begin{table}")
print(r"\centering")
print(r"\caption{Cross-data evaluation results showing NME$\pm$STD for all train--test combinations across four datasets (FP, HC18, UCL, MULTI-CENTRE) and three anatomies. NME is unitless (measurement error normalised by inter-landmark distance; scale-invariant). Multi-centre models are trained on combined multi-centre data. Within each \emph{training dataset block} and for each biometric measurement, \textbf{bold} indicates the best (lowest) NME across the test datasets in that block and \underline{underline} indicates the second-best.}")
print(r"\label{tab:cross_validation}")
print(r"\begin{tabular}{|c|c|c|c|c|c|c|}")
print(r"\hline")
print(r"\multirow{3}{*}{\textbf{Train}} & \multirow{3}{*}{\textbf{Test}} & \multicolumn{5}{c|}{\textbf{NME $\pm$ STD}} \\")
print(r"\cline{3-7}")
print(r" &  & \multicolumn{2}{c|}{\textbf{Head}} & \multicolumn{2}{c|}{\textbf{Abdomen}} & \textbf{Femur} \\")
print(r"\cline{3-7}")
print(r" &  & \textbf{BPD} & \textbf{OFD} & \textbf{APAD} & \textbf{TAD} & \textbf{FL} \\")
print(r"\hline")

for tr in TRAINS:
    print(rf"\multirow{{4}}{{*}}{{{tr}}} & {TESTS[0]}")
    row = [
        cell(tr, TESTS[0], "BPD"),
        cell(tr, TESTS[0], "OFD"),
        cell(tr, TESTS[0], "APAD"),
        cell(tr, TESTS[0], "TAD"),
        cell(tr, TESTS[0], "FL"),
    ]
    print("& " + " & ".join(row) + r" \\")
    print(r"\cline{2-7}")
    for te in TESTS[1:]:
        print(rf"& {te}")
        row = [
            cell(tr, te, "BPD"),
            cell(tr, te, "OFD"),
            cell(tr, te, "APAD"),
            cell(tr, te, "TAD"),
            cell(tr, te, "FL"),
        ]
        print("& " + " & ".join(row) + r" \\")
        if te != TESTS[-1]:
            print(r"\cline{2-7}")
    print(r"\hline")

print(r"\end{tabular}")
print(r"\end{table}")