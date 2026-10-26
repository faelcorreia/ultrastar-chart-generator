"""I4: compare a generated UltraStar .txt with a reference one.

Usage: python scripts/i4_compare.py generated.txt reference.txt [tol_seconds]
Prints line counts, line-start timing error (nearest generated line start for
each reference line) and the share of reference lines within the tolerance.
"""
import re
import statistics
import sys


def line_starts(path):
    bpm, gap, relative = 0.0, 0.0, False
    starts, cur, words = [], None, []
    lines_text = []
    for raw in open(path, encoding="utf-8", errors="ignore"):
        raw = raw.rstrip("\r\n")
        if raw.startswith("#"):
            k, _, v = raw[1:].partition(":")
            k = k.upper()
            if k == "BPM":
                bpm = float(v.replace(",", "."))
            elif k == "GAP":
                gap = float(v.replace(",", "."))
            elif k == "RELATIVE":
                relative = v.strip().upper() == "YES"
            continue
        if raw[:1] in ":*FRG" and raw[1:2] == " ":
            parts = raw.split(" ", 4)
            beat = int(parts[1])
            if cur is None:
                cur = beat
            words.append(parts[4] if len(parts) > 4 else "")
        elif raw.startswith("-") or raw.startswith("E"):
            if cur is not None:
                starts.append(cur * 15.0 / bpm + gap / 1000.0)
                lines_text.append("".join(words).strip())
            cur, words = None, []
    return starts, lines_text, relative


def compare(gen, ref, tol=0.5):
    g, gt, _ = line_starts(gen)
    r, rt, rel = line_starts(ref)
    errs = [min(abs(x - y) for y in g) for x in r] if g else []
    signed = []
    for x in r:
        if g:
            y = min(g, key=lambda y: abs(y - x))
            signed.append(y - x)
    ok = sum(e <= tol for e in errs)
    syl = lambda t: max(len(re.findall(r"\S+", " ".join(t))), 0)
    return {
        "gen_lines": len(g), "ref_lines": len(r),
        "median_err": statistics.median(errs) if errs else None,
        "median_signed": statistics.median(signed) if signed else None,
        "pct_ok": 100.0 * ok / len(r) if r else 0.0,
        "tol": tol,
        "max_gen_words": max((len(t.split()) for t in gt), default=0),
        "ref_relative": rel,
    }


if __name__ == "__main__":
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    for k, v in compare(sys.argv[1], sys.argv[2], tol).items():
        print(f"{k}: {v}")
