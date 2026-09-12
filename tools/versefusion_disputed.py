"""The twelve cases where VerSeFusion and its own audit disagree about the phenotype.

    python tools/versefusion_disputed.py --audit .../lstv_audit_summary.csv \
        --report .../vf_convert_*.json --out logs/versefusion_disputed.csv

WHY THIS EXISTS. Counting L6 in the masks gives 50; the repo's audit CSV says 44. Counting
T13 gives 6; the audit says 18. Two counts of the same thing disagreeing in OPPOSITE
directions is not a counting error -- it is two different objects being counted.

They are separated by the VERIDAH corrections. The audit describes the corpus BEFORE them
and data/unified is AFTER, and every one of the 18 discrepancies is a `t13_shift` case. The
correction reinterprets a supernumerary THORACIC vertebra as a supernumerary LUMBAR one:
T13 becomes L1, the chain shifts down, and L5 becomes L6. Twelve T13s disappear and six L6s
appear -- six, not twelve, because in the other six the field of view stops above the level
that would have become the sixth.

WHY IT MATTERS MORE HERE THAN ANYWHERE ELSE. T13-versus-L6 is the phenotype this dataset
exists to study. A thirteenth thoracic vertebra and a sixth lumbar are the same bone under
two readings, and the release carries separate identifiers for exactly that reason. So
these twelve are not noise to be corrected away, and they are not obviously right either:
they are the most interesting cases in the corpus, and which reading is used has to be a
recorded decision rather than whichever directory was loaded.

The output is a case list, so a run can take either reading, or hold the twelve out, and
say which it took.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, help="VerSeFusion lstv_audit_summary.csv")
    ap.add_argument("--report", required=True,
                    help="convert_versefusion.py --report JSON, read from the MASKS")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    audit = {r["series_id"]: r for r in csv.DictReader(open(a.audit))}
    rep = {r["case"]: r for r in json.loads(Path(a.report).read_text()) if r.get("ok")}

    rows = []
    for cid in sorted(set(audit) & set(rep)):
        au, ms = audit[cid], rep[cid]
        a_l6, a_t13 = au["has_l6"] == "True", au["has_t13"] == "True"
        m_l6, m_t13 = bool(ms["has_L6"]), bool(ms["has_T13"])
        if (a_l6, a_t13) == (m_l6, m_t13):
            continue
        rows.append({
            "case": cid,
            "audit_says_L6": int(a_l6), "mask_says_L6": int(m_l6),
            "audit_says_T13": int(a_t13), "mask_says_T13": int(m_t13),
            "veridah_correction": au["veridah_correction_type"],
            "reading_before": "T13" if a_t13 else ("L6" if a_l6 else "neither"),
            "reading_after": "T13" if m_t13 else ("L6" if m_l6 else "neither"),
        })

    print(f"{len(rows)} cases where the two readings differ\n")
    print(f"  {'case':<12} {'before':<8} {'after':<8} correction")
    for r in rows:
        print(f"  {r['case']:<12} {r['reading_before']:<8} {r['reading_after']:<8} "
              f"{r['veridah_correction']}")
    kinds = {}
    for r in rows:
        k = (r["reading_before"], r["reading_after"])
        kinds[k] = kinds.get(k, 0) + 1
    print("\n  transitions:")
    for (b, af), n in sorted(kinds.items()):
        print(f"    {b} -> {af}: {n}")
    print("\n  Using data/unified as-is takes the AFTER reading on every one of these.")

    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
