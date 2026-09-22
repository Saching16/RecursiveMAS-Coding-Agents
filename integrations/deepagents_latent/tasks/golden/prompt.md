# Golden task prompt

The text below is what gets handed to the agent under test. It is deliberately
underspecified on one point — the unit basis — because that is the fact the
delegation boundary has to carry (see `oracle.json` → `p4_ambiguity`). Do not
"fix" the prompt by naming mebibytes in it; that would remove the measurement.

---

## Task

The `srcpkg` package summarizes network transfer samples. A sample is a byte
count for one observation window.

`srcpkg/report.py`'s `summarize()` currently reports sample count, total bytes,
and peak. It should also report **average throughput**.

Add throughput support and make the test suite pass:

```
python3 -m unittest discover -s tests -t .
```

The existing tests define the expected behavior. Read them.

---

## Notes for the harness (not part of the agent prompt)

- **Starting state:** `repo/` as committed. Baseline is 8 tests, 5 passing,
  3 failing, all three from `srcpkg.stats.throughput_mib_s` not existing.
- **Known-good solution:** `solution.patch` — `patch -p1 < ../solution.patch`
  from inside `repo/`. Touches `srcpkg/stats.py` and `srcpkg/report.py`.
  Round-trip (apply → pass → revert → fail) is verified.
- **Reset between arms:** restore `repo/` to the committed state. Every arm must
  start from an identical fixture or the A/B is meaningless (`PLAN.md` §6,
  "same weights, both arms" — the same applies to the starting tree).
- **Do not commit a solved `repo/`.** If `python3 -m unittest discover -s tests
  -t .` passes on a clean checkout, the fixture has been contaminated and the
  baseline failure signature in `oracle.json` is void.
