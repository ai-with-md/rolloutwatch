import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="rolloutwatch: did my change make things worse?",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _(mo):
    import sys

    if sys.platform == "emscripten":
        # In the browser, install the library from the wheel published with this page.
        # wheel.txt holds the wheel's file name, so a new version needs no change here.
        import micropip
        from pyodide.http import pyfetch

        _public = mo.notebook_location() / "public"
        _wheel = (await (await pyfetch(str(_public / "wheel.txt"))).string()).strip()
        await micropip.install(str(_public / _wheel))
    import rolloutwatch as rw
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7, 3.2),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    INK, ACCENT, MUTED, BAD = "#1f2937", "#b45309", "#9ca3af", "#b91c1c"
    return ACCENT, BAD, MUTED, plt, rw


@app.cell
def _(mo):
    mo.md("""
    # Did my change make things worse?

    You changed a prompt or switched models, and graded production traffic is coming in every day.
    These tools tell you, as early as the data allows, whether quality dropped, and you can check as
    often as you like without inflating false alarms. Everything runs in your browser, so nothing is
    sent anywhere. The first load takes a few seconds while Python starts up.

    1. [Why not just run a test every day?](#peeking)
    2. [How long until I know?](#plan)
    3. [Check my rollout](#check)

    The same calculations are in the Python library: `pip install rolloutwatch`.
    """)
    return


@app.cell
def _(mo):
    pk_checks = mo.ui.slider(1, 60, step=1, value=30, label="Daily checks", show_value=True)
    return (pk_checks,)


@app.cell
def _(ACCENT, MUTED, mo, pk_checks, plt, rw):
    _curve = rw.peeking_curve(60, per_check=150, simulations=800)
    _rate = _curve[pk_checks.value - 1]
    _xs = list(range(1, 61))
    _ys = [100 * v for v in _curve]
    _fig, _ax = plt.subplots()
    _ax.plot(_xs, _ys, color=ACCENT, linewidth=2)
    _ax.scatter([pk_checks.value], [100 * _rate], color=ACCENT, zorder=3)
    _ax.axhline(5, color=MUTED, linestyle="--")
    _ax.text(60, 5, " promised", va="bottom", ha="right", color=MUTED, fontsize=8)
    _ax.set_xlabel("Daily checks, stopping at the first alarm")
    _ax.set_ylabel("False alarms (%)")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                """
                <h2 id="peeking">1. Why not just run a test every day?</h2>

                A significance test promises 5% false alarms if you run it once. Run it every day and
                stop at the first alarm, and the promise breaks: with no real change at all, the
                new version gets flagged far more often. The same goes for stopping at the first
                "looks fine".
                """
            ),
            pk_checks,
            mo.hstack(
                [
                    mo.stat(f"{_rate:.0%}", label="False alarms with an ordinary test", caption=f"{pk_checks.value} daily checks, no real change", bordered=True),
                    mo.stat("under 5%", label="With rolloutwatch", caption="however often you check", bordered=True),
                ],
                justify="start",
                wrap=True,
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    pl_base = mo.ui.slider(0.30, 0.98, step=0.01, value=0.82, label="Old pass rate", show_value=True)
    pl_change = mo.ui.slider(-0.15, 0.05, step=0.01, value=-0.04, label="Real change to simulate", show_value=True)
    pl_day = mo.ui.number(10, 1_000_000, step=10, value=300, label="Graded requests per day")
    pl_fixed = mo.ui.switch(value=False, label="All traffic on the new version (fixed baseline)")
    pl_drop = mo.ui.slider(0.01, 0.10, step=0.005, value=0.03, label="Smallest drop worth catching", show_value=True)
    return pl_base, pl_change, pl_day, pl_drop, pl_fixed


@app.cell
def _(mo, pl_base, pl_change, pl_day, pl_drop, pl_fixed, rw):
    _header = mo.md(
        """
        <h2 id="plan">2. How long until I know?</h2>

        Simulates hundreds of rollouts with the same monitor you would run. Side by side means half
        the graded traffic still goes to the old version, which protects against changes in traffic
        over time. A fixed baseline sends everyone to the new version and compares against a pass
        rate you trust from history.
        """
    )
    if not 0 < pl_base.value + pl_change.value < 1:
        _body = mo.callout(mo.md("The old pass rate plus the change must stay between 0% and 100%."), kind="warn")
    else:
        _p = rw.plan_rollout(
            pl_base.value, pl_change.value, int(pl_day.value), drop=pl_drop.value,
            fixed_baseline=pl_fixed.value, simulations=200,
        )
        if _p.median_days is None:
            _cards = [mo.stat("60+ days", label="Typical time to an answer", caption=f"{_p.decided_share:.0%} decided within 60 days", bordered=True)]
        else:
            _cards = [
                mo.stat(f"{_p.median_days:.0f} days", label="Typical time to an answer", caption=f"answer: {_p.expected_status}", bordered=True),
                mo.stat(f"{_p.slow_days:.0f} days", label="Slow rollouts", caption="1 in 10 take this long or more", bordered=True),
                mo.stat(f"{_p.right_share:.0%}", label="Right call", caption="when an answer is given", bordered=True),
            ]
        _body = mo.hstack(_cards, justify="start", wrap=True)
    mo.vstack([_header, mo.hstack([pl_base, pl_change], wrap=True), mo.hstack([pl_day, pl_drop], wrap=True), pl_fixed, _body])
    return


@app.cell
def _(mo):
    _example = [
        ("11-02", 96, 123, 121, 151), ("11-03", 104, 133, 123, 150), ("11-04", 99, 124, 116, 144),
        ("11-05", 113, 155, 130, 151), ("11-06", 124, 153, 100, 115), ("11-07", 107, 147, 117, 139),
        ("11-08", 105, 136, 138, 165), ("11-09", 116, 143, 129, 148), ("11-10", 123, 149, 115, 134),
        ("11-11", 133, 165, 105, 119), ("11-12", 116, 151, 91, 116), ("11-13", 120, 142, 109, 133),
        ("11-14", 85, 123, 120, 143), ("11-15", 115, 156, 101, 115), ("11-16", 89, 124, 111, 141),
        ("11-17", 130, 158, 121, 150), ("11-18", 108, 130, 91, 116), ("11-19", 129, 163, 112, 139),
        ("11-20", 100, 127, 125, 152), ("11-21", 104, 127, 125, 151), ("11-22", 125, 148, 97, 115),
    ]
    ck_text = mo.ui.text_area(
        value="\n".join(", ".join(map(str, r)) for r in _example),
        label="One row per day: label, new passed, new total, old passed, old total (leave out the old columns when using a baseline)",
        rows=8,
        full_width=True,
    )
    ck_use_base = mo.ui.switch(value=False, label="Compare against a fixed baseline")
    ck_base = mo.ui.slider(0.30, 0.98, step=0.01, value=0.83, label="Baseline pass rate", show_value=True)
    ck_drop = mo.ui.slider(0.01, 0.10, step=0.005, value=0.03, label="Smallest drop worth catching", show_value=True)
    return ck_base, ck_drop, ck_text, ck_use_base


@app.cell
def _(ACCENT, BAD, MUTED, ck_base, ck_drop, ck_text, ck_use_base, mo, plt, rw):
    _header = mo.md(
        """
        <h2 id="check">3. Check my rollout</h2>

        Paste each day's graded counts. The box starts with three weeks of a simulated prompt change
        that is really four points worse. The shaded band is valid on every day at once, so the
        first day it clears a line is a decision you can act on.
        """
    )
    try:
        _monitor = rw.QualityMonitor(drop=ck_drop.value, baseline=ck_base.value if ck_use_base.value else None)
        _labels = []
        for _line in ck_text.value.strip().splitlines():
            _parts = [p.strip() for p in _line.split(",") if p.strip()]
            if not _parts:
                continue
            _nums = [int(float(p)) for p in _parts[1:]]
            _labels.append(_parts[0])
            if ck_use_base.value:
                _monitor.update(_nums[0], _nums[1])
            else:
                _monitor.update(_nums[0], _nums[1], _nums[2], _nums[3])
        if not _monitor.history:
            raise ValueError("Paste at least one row.")
        _h = _monitor.history
        _x = list(range(1, len(_h) + 1))
        _fig, _ax = plt.subplots()
        _ax.fill_between(_x, [100 * c.low for c in _h], [100 * c.high for c in _h], color=ACCENT, alpha=0.18, linewidth=0)
        _ax.plot(_x, [100 * c.difference for c in _h], color=ACCENT, linewidth=2)
        _ax.axhline(0, color=MUTED, linewidth=1)
        _ax.axhline(-100 * ck_drop.value, color=BAD, linestyle="--", linewidth=1)
        _ax.set_ylim(max(-25, min(100 * c.low for c in _h[2:] or _h)) - 1, min(25, max(100 * c.high for c in _h[2:] or _h)) + 1)
        _ax.set_xticks(_x[:: max(1, len(_x) // 7)])
        _ax.set_xticklabels([_labels[i - 1] for i in _x[:: max(1, len(_x) // 7)]], fontsize=8)
        _ax.set_ylabel("New minus old (points)")
        _fig.tight_layout()
        _last = _h[-1]
        _kind = {"worse": "danger", "not worse": "success"}.get(_monitor.status, "neutral")
        _verdict = _monitor.report().splitlines()[-1]
        _first = next((c for c in _h if c.status != rw.WATCHING), None)
        if _first is not None:
            _verdict = _verdict.replace(f"check {_first.check}", _labels[_first.check - 1])
        _body = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(f"{_last.difference:+.1%}", label="New minus old so far", caption=f"{_last.new_total:,} new results", bordered=True),
                        mo.stat(f"{_last.low:+.1%} to {_last.high:+.1%}", label="Always-valid interval", caption="holds on every day at once", bordered=True),
                        mo.stat(_monitor.status, label="Decision", bordered=True),
                    ],
                    justify="start",
                    wrap=True,
                ),
                mo.callout(mo.md(_verdict), kind=_kind),
                _fig,
            ]
        )
    except (ValueError, IndexError) as _exc:
        _body = mo.callout(mo.md(f"Could not read the rows: {_exc}"), kind="warn")
    mo.vstack([_header, ck_text, mo.hstack([ck_use_base, ck_base, ck_drop], wrap=True), _body])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    Built with [rolloutwatch](https://github.com/ai-with-md/rolloutwatch), MIT licensed. The method:
    an always-valid confidence sequence for the difference in pass rates, from the normal mixture
    boundary behind the mixture sequential probability ratio test (Johari and others, 2017; Howard
    and others, 2021). Every promise is checked against simulated rollouts in the test suite.
    """)
    return


if __name__ == "__main__":
    app.run()
