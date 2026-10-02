from calign.evals.lite import render_markdown, rule_check


def _row(cfg, recall=0.95, p6=0.9, mention=0.3, mixed=0.3, dev=None):
    row = {
        "config": cfg,
        "quiz": {"recall": {"mean": recall}, "p6": {"mean": p6}},
        "moralchoice_dev": {"alignment": {"mean": 0.8, "ci95_low": 0.7, "ci95_high": 0.9}, "mention_rate": mention},
        "dilemmas": {
            "v1": {"all": {"n": 10, "all_pass": 5, "mixed": 3, "mean_pass_rate": 0.8}, "mention_rate": 0.2},
            "combined": {"mixed_share": mixed},
        },
    }
    if dev is not None:
        row["_dev_items"] = dev
    return row


REF = _row("ref", dev={f"s{i}": 1.0 for i in range(20)})


def test_rule_passes_and_fails_on_each_condition():
    assert rule_check(_row("a"), REF)["passes"] is True
    assert rule_check(_row("a", recall=0.85), REF)["passes"] is False
    assert rule_check(_row("a", p6=0.8), REF)["passes"] is False
    assert rule_check(_row("a", mention=0.04), REF)["passes"] is False
    assert rule_check(_row("a", mixed=0.1), REF)["passes"] is False  # no dev items -> no mc headroom either


def test_rule_headroom_from_moralchoice_dev_below_reference():
    worse = {f"s{i}": 0.5 for i in range(20)}
    c = rule_check(_row("a", mixed=0.1, dev=worse), REF)
    assert c["headroom_mc_dev_below_ref"] is True and c["passes"] is True
    same = rule_check(_row("a", mixed=0.1, dev=dict(REF["_dev_items"])), REF)
    assert same["headroom_mc_dev_below_ref"] is False and same["passes"] is False


def test_rule_missing_observables_give_none():
    row = _row("a")
    row["quiz"] = {"recall": {"mean": None}, "p6": {"mean": None}}
    c = rule_check(row, None)
    assert c["recall_ge_0.9"] is None and c["passes"] is None and c["headroom_mc_dev_below_ref"] is None


def test_render_markdown_names_the_proposal():
    rows = [_row("C2kn@e1", recall=0.6), _row("C2kn@e2")]
    rule = {r["config"]: rule_check(r, REF) for r in rows}
    md = render_markdown(rows, REF, rule, "C2kn@e2")
    assert "C2kn@e2" in md and "ref (ref)" in md and "**C2kn@e2**" in md
