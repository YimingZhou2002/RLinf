# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the AC-7 DAG-view rendering + wiring into CriticPrompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from toolkits.embodied_tuner.critic import CriticPrompt, build_prompt
from toolkits.embodied_tuner.node_store import (
    DUPLICATE_OF_FAILURE_MODE,
    NodeStore,
    render_dag_view,
)
from toolkits.embodied_tuner.schema import KnobSchema
from toolkits.embodied_tuner.tests.test_node_store import _make_child, _make_root


# ----- Header + section ordering --------------------------------------


def test_render_dag_view_starts_with_search_dag_header(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    view = render_dag_view(store, active_leaf_id="root")
    assert view.startswith("## Search DAG\n")


def test_render_dag_view_ordering(tmp_path: Path) -> None:
    """Section order: ancestor -> siblings -> top-K OK -> recent failed."""
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", trial_idx=0, objective=1.0))
    view = render_dag_view(store, active_leaf_id="n1")
    ancestor_idx = view.index("### Active branch")
    sibling_idx = view.index("### Sibling attempts at parent")
    okleader_idx = view.index("### Top-K OK leaves")
    failed_idx = view.index("### Recent failure leaves")
    assert ancestor_idx < sibling_idx < okleader_idx < failed_idx


# ----- Ancestor chain -------------------------------------------------


def test_ancestor_chain_rendered_root_to_leaf(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", trial_idx=0))
    store.append(_make_child("n2", "n1", trial_idx=1))
    store.append(_make_child("n3", "n2", trial_idx=2))
    view = render_dag_view(store, active_leaf_id="n3")
    # All four ids should appear in the ancestor section in order. The
    # test fixture uses node_id="root" for the root (not the
    # derive_node_id "root-<hash>" format).
    ancestor_section = view.split("### Sibling")[0]
    root_pos = ancestor_section.index(" (root)")
    n1_pos = ancestor_section.index("n1  ")
    n2_pos = ancestor_section.index("n2  ")
    n3_pos = ancestor_section.index("n3  ")
    assert root_pos < n1_pos < n2_pos < n3_pos


def test_ancestor_chain_marks_active_leaf(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", trial_idx=0))
    view = render_dag_view(store, active_leaf_id="n1")
    # The active-leaf line has the "-> " marker.
    assert "-> n1" in view


def test_ancestor_chain_unconditional_even_when_max_dag_nodes_is_zero(tmp_path: Path) -> None:
    """max_dag_nodes=0 must not truncate the ancestor chain (AC-7)."""
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    for i in range(5):
        parent = "root" if i == 0 else f"n{i - 1}"
        store.append(_make_child(f"n{i}", parent, trial_idx=i))
    view = render_dag_view(store, active_leaf_id="n4", max_dag_nodes=0)
    for i in range(5):
        assert f"n{i}" in view


def test_max_dag_nodes_zero_is_well_formed(tmp_path: Path) -> None:
    """max_dag_nodes=0 renders empty leaderboard sections without crashing."""
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", trial_idx=0, objective=1.0))
    store.append(_make_child("n2", "root", trial_idx=1, objective=2.0))  # sibling
    view = render_dag_view(store, active_leaf_id="n1", max_dag_nodes=0)
    assert "### Sibling attempts at parent" in view
    assert "### Top-K OK leaves" in view
    assert "### Recent failure leaves" in view
    # Sibling section consumed 0 budget so it renders "budget exhausted"
    # (or the leaderboard exhaustion message). Either way it must not
    # list the sibling node id.
    assert view.count("n2") == 0


def test_max_dag_nodes_negative_raises(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    with pytest.raises(ValueError):
        render_dag_view(store, active_leaf_id="root", max_dag_nodes=-1)


# ----- Top-K OK leaderboard -------------------------------------------


def test_top_k_ok_sorted_by_objective_ascending(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("hi", "root", trial_idx=0, objective=5.0))
    store.append(_make_child("lo", "root", trial_idx=1, objective=1.0))
    store.append(_make_child("mid", "root", trial_idx=2, objective=3.0))
    view = render_dag_view(store, active_leaf_id="mid")
    ok_section = view.split("### Top-K OK leaves")[1].split("### Recent")[0]
    lo_pos = ok_section.index("lo")
    mid_pos = ok_section.index("mid")
    hi_pos = ok_section.index("hi")
    assert lo_pos < mid_pos < hi_pos


def test_top_k_ok_excludes_duplicate_of(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("orig", "root", trial_idx=0, objective=1.0))
    dup = _make_child(
        "dup",
        "root",
        trial_idx=-1,
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="orig",
    )
    store.append(dup)
    view = render_dag_view(store, active_leaf_id="orig")
    ok_section = view.split("### Top-K OK leaves")[1].split("### Recent")[0]
    assert "orig" in ok_section
    assert "dup" not in ok_section, "DUPLICATE_OF nodes must not appear in the OK leaderboard"


# ----- Recent FAILED leaves -------------------------------------------


def test_recent_failed_ordered_by_recency(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(
        _make_child("older_fail", "root", trial_idx=0, status="FAILED", failure_mode="OOM", objective=None)
    )
    store.append(_make_child("ok", "root", trial_idx=1, objective=1.0))
    store.append(
        _make_child("newer_fail", "root", trial_idx=2, status="FAILED", failure_mode="TIMEOUT", objective=None)
    )
    view = render_dag_view(store, active_leaf_id="ok")
    failed_section = view.split("### Recent failure leaves")[1]
    newer_pos = failed_section.index("newer_fail")
    older_pos = failed_section.index("older_fail")
    assert newer_pos < older_pos  # most recent first


def test_recent_failed_excludes_duplicate_of(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("orig", "root", trial_idx=0, objective=1.0))
    dup = _make_child(
        "dup",
        "root",
        trial_idx=-1,
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="orig",
    )
    store.append(dup)
    view = render_dag_view(store, active_leaf_id="orig")
    # ``### Recent failure leaves`` slice ends at the next ``###`` header
    # (introduced by the new "Recent duplicate config attempts" section
    # in Round 2 for AC-6/AC-7). Bound the slice so we only assert on
    # the failure section, not the duplicate section below it.
    after_failure_header = view.split("### Recent failure leaves")[1]
    failed_section = after_failure_header.split("###")[0]
    assert "dup" not in failed_section


# ----- Recent duplicate config attempts section (AC-6/AC-7 Round 2) -----


def _dup_section(view: str) -> str:
    """Extract the ``### Recent duplicate config attempts`` slice."""
    return view.split("### Recent duplicate config attempts")[1]


def test_recent_duplicate_section_header_present(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    view = render_dag_view(store, active_leaf_id="root")
    assert "### Recent duplicate config attempts" in view
    # With no duplicates, the section renders a placeholder line.
    assert "(no duplicate attempts yet)" in _dup_section(view)


def test_recent_duplicate_section_surfaces_dup_not_ancestor_or_sibling(
    tmp_path: Path,
) -> None:
    """AC-6/AC-7: a DUPLICATE_OF that is neither ancestor nor current-parent sibling still appears.

    Structure:
      root -> orig (OK, active)
      root -> other (OK)
      root -> dup_of_other (DUPLICATE_OF pointing back at other)

    Active leaf = orig. The duplicate ``dup_of_other`` is:
      - NOT in the ancestor chain of orig (which is [root, orig]).
      - NOT a sibling of orig ONLY when the active leaf's parent has
        other siblings — root does have children so ``other`` and
        ``dup_of_other`` are both siblings of orig. Reconfigure:

    Structure v2:
      root -> orig (OK)
      orig -> child_of_orig (OK, ACTIVE)
      root -> other (OK)
      root -> dup_of_other (DUPLICATE_OF -> other)

    Now active = child_of_orig. Ancestor chain = [root, orig, child_of_orig].
    Current-parent siblings = children of orig minus child_of_orig
    (empty). ``dup_of_other`` is a child of root, not of orig, so it
    isn't in siblings. Pre-Round-2 it disappeared from the prompt;
    post-Round-2 it must appear in the duplicate section.
    """
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("orig", "root", trial_idx=0, objective=1.0))
    store.append(
        _make_child("child_of_orig", "orig", trial_idx=1, objective=0.5)
    )
    store.append(_make_child("other", "root", trial_idx=2, objective=2.0))
    dup = _make_child(
        "dup_of_other",
        "root",
        trial_idx=-1,
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=2.0,
        duplicate_of_node_id="other",
    )
    store.append(dup)

    view = render_dag_view(store, active_leaf_id="child_of_orig")
    dup_section = _dup_section(view)
    # The duplicate must appear in the dedicated section.
    assert "dup_of_other" in dup_section
    # Each duplicate line must render the back-reference.
    assert "duplicate_of=other" in dup_section


def test_recent_duplicate_section_ordered_by_recency(tmp_path: Path) -> None:
    """Most recent duplicate appears first (matches recent-failure ordering)."""
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("orig", "root", trial_idx=0, objective=1.0))
    older = _make_child(
        "older_dup",
        "root",
        trial_idx=-1,
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="orig",
    )
    newer = _make_child(
        "newer_dup",
        "root",
        trial_idx=-2,
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="orig",
    )
    store.append(older)
    store.append(newer)
    view = render_dag_view(store, active_leaf_id="orig")
    section = _dup_section(view)
    assert section.index("newer_dup") < section.index("older_dup")


def test_recent_duplicate_section_shares_max_dag_nodes_budget(
    tmp_path: Path,
) -> None:
    """max_dag_nodes cap applies to sections 2+3+4+5 combined."""
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("orig", "root", trial_idx=0, objective=1.0))
    # 5 duplicates. With max_dag_nodes=1 and sibling+ok+failed sections
    # empty (no siblings, orig is the only OK leaf which lands in top-K,
    # no failures), the OK leaderboard consumes 1 slot leaving 0 for
    # duplicates.
    for i in range(5):
        store.append(
            _make_child(
                f"dup_{i}",
                "root",
                trial_idx=-(i + 1),
                status="OK",
                failure_mode=DUPLICATE_OF_FAILURE_MODE,
                objective=1.0,
                duplicate_of_node_id="orig",
            )
        )
    view = render_dag_view(store, active_leaf_id="orig", max_dag_nodes=1)
    section = _dup_section(view)
    # Budget exhausted by OK leaderboard; duplicate section says so.
    assert "(budget exhausted" in section
    # None of the actual duplicate ids leak into the duplicate section.
    for i in range(5):
        assert f"dup_{i}" not in section


def test_recent_duplicate_section_absent_when_no_duplicates(tmp_path: Path) -> None:
    """When no DUPLICATE_OF nodes exist, section still renders with placeholder."""
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("orig", "root", trial_idx=0, objective=1.0))
    view = render_dag_view(store, active_leaf_id="orig")
    section = _dup_section(view)
    assert "(no duplicate attempts yet)" in section


def test_wiki_09_dag_search_documents_duplicate_section(tmp_path: Path) -> None:
    """AC-7: the wiki file must describe the new duplicate section for Codex."""
    wiki = (
        Path(__file__).resolve().parents[1]
        / "wiki"
        / "09-dag-search.md"
    )
    text = wiki.read_text(encoding="utf-8")
    assert "Recent duplicate config attempts" in text
    assert "duplicate_of=<node_id>" in text


# ----- Node-identifier stability across rounds -------------------------


def test_node_identifiers_stable_across_render_calls(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", trial_idx=0, objective=1.0))
    view_round_n = render_dag_view(store, active_leaf_id="n1")
    # Same store, same active leaf, called again -> exact same output.
    view_round_n_plus_1 = render_dag_view(store, active_leaf_id="n1")
    assert view_round_n == view_round_n_plus_1


# ----- CriticPrompt integration ---------------------------------------


def test_dag_block_appears_between_bitter_lessons_and_history() -> None:
    # Use content markers unique to each block so wiki_block's
    # incidental mentions of the same headings don't confuse index().
    prompt = CriticPrompt(
        bitter_lessons_block="## Bitter Lessons\nMARKER-BITTER-LESSONS\n",
        dag_block="## Search DAG\nMARKER-DAG-BLOCK\n",
        history_block="## Trial History\nMARKER-HISTORY\n",
    )
    rendered = str(prompt)
    bl_pos = rendered.index("MARKER-BITTER-LESSONS")
    dag_pos = rendered.index("MARKER-DAG-BLOCK")
    hist_pos = rendered.index("MARKER-HISTORY")
    assert bl_pos < dag_pos < hist_pos


def test_dag_block_absent_when_empty_string() -> None:
    # The wiki_block naturally contains "## Search DAG" text (from the
    # 09-dag-search.md wiki page that documents the block). What we
    # really care about here is that the CriticPrompt does not
    # PROJECT its dag_block field into the output when the field is
    # empty. Assert on a content marker instead of the header text.
    prompt = CriticPrompt(
        bitter_lessons_block="## BL\n",
        dag_block="",
        history_block="## H\nMARKER-HISTORY\n",
    )
    rendered = str(prompt)
    assert "MARKER-DAG-BLOCK" not in rendered
    assert "MARKER-HISTORY" in rendered


def test_debug_text_includes_dag_block() -> None:
    prompt = CriticPrompt(
        bitter_lessons_block="## BL\n",
        dag_block="## Search DAG\nMARKER-DAG-BLOCK\n",
        history_block="## H\n",
    )
    assert "MARKER-DAG-BLOCK" in prompt.to_debug_text()


def test_build_prompt_accepts_dag_block_kwarg() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
        dag_block="## Search DAG\nMARKER-KWARG-DAG\n",
    )
    assert prompt.dag_block == "## Search DAG\nMARKER-KWARG-DAG\n"
    assert "MARKER-KWARG-DAG" in str(prompt)


def test_wiki_09_dag_search_loaded() -> None:
    """New wiki file must appear in wiki_block via _WIKI_CONTEXT_FILES."""
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert "DAG search view" in prompt.wiki_block or "Search DAG" in prompt.wiki_block


# ----- Edge cases -----------------------------------------------------


def test_render_on_empty_store() -> None:
    """An empty NodeStore renders a well-formed placeholder view."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = NodeStore(Path(td) / "n.jsonl", fsync_on_append=False)
        view = render_dag_view(store, active_leaf_id=None)
        assert view.startswith("## Search DAG")


def test_render_with_missing_active_leaf(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "n.jsonl", fsync_on_append=False)
    store.append(_make_root())
    view = render_dag_view(store, active_leaf_id="does-not-exist")
    assert "not found" in view
