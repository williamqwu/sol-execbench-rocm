#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A board built without the dataset says so, once, instead of everywhere.

`data/` is gitignored and does not travel with the repo, so a deploy that
clones and runs `ingest.py` gets every MEASURED number — those come from the
manifest and the run artifacts — and none of the problem definitions. What is
lost is the description, the reference implementation, the inputs, the outputs,
the axes, each workload's parameters and each workload's dataset number.

Before this, each of those failed in its own way and none of them named the
cause: the problem listing printed the literal string `None` under all 235
names, the inputs and outputs tables rendered as headers with no rows, the
reference pane was empty, and every workload's parameters read "none declared"
— a sentence about the dataset that was, in that situation, false. The one
thing a reader needed to know appeared nowhere.

This is the deployed board's actual state (solbench.matrixforge.org), not a
hypothetical, which is why it is a fixture and not a comment.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

KEY = "L1__001_alpha"


@pytest.fixture()
def bare_client(board, app_mod):
    """The fixture board with everything the dataset supplies stripped out."""
    from fastapi.testclient import TestClient
    conn = sqlite3.connect(board.path("MI350X"))
    conn.execute("UPDATE problem SET description=NULL, reference=NULL, "
                 "axes_json='{}', inputs_json='{}', outputs_json='{}'")
    conn.execute("UPDATE workload SET axes_json='{}', dataset_index=NULL")
    conn.execute("INSERT OR REPLACE INTO meta(key,value) "
                 "VALUES ('dataset_problems','0')")
    conn.commit()
    conn.close()
    return TestClient(app_mod.app)


def test_the_banner_names_the_cause_and_the_fix(bare_client):
    body = bare_client.get(f"/problems/{KEY}").text
    assert "built without the benchmark dataset" in body
    assert "materialize_dataset.py" in body


def test_the_banner_says_the_measurements_are_unaffected(bare_client):
    """Otherwise it reads as "this board is broken", and the scores are fine —
    they never came from the dataset."""
    body = bare_client.get("/").text
    banner = body.split("built without the benchmark dataset", 1)[1][:600]
    assert "unaffected" in banner


def test_no_page_prints_the_string_None(bare_client):
    """`{{ problem.description }}` on a Python None renders four characters."""
    for url in ("/problems", f"/problems/{KEY}"):
        body = bare_client.get(url).text
        assert ">None<" not in body, url
        assert "trunc\">None" not in body, url


def test_a_missing_parameter_is_not_reported_as_an_absent_one(bare_client):
    """"none declared" is a claim about the dataset. With no dataset to read,
    the only true statement is that nothing was read."""
    body = bare_client.get(f"/problems/{KEY}").text
    assert "none declared" not in body
    assert "not available" in body


def test_the_dataset_number_is_absent_rather_than_invented(bare_client):
    """The `#` column is the dataset's own position, which is what makes it
    line up with upstream's listing of the same problem. A loop counter over
    uuid-sorted rows would look identical and mean a different workload."""
    body = bare_client.get(f"/problems/{KEY}").text
    rows = body.split('id="wl-table"', 1)[1].split("</table>", 1)[0]
    assert '<td class="r mono sm dim-n">—</td>' in rows
    assert '<td class="r mono sm dim-n">1</td>' not in rows


def test_the_reference_pane_says_why_it_is_empty(bare_client):
    body = bare_client.get(f"/problems/{KEY}").text
    after = body.split('<h2 id="reference">', 1)[1].split("<h2", 1)[0]
    assert "srcclamp" not in after
    assert "Not available" in after


def test_a_board_that_predates_the_key_raises_no_alarm(board, app_mod):
    """`dataset_problems` absent means "built by an older ingest", not zero."""
    from fastapi.testclient import TestClient
    conn = sqlite3.connect(board.path("MI350X"))
    conn.execute("DELETE FROM meta WHERE key='dataset_problems'")
    conn.commit()
    conn.close()
    assert "built without the benchmark dataset" not in TestClient(
        app_mod.app).get("/").text


def test_the_banner_is_absent_when_the_dataset_is_there(client):
    assert "built without the benchmark dataset" not in client.get("/").text


# --------------------------------------------------------------------------
# transcripts: the same class of absence, decided the other way
# --------------------------------------------------------------------------
# They are NOT tracked in git -- 78 MB for two runs, and their provenance
# records carry gateway hostnames and key prefixes -- so every deploy indexes
# none of them. That is a deliberate storage decision, not a defect, and the
# page has to say which of the two absences it is looking at rather than
# dropping the section.

def test_the_transcript_section_is_always_there(client):
    """It used to render only when a transcript existed, so a board with none
    -- which is every deploy -- looked like a run that recorded none."""
    body = client.get("/submissions/agent-alpha/problems/L1__001_alpha").text
    assert '<h2 id="transcript">' in body


def test_the_empty_transcript_says_which_absence_it_is(client):
    body = client.get("/submissions/agent-alpha/problems/L1__001_alpha").text
    after = body.split('<h2 id="transcript">', 1)[1].split("<h2", 1)[0]
    assert "No transcript is indexed" in after
    assert "not tracked in git" in after


def test_the_nav_still_matches_the_headings(client):
    """The entry is unconditional now because the heading is. test_sidenav
    checks this globally; this pins the pair that just changed."""
    body = client.get("/submissions/agent-alpha/problems/L1__001_alpha").text
    nav = body.split("</aside>", 1)[0]
    assert 'href="#transcript"' in nav
