# SPDX-License-Identifier: Apache-2.0
"""Provisional rows are visible evidence and impossible to rank."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")


SLUG = "provisional-amdpilot-v2-glm-5-2-local"


def add_provisional(board, part: str = "MI350X") -> None:
    conn = sqlite3.connect(board.path(part))
    sub_id = conn.execute(
        """INSERT INTO submission
           (slug,name,kind,author,model,created_utc,notes,provenance_json,
            board_visible,exclusion_reason,part,depth_note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            SLUG,
            "AMDPilot v2 · GLM-5.2-local",
            "provisional",
            "AMDPilot v2 KDA internal workflow",
            "GLM-5.2-local",
            "2026-09-03T00:00:00+00:00",
            "Local validation only.",
            "{}",
            0,
            "Provisional and unranked.",
            part,
            "No authoritative timing.",
        ),
    ).lastrowid
    conn.executemany(
        """INSERT INTO provisional_job
           (job_id,submission_id,task_id,task_name,problem_key,model,
            created_utc,finished_utc,submission_n,submission_name,
            validation_note,evidence,kernel_sha256,kernel_bytes,artifact_id,
            provenance_json,selected)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                "j-old", sub_id, "t-1", "solbench/L1__001_alpha",
                "L1__001_alpha", "GLM-5.2-local",
                "2026-09-01T00:00:00+00:00",
                "2026-09-01T01:00:00+00:00", 2, "0002",
                "All local workloads pass at 2x.",
                "kernel_and_validation_note", "1" * 64, 100,
                "old-kernel", "{}", 0,
            ),
            (
                "j-new", sub_id, "t-1", "solbench/L1__001_alpha",
                "L1__001_alpha", "GLM-5.2-local",
                "2026-09-02T00:00:00+00:00",
                "2026-09-02T01:00:00+00:00", 3, "0003",
                "All local workloads pass at 3x.",
                "kernel_and_validation_note", "2" * 64, 120,
                "new-kernel", "{}", 1,
            ),
        ],
    )
    conn.execute(
        """INSERT INTO run_kernel
           (submission_id,problem_key,source,n_lines,sha256,retime_ok,retime_error)
           VALUES (?,?,?,?,?,NULL,?)""",
        (
            sub_id, "L1__001_alpha", "def run(x):\n    return x\n", 2,
            "2" * 64, "Provisional local validation only.",
        ),
    )
    conn.commit()
    conn.close()


def test_provisional_rows_do_not_change_the_formal_leaderboard(client, board):
    before = client.get("/api/v1/leaderboard").json()
    add_provisional(board)
    after = client.get("/api/v1/leaderboard").json()
    assert after == before
    assert SLUG not in {row["slug"] for row in after}


def test_provisional_api_has_no_score_or_rank(client, board):
    add_provisional(board)
    rows = client.get("/api/v1/provisional").json()
    assert rows == [
        {
            "slug": SLUG,
            "name": "AMDPilot v2 · GLM-5.2-local",
            "model": "GLM-5.2-local",
            "jobs": 2,
            "problems": 1,
            "kernels": 1,
            "latest_utc": "2026-09-02T01:00:00+00:00",
            "evidence_tier": "provisional",
            "ranked": False,
            "url": f"/submissions/{SLUG}",
        }
    ]
    assert not ({"score", "rank"} & set(rows[0]))
    assert client.get("/api/v1/provisional?category=L2").json() == []


def test_index_renders_a_separate_unranked_table(client, board):
    add_provisional(board)
    page = client.get("/").text
    assert "AMDPilot v2 provisional results" in page
    assert "excluded from both rankings" in page
    assert "AMDPilot v2 · GLM-5.2-local" in page
    assert "tag-provisional" in page


def test_provisional_submission_keeps_every_job_and_one_source(client, board):
    add_provisional(board)
    detail = client.get(f"/api/v1/submissions/{SLUG}").json()
    jobs = detail["provisional_jobs"]
    assert [row["job_id"] for row in jobs] == ["j-old", "j-new"]
    assert [row["selected"] for row in jobs] == [0, 1]

    page = client.get(f"/submissions/{SLUG}").text
    assert "Retained KDA job evidence" in page
    assert "All local workloads pass at 2x." in page
    assert "All local workloads pass at 3x." in page
    assert page.count(">source</a>") == 1


def test_code_rollout_can_still_read_a_pre_provisional_database(client, board):
    board.write("DROP TABLE provisional_job")
    assert client.get("/api/v1/provisional").json() == []
    assert client.get("/").status_code == 200


def test_part_switch_names_provisional_evidence_instead_of_zero_results(
        client, board):
    board.add("MI355X")
    board.write("DELETE FROM result", part="MI355X")
    add_provisional(board, "MI355X")
    response = client.get("/api/v1/parts?part=MI355X")
    mi355 = next(row for row in response.json() if row["name"] == "MI355X")
    assert mi355["n_results"] == 0
    assert mi355["n_provisional"] == 2
    assert "2 provisional" in client.get("/?part=MI355X").text
