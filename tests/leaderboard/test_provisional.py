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
            validation_note,evidence,kernel_source,kernel_lines,kernel_sha256,
            kernel_bytes,artifact_id,provenance_json,selected)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                "j-old", sub_id, "t-1", "solbench/L1__001_alpha",
                "L1__001_alpha", "GLM-5.2-local",
                "2026-09-01T00:00:00+00:00",
                "2026-09-01T01:00:00+00:00", 2, "0002",
                "All local workloads pass at 2x.",
                "kernel_and_validation_note",
                "# old\ndef run(x):\n    return x\n", 3, "1" * 64, 100,
                "old-kernel", "{}", 0,
            ),
            (
                "j-new", sub_id, "t-1", "solbench/L1__001_alpha",
                "L1__001_alpha", "GLM-5.2-local",
                "2026-09-02T00:00:00+00:00",
                "2026-09-02T01:00:00+00:00", 3, "0003",
                "All local workloads pass at 3x.",
                "kernel_and_validation_note",
                "# new\ndef run(x):\n    return x\n", 3, "2" * 64, 120,
                "new-kernel", "{}", 1,
            ),
        ],
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
            "url": f"/submissions/{SLUG}?part=MI350X",
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


def test_provisional_submission_keeps_every_job_and_source_separate(client, board):
    add_provisional(board)
    detail = client.get(f"/api/v1/submissions/{SLUG}").json()
    jobs = detail["provisional_jobs"]
    assert [row["job_id"] for row in jobs] == ["j-old", "j-new"]
    assert [row["selected"] for row in jobs] == [0, 1]
    assert detail["provisional_sources"] == 2
    assert all("part=MI350X" in row["url"] for row in jobs)
    assert all("part=MI350X" in row["source_url"] for row in jobs)
    old_source = client.get(jobs[0]["source_url"])
    new_source = client.get(jobs[1]["source_url"])
    assert old_source.text.startswith("# old")
    assert new_source.text.startswith("# new")
    assert old_source.headers["content-type"].startswith("text/plain")
    formal = client.get(
        f"/api/v1/submissions/{SLUG}/problems/L1__001_alpha")
    assert formal.status_code == 404
    assert "no authoritative run detail" in formal.text
    assert client.get(
        f"/submissions/{SLUG}/problems/L1__001_alpha").status_code == 404

    page = client.get(f"/submissions/{SLUG}").text
    assert "Retained KDA job evidence" in page
    assert "All local workloads pass at 2x." in page
    assert "All local workloads pass at 3x." in page
    assert page.count(">source</a>") == 2
    assert "contribution to benchmark" not in page


def test_code_rollout_can_still_read_a_pre_provisional_database(client, board):
    board.write("DROP TABLE provisional_job")
    assert client.get("/api/v1/provisional").json() == []
    assert client.get("/").status_code == 200


def test_previous_provisional_schema_degrades_without_a_500(client, board):
    add_provisional(board)
    board.write("ALTER TABLE provisional_job DROP COLUMN kernel_source")
    board.write("ALTER TABLE provisional_job DROP COLUMN kernel_lines")

    listing = client.get("/api/v1/provisional")
    detail = client.get(f"/api/v1/submissions/{SLUG}")
    source = client.get("/api/v1/provisional/jobs/j-new/kernel")
    assert listing.status_code == detail.status_code == 200
    assert detail.json()["provisional_sources"] == 0
    assert all(job["source_url"] is None
               for job in detail.json()["provisional_jobs"])
    assert source.status_code == 404
    assert client.get("/").status_code == 200
    page = client.get(f"/submissions/{SLUG}")
    assert page.status_code == 200
    assert "contribute zero" not in page.text
    assert "Submitted, but never measured" not in page.text


def test_part_switch_names_provisional_evidence_instead_of_zero_results(
        client, board):
    board.add("MI355X")
    board.write("DELETE FROM result", part="MI355X")
    add_provisional(board, "MI355X")
    response = client.get("/api/v1/parts?part=MI355X")
    mi355 = next(row for row in response.json() if row["name"] == "MI355X")
    assert mi355["n_results"] == 0
    assert mi355["n_provisional"] == 2
    row = client.get("/api/v1/provisional?part=MI355X").json()[0]
    assert row["url"].endswith("?part=MI355X")
    detail = client.get(
        f"/api/v1/submissions/{SLUG}?part=MI355X").json()
    assert all(job["source_url"].endswith("?part=MI355X")
               for job in detail["provisional_jobs"])
    assert "2 provisional" in client.get("/?part=MI355X").text
