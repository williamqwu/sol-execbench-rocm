#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The JSON API is not linked from the header. It is still served.

A decision about the *page*, not about the API: nothing consumes `/api/v1`
locally today, and `/api/docs` is Swagger UI — its own bundle, the whole schema
expanded — which on the small public host is the heaviest page on the site and
was reachable from every other page, by every crawler.

So the nav entry is behind `SOLBENCH_API_NAV=1`, and everything else is exactly
as it was. This file exists to keep those two facts from drifting into each
other: an unlinked API is one commit away from an API somebody deletes because
"nothing links to it".
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")


def test_the_header_does_not_link_the_api_by_default(client):
    body = client.get("/").text
    nav = body.split("<nav>", 1)[1].split("</nav>", 1)[0]
    assert "/api/docs" not in nav
    assert ">API<" not in nav


def test_the_flag_puts_it_back(monkeypatch, app_mod, client):
    monkeypatch.setitem(app_mod.templates.env.globals, "api_nav", True)
    nav = client.get("/").text.split("<nav>", 1)[1].split("</nav>", 1)[0]
    assert '<a href="/api/docs">API</a>' in nav


def test_the_schema_browser_still_serves(client):
    assert client.get("/api/docs").status_code == 200


def test_the_openapi_schema_is_still_generated(client):
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/api/v1/leaderboard" in paths
    assert "/api/v1/problems/{key}" in paths


def test_every_v1_route_still_answers(client):
    for url in ("/api/v1/parts", "/api/v1/stats", "/api/v1/leaderboard",
                "/api/v1/problems", "/api/v1/problems/L1__001_alpha",
                "/api/v1/submissions/agent-alpha", "/healthz"):
        assert client.get(url).status_code == 200, url
