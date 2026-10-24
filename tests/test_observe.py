"""Tests for the observability layer.

The invariants that matter: it never requires a credential, it never publishes, and it reports
where a credential came from without ever revealing it.
"""

from __future__ import annotations

import pytest

from v2d.observe import credentials
from v2d.observe.hf_store import open_store
from v2d.observe.wandb_run import start_run


def test_a_run_without_wandb_is_a_working_no_op(monkeypatch):
    monkeypatch.setattr("v2d.observe.wandb_run.wandb_available", lambda: False)
    with start_run("test", {"a": 1}) as run:
        assert not run.active
        assert run.reason
        run.log({"loss": 1.0})  # must not raise
        run.summary({"auc": 0.5})
        run.log_cost(1.0, 0.68)
        assert run.url is None


def test_a_public_store_is_refused_outright():
    with pytest.raises(ValueError, match="private-only"):
        open_store("anything", private=False)


def test_a_store_without_a_credential_is_inert(monkeypatch):
    monkeypatch.setattr("v2d.observe.hf_store.hf_available", lambda: False)
    store = open_store("v2d-track2")
    assert not store.active
    assert store.upload.__self__ is store
    assert store.upload(__import__("pathlib").Path("/nonexistent")) is None
    assert store.url is None
    assert "credential" in store.reason


def test_credentials_are_located_not_revealed(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "super-secret-value")
    found = credentials.which_credentials()
    assert found["wandb"] == "env:WANDB_API_KEY"
    assert "super-secret-value" not in str(found)


def test_env_takes_precedence_over_netrc(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    assert credentials.which_credentials()["huggingface"] == "env:HF_TOKEN"


def test_absent_everything_reports_none(monkeypatch, tmp_path):
    for name in (*credentials.WANDB_ENV, *credentials.HF_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(credentials, "_netrc_host", lambda: None)
    monkeypatch.setattr(credentials, "_hf_token_file", lambda: None)
    found = credentials.which_credentials()
    assert found == {"wandb": None, "huggingface": None}
