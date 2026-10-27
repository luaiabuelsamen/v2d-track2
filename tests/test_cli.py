"""The CLI's one piece of logic: how a forwarded stage's exit is reported.

Stages refuse with `SystemExit("why")`. Passing that string to `int()` raised a ValueError and
buried the reason in a traceback, which is how a clean "still awaiting B4x e26" refusal came back
looking like a crash.
"""

from __future__ import annotations

from v2d.cli import _forward, main


def test_a_string_refusal_is_printed_and_reported_as_exit_two(capsys, monkeypatch):
    import v2d.submit.wave2 as wave2

    monkeypatch.setattr(wave2, "main", lambda argv: (_ for _ in ()).throw(SystemExit("nope")))
    assert _forward("v2d.submit.wave2", []) == 2
    assert "nope" in capsys.readouterr().err


def test_an_integer_code_is_passed_through(monkeypatch):
    import v2d.submit.wave2 as wave2

    monkeypatch.setattr(wave2, "main", lambda argv: (_ for _ in ()).throw(SystemExit(3)))
    assert _forward("v2d.submit.wave2", []) == 3


def test_a_clean_return_is_zero(monkeypatch):
    import v2d.submit.wave2 as wave2

    monkeypatch.setattr(wave2, "main", lambda argv: None)
    assert _forward("v2d.submit.wave2", []) == 0


def test_the_wave_two_stager_is_reachable_by_name(capsys):
    assert main(["stage-entry-wave2", "--help"]) == 0
    assert "--adoption" in capsys.readouterr().out


def test_every_shell_wrapper_propagates_the_cli_exit_code():
    """A wrapper that calls `main()` and drops its return value reports every failure as success.

    All four scripts did, and it was masked: the old `_forward` turned a string refusal into a
    ValueError, which exited non-zero by accident, so only integer refusals were swallowed. Once
    refusals became clean exit codes the stage ran on past one, which is how this was found.
    """
    from pathlib import Path

    for script in sorted(Path("brev").glob("*.sh")):
        text = script.read_text()
        if "from v2d.cli import main" not in text:
            continue
        assert "raise SystemExit(main())" in text, f"{script} drops the CLI's exit code"
        assert "import main; main()" not in text, f"{script} drops the CLI's exit code"
