"""``kandy-sr --resolution-scale`` keeps the fractional 2.25.

What & why: cyclopts coerces a token for the mixed ``Literal[2, 2.25, 4]``
through its int members, so ``--resolution-scale 2.25`` silently became
``2`` and the run took the plain x2 path. The CLI now reads the token as a
number and lets the literal validation reject anything else.
How: ``app.parse_args`` on the ``from-video`` command, no models.
Corner cases: each supported scale; an unsupported one is refused.
"""

from __future__ import annotations

import pytest

from kandinsky_sr import cli as sr_cli


@pytest.mark.parametrize(("token", "expected"), [("2.25", 2.25), ("2", 2.0), ("4", 4.0)])
def test_resolution_scale_token_is_read_as_a_number(token: str, expected: float) -> None:
    _command, bound, _ignored = sr_cli.app.parse_args(
        ["from-video", "--input", "clip.mp4", "--resolution-scale", token], print_error=False, exit_on_error=False
    )
    assert bound.arguments["resolution_scale"] == expected


def test_unsupported_resolution_scale_is_refused() -> None:
    with pytest.raises(Exception, match="2, 2.25 or 4"):
        sr_cli.app.parse_args(
            ["from-video", "--input", "clip.mp4", "--resolution-scale", "3"], print_error=False, exit_on_error=False
        )
