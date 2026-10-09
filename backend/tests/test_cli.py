import pytest

from charlab import __version__
from charlab.cli import main


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"charlab {__version__}"


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage: charlab" in capsys.readouterr().out
