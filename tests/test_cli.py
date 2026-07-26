import pytest

from minispice.__main__ import main

VOLTAGE_DIVIDER = """\
Voltage Divider
V1 in 0 5
R1 in out 1k
R2 out 0 1k
.end
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_cli_solves_voltage_divider(tmp_path, capsys):
    netlist = _write(tmp_path, "divider.cir", VOLTAGE_DIVIDER)

    exit_code = main([str(netlist)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Parsing netlist" in out
    assert "Parsed circuit 'Voltage Divider': 3 component(s), 2 node(s)" in out
    assert "Solving DC operating point" in out
    assert "Node Voltages" in out
    assert "5 V" in out
    assert "2.5 V" in out
    assert "Resistor Currents" in out
    assert "2.5 mA" in out
    assert "Resistor Power Dissipation" in out
    assert "6.25 mW" in out


def test_cli_reports_missing_file(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.cir"

    exit_code = main([str(missing)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not found" in err
    assert str(missing) in err


def test_cli_reports_malformed_netlist(tmp_path, capsys):
    netlist = _write(tmp_path, "bad.cir", "Bad\nR1 a b\n")

    exit_code = main([str(netlist)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Error parsing netlist" in err
    assert "expects exactly 3 fields" in err


def test_cli_reports_unsupported_component(tmp_path, capsys):
    netlist = _write(tmp_path, "cap.cir", "Cap Circuit\nC1 a 0 1u\n")

    exit_code = main([str(netlist)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "isn't supported yet" in err


def test_cli_reports_singular_system(tmp_path, capsys):
    netlist = _write(tmp_path, "floating.cir", "Floating\nR1 a b 1k\n")

    exit_code = main([str(netlist)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "floating node" in err
    assert "ground" in err


def test_cli_requires_a_netlist_argument(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "netlist" in err
