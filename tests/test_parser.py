import pytest

from minispice.components import Resistor, VoltageSource
from minispice.parser import SpiceParseError, SpiceParser

VOLTAGE_DIVIDER = """\
Voltage Divider
V1 in 0 5
R1 in out 1k
R2 out 0 1k
.end
"""


def test_parse_text_sets_title_and_components():
    circuit = SpiceParser().parse_text(VOLTAGE_DIVIDER)

    assert circuit.title == "Voltage Divider"
    assert len(circuit.components) == 3
    assert isinstance(circuit.components[0], VoltageSource)
    assert isinstance(circuit.components[1], Resistor)
    assert circuit.components[1].resistance == 1000.0


def test_parse_text_ignores_comments_and_blank_lines():
    text = "Title\n* this is a comment\n\nR1 a 0 10\n"
    circuit = SpiceParser().parse_text(text)

    assert len(circuit.components) == 1
    assert circuit.components[0].name == "R1"


def test_parse_text_stops_at_end_directive():
    text = "Title\nR1 a 0 10\n.end\nR2 b 0 20\n"
    circuit = SpiceParser().parse_text(text)

    assert len(circuit.components) == 1


def test_unsupported_element_raises():
    with pytest.raises(SpiceParseError):
        SpiceParser().parse_text("Title\nQ1 a 0 10\n")


def test_malformed_line_raises():
    with pytest.raises(SpiceParseError):
        SpiceParser().parse_text("Title\nR1 a\n")


def test_parse_file(tmp_path):
    netlist_file = tmp_path / "circuit.cir"
    netlist_file.write_text(VOLTAGE_DIVIDER)

    circuit = SpiceParser().parse_file(netlist_file)

    assert circuit.title == "Voltage Divider"
    assert len(circuit.components) == 3
