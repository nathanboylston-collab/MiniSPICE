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


def test_parse_text_strips_inline_comments():
    text = "Title\nR1 a 0 10 ; a trailing note\n"
    circuit = SpiceParser().parse_text(text)

    assert len(circuit.components) == 1
    assert circuit.components[0].resistance == 10.0


def test_parse_text_stops_at_end_directive():
    text = "Title\nR1 a 0 10\n.end\nR2 b 0 20\n"
    circuit = SpiceParser().parse_text(text)

    assert len(circuit.components) == 1


def test_parse_text_empty_input_yields_empty_circuit():
    circuit = SpiceParser().parse_text("")

    assert circuit.title == ""
    assert circuit.components == []


def test_title_only_netlist_yields_no_components():
    circuit = SpiceParser().parse_text("Just A Title\n.end\n")

    assert circuit.title == "Just A Title"
    assert circuit.components == []


# ---------------------------------------------------------------------------
# Malformed netlists
# ---------------------------------------------------------------------------


def test_unsupported_element_raises():
    with pytest.raises(SpiceParseError, match="unsupported element type"):
        SpiceParser().parse_text("Title\nQ1 a 0 10\n")


def test_unsupported_element_message_lists_supported_prefixes():
    with pytest.raises(SpiceParseError, match=r"supported prefixes: C, I, L, R, V"):
        SpiceParser().parse_text("Title\nQ1 a 0 10\n")


def test_missing_fields_raises():
    with pytest.raises(SpiceParseError, match="expects exactly 3 fields"):
        SpiceParser().parse_text("Title\nR1 a\n")


def test_missing_value_field_raises():
    with pytest.raises(SpiceParseError, match="expects exactly 3 fields"):
        SpiceParser().parse_text("Title\nR1 a b\n")


def test_extra_fields_raises():
    with pytest.raises(SpiceParseError, match="expects exactly 3 fields"):
        SpiceParser().parse_text("Title\nR1 a b 10 20\n")


def test_self_loop_component_raises():
    with pytest.raises(SpiceParseError, match="connects node 'a' to itself"):
        SpiceParser().parse_text("Title\nR1 a a 10\n")


def test_duplicate_component_name_raises():
    text = "Title\nR1 a 0 10\nR1 a b 20\n"
    with pytest.raises(SpiceParseError, match="duplicate component name"):
        SpiceParser().parse_text(text)


def test_duplicate_component_name_is_case_insensitive():
    text = "Title\nR1 a 0 10\nr1 a b 20\n"
    with pytest.raises(SpiceParseError, match="duplicate component name"):
        SpiceParser().parse_text(text)


@pytest.mark.parametrize("bad_value", ["-10", "0"])
def test_non_positive_resistance_raises(bad_value):
    with pytest.raises(SpiceParseError, match="resistance must be positive"):
        SpiceParser().parse_text(f"Title\nR1 a b {bad_value}\n")


@pytest.mark.parametrize("bad_value", ["-1u", "0"])
def test_non_positive_capacitance_raises(bad_value):
    with pytest.raises(SpiceParseError, match="capacitance must be positive"):
        SpiceParser().parse_text(f"Title\nC1 a b {bad_value}\n")


@pytest.mark.parametrize("bad_value", ["-1m", "0"])
def test_non_positive_inductance_raises(bad_value):
    with pytest.raises(SpiceParseError, match="inductance must be positive"):
        SpiceParser().parse_text(f"Title\nL1 a b {bad_value}\n")


@pytest.mark.parametrize("value", ["-5", "0"])
def test_voltage_source_allows_zero_and_negative_values(value):
    circuit = SpiceParser().parse_text(f"Title\nV1 a 0 {value}\n")

    assert circuit.components[0].voltage == float(value)


def test_malformed_value_raises_with_line_number():
    text = "Title\nR1 a b 10\nR2 c d not-a-number\n"
    with pytest.raises(SpiceParseError, match=r"line 3: .*invalid numeric value"):
        SpiceParser().parse_text(text)


def test_malformed_line_raises():
    with pytest.raises(SpiceParseError):
        SpiceParser().parse_text("Title\nR1 a\n")


def test_parse_file(tmp_path):
    netlist_file = tmp_path / "circuit.cir"
    netlist_file.write_text(VOLTAGE_DIVIDER)

    circuit = SpiceParser().parse_file(netlist_file)

    assert circuit.title == "Voltage Divider"
    assert len(circuit.components) == 3
