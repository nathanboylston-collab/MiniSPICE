import pytest

from minispice.parser.units import parse_value


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1000", 1000.0),
        ("1.5", 1.5),
        ("1e-9", 1e-9),
        ("-1e-9", -1e-9),
        (".5", 0.5),
        # Engineering suffixes.
        ("1k", 1e3),
        ("4.7k", 4.7e3),
        ("10meg", 10e6),
        ("10MEG", 10e6),
        ("10Meg", 10e6),
        ("1t", 1e12),
        ("1g", 1e9),
        ("10m", 10e-3),
        ("10u", 10e-6),
        ("10n", 10e-9),
        ("10p", 10e-12),
        ("10f", 10e-15),
        # Suffix followed by a cosmetic (ignored) unit name.
        ("10uF", 10e-6),
        ("4.7kOhm", 4.7e3),
        ("1meg Hz".replace(" ", ""), 1e6),
        # Bare unit name with no scale suffix at all.
        ("5V", 5.0),
        ("2A", 2.0),
    ],
)
def test_parse_value(text, expected):
    assert parse_value(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "not-a-number",
        "-",
        "+",
        ".",
        "10k5",  # digit stuck onto a valid suffix
        "10k#",  # symbol stuck onto a valid suffix
        "10#zz",  # symbol with no recognized suffix at all
    ],
)
def test_parse_value_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        parse_value(text)
