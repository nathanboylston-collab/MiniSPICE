import pytest

from minispice.parser.units import parse_value


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1000", 1000.0),
        ("1.5", 1.5),
        ("1e-9", 1e-9),
        ("1k", 1e3),
        ("4.7k", 4.7e3),
        ("10meg", 10e6),
        ("10m", 10e-3),
        ("10u", 10e-6),
        ("10n", 10e-9),
        ("10p", 10e-12),
        ("10f", 10e-15),
        ("10uF", 10e-6),
        ("5V", 5.0),
    ],
)
def test_parse_value(text, expected):
    assert parse_value(text) == pytest.approx(expected)


def test_parse_value_rejects_garbage():
    with pytest.raises(ValueError):
        parse_value("not-a-number")
