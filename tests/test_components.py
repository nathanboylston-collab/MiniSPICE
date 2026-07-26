import pytest

from minispice.components import (
    Capacitor,
    CurrentSource,
    Inductor,
    Resistor,
    VoltageSource,
)


def test_resistor_attributes():
    r = Resistor("R1", "n1", "0", 1000.0)
    assert r.name == "R1"
    assert r.nodes == ("n1", "0")
    assert r.node_pos == "n1"
    assert r.node_neg == "0"
    assert r.resistance == 1000.0


def test_capacitor_attributes():
    c = Capacitor("C1", "n1", "n2", 1e-6)
    assert c.capacitance == 1e-6
    assert c.nodes == ("n1", "n2")


def test_inductor_attributes():
    l = Inductor("L1", "n1", "n2", 1e-3)
    assert l.inductance == 1e-3


def test_voltage_source_attributes():
    v = VoltageSource("V1", "n1", "0", 5.0)
    assert v.voltage == 5.0


def test_current_source_attributes():
    i = CurrentSource("I1", "n1", "0", 0.001)
    assert i.current == 0.001


@pytest.mark.parametrize(
    "component",
    [
        Capacitor("C1", "n1", "0", 1.0),
        Inductor("L1", "n1", "0", 1.0),
        VoltageSource("V1", "n1", "0", 1.0),
        CurrentSource("I1", "n1", "0", 1.0),
    ],
)
def test_stamp_not_yet_implemented(component):
    # Resistor.stamp() is implemented (see tests/test_stamping.py);
    # every other component's stamp() is still a placeholder.
    with pytest.raises(NotImplementedError):
        component.stamp(system=None)
