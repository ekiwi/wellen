# We are trying to be somewhat compatible with VCDVCD

from test_waveform import _git_root_rel
from pywellen import Waveform


def test_vcd_vcd_data():
    filename = _git_root_rel("wellen/inputs/icarus/counter_tb.vcd")
    waves = Waveform(path=filename)
    var = waves["counter_tb.out[1:0]"]
    # pywellen will always convert values to int eagerly
    assert var.tv[:6] == [
        (0, "xx"),
        (2, 0b00),
        (6, 0b01),
        (8, 0b10),
        (10, 0b11),
        (12, 0b00),
    ]

    assert var[0] == "xx"
    assert var[1] == "xx"
    assert var[2] == 0b00
    assert var[3] == 0b00
    assert var[6] == 0b01
    assert var[7] == 0b01
    assert var[24] == 0b10
    assert var[25] == 0b10
