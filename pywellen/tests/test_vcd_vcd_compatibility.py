# We are trying to be somewhat compatible with VCDVCD

from test_waveform import _git_root_rel
from pywellen import Waveform


def test_vcd_vcd_data():
    filename = _git_root_rel("wellen/inputs/icarus/counter_tb.vcd")
    waves = Waveform(path=filename)
    var = waves["counter_tb.out[1:0]"]
    # pywellen will always pad the values to fit the fixed width
    assert var.tv[:6] == [
        (0, "xx"),
        (2, "00"),
        (6, "01"),
        (8, "10"),
        (10, "11"),
        (12, "00"),
    ]

    assert var[0] == "xx"
    assert var[1] == "xx"
    assert var[2] == "00"
    assert var[3] == "00"
    assert var[6] == "01"
    assert var[7] == "01"
    assert var[24] == "10"
    assert var[25] == "10"
