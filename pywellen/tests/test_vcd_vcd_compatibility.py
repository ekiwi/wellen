# We are trying to be somewhat compatible with VCDVCD

from test_waveform import _git_root_rel
from pywellen import Waveform

def test_vcd_vcd_data():
    filename = _git_root_rel("wellen/inputs/icarus/counter_tb.vcd")
    waves = Waveform(path=filename)
    var = waves['counter_tb.out[1:0]']
    assert var.tv[:6] == [( 0,  'x'),
                          ( 2,  '0'),
                          ( 6,  '1'),
                          ( 8, '10'),
                          (10, '11'),
                          (12,  '0'),]

