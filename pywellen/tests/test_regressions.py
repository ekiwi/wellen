# Regressions tests for pywellen issues

from test_waveform import _git_root_rel
from pywellen import Waveform


# https://github.com/ekiwi/wellen/issues/92
def test_issue_92():
    filename = _git_root_rel("wellen/inputs/vcs/datapath_log.vcd")
    waves = Waveform(path=filename)
    var = waves["pipeline_datapath.Alu_output"]
    changes_at_1300 = [v for t, v in var.signal if t == 1300]
    assert changes_at_1300 == [
        0b0000000000000000000000000000000000000000000000000000000000000100,
        0b1111111111111111111111111111111111111111111111111111111111111100,
    ]


def test_issue_95():
    filename = _git_root_rel("wellen/inputs/xilinx_isim/test2x2_regex22_string1.vcd")
    waves = Waveform(path=filename)
    signal_name = (
        "AXI_top_tb_from_compiled.dut.a_regex_coprocessor.genblk1.a_topology.clk"
    )
    clk_in_genblk1_a_topology = waves[signal_name]
    assert clk_in_genblk1_a_topology.name == "clk"
    assert clk_in_genblk1_a_topology.full_name == signal_name


# https://github.com/ekiwi/wellen/issues/133
def test_issue_133():
    # stream_changes / stream_time_steps panicked on the highest-id signal:
    # SignalToVarMap sized its backing array to the max signal index instead
    # of max+1, so populating it for that signal indexed past the end (an
    # empty array for a single-signal file). Streaming all signals of a
    # single-signal VCD must deliver the changes instead of panicking.
    waves = Waveform(path=_git_root_rel("wellen/inputs/github_issues/issue133.vcd"))

    changes = []
    waves.stream_changes(lambda t, _sig, val: changes.append((t, val)), None)
    assert changes == [(0, 0), (10, 1), (20, 0)]

    steps = []
    waves.stream_time_steps(lambda t, _vals, _changed: steps.append(t), None)
    assert steps == [0, 10, 20]
