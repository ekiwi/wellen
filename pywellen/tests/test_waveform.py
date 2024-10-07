from pywellen import Waveform
from pathlib import Path
import subprocess


def _git_root_rel(path: str) -> str:
    try:
        git_root = (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
            .strip()
            .decode("utf-8")
        )
        return f"{git_root}/{path}"
    except subprocess.CalledProcessError:
        # raise RuntimeError("This directory is not a git repository.")
        return path


def iterate_hierarchy_example():
    """
    FIXME: this test doesnt do anything right now,
    just an example of some of the pywellen apis
    """

    wave = Waveform(path=_git_root_rel("wellen/inputs/verilator/swerv1.vcd"))
    hier = wave.hierarchy

    # only get the first ten
    all_vars = [var for var in hier.all_vars()][0:10]
    for var in all_vars:
        sig = wave.get_signal(var)
        print(f"printing all the changes for {var.full_name(hier)}")
        for change_time, value in sig.all_changes():
            print(f"Change recorded at time {change_time} with new value {value}")


def test_vcd_not_starting_at_zero():
    filename = _git_root_rel("wellen/inputs/gameroy/trace_prefix.vcd")
    waves = Waveform(path=filename)

    h = waves.hierarchy

    # the first signal change only happens at 4
    assert waves.time_table[0] == 4

    top = next(h.top_scopes())
    assert top.name(h) == "gameroy"
    cpu = next(top.scopes(h))

    assert cpu.name(h) == "cpu"

    pc = next(v for v in cpu.vars(h) if v.name(h) == "pc")
    assert pc.full_name(h) == "gameroy.cpu.pc"
    sp = next(v for v in cpu.vars(h) if v.name(h) == "sp")
    assert sp.full_name(h) == "gameroy.cpu.sp"

    ## querying a signal before it has a value should return none
    pc_sig = waves.get_signal(pc)
    sp_sig = waves.get_signal(sp)

    ## pc is fine since it changes at 4 which is time_table idx 0
    # pc_signal = waves.get_signal(pc.signal_ref())
    assert pc_sig.value_at_idx(0) is not None

    ## sp only changes at 16 which is time table idx 1
    assert sp_sig.value_at_idx(1) is not None
    assert sp_sig.value_at_idx(0) is None


test_vcd_not_starting_at_zero()
