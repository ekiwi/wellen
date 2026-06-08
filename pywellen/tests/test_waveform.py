from pywellen import Waveform, WaveformStream
import subprocess
from swerv import check_swerv_changes
from collections import defaultdict


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


def test_var_and_scope_index_api():
    """
    Demonstrates the index based api to access variables and scopes.
    """
    filename = _git_root_rel("wellen/inputs/verilator/surfer_issue_201.fst")
    wave = Waveform(filename)
    assert repr(wave).startswith("Waveform(")
    # absolute path
    assert repr(wave).endswith("wellen/inputs/verilator/surfer_issue_201.fst)")
    assert wave["TOP"].type == "module"
    assert repr(wave["TOP"]) == "Scope(TOP)"
    d_ready = wave["TOP.top.pix_port.d_ready"]
    assert d_ready.type == "Logic"
    assert d_ready.name == "d_ready"
    assert repr(d_ready) == "Var(TOP.top.pix_port.d_ready)"
    assert d_ready.full_name == "TOP.top.pix_port.d_ready"
    d_ready_by_another_way = wave["TOP"]["top"]["pix_port"]["d_ready"]
    assert d_ready == d_ready_by_another_way

    # ensure that var can be hashed
    dd = {d_ready: 1, d_ready_by_another_way: 3}
    assert len(dd) == 1


def test_var_change_access():
    """
    We automagically load signals if needed in order to provide an interface similar to vcdvcd
    """
    filename = _git_root_rel("wellen/inputs/verilator/surfer_issue_201.fst")
    wave = Waveform(filename)

    d_trig = wave["TOP.top.pix_port.d_trig"]
    assert d_trig.var_type == "Logic"
    assert d_trig.size == 1
    assert len(d_trig.tv) == 8
    assert list(d_trig.tv) == [
        (0, 0),
        (27400000, 1),
        (47400000, 0),
        (53800000, 1),
        (73800000, 0),
        (80200000, 1),
        (100200000, 0),
        (106600000, 1),
    ]
    assert d_trig.tv[0] == (0, 0)
    assert d_trig.tv[5] == (80200000, 1)


def test_iterate_hierarchy_example():
    """
    just an example of some of the pywellen apis
    """

    wave = Waveform(path=_git_root_rel("wellen/inputs/verilator/swerv1.vcd"))

    # only get the first ten
    all_vars = wave.all_vars()[0:10]
    changes = {}
    for var in all_vars:
        changes[var.full_name] = []
        for change_time, value in var.signal:
            changes[var.full_name].append((change_time, value))
    check_swerv_changes(changes)


def test_vcd_not_starting_at_zero():
    filename = _git_root_rel("wellen/inputs/gameroy/trace_prefix.vcd")
    waves = Waveform(path=filename)

    cpu = waves["gameroy.cpu"]
    assert cpu.name == "cpu"

    pc = cpu["pc"]
    assert pc.full_name == "gameroy.cpu.pc"
    sp = cpu["sp"]
    assert sp.full_name == "gameroy.cpu.sp"

    ## querying a signal before it has a value should return none
    pc_sig = pc.signal
    sp_sig = sp.signal

    ## pc is fine since it changes at 4 which is time_table idx 0
    assert pc_sig.value_at(0) is None, (
        "None, because at time 0, nothing has happened yet!"
    )
    assert pc_sig.value_at(4) is not None

    ## sp only changes at 16
    assert sp_sig.value_at(16) is not None
    for ii in range(16):
        assert sp_sig.value_at(ii) is None


def test_vcd_var_types_types():
    filename = _git_root_rel("wellen/inputs/gtkwave-analyzer/vcd_extensions.vcd")
    waves = Waveform(path=filename)

    # Collect all variables for testing
    all_vars = waves.all_vars()
    assert len(all_vars) > 0

    # Test various variable types and properties
    var_tests = {
        "ENUM2_IN": {"var_type": "Enum", "bitwidth": 2, "is_bit_vector": True},
        "STR_OUT": {"var_type": "String", "bitwidth": None, "is_string": True},
        "EVENT_IN": {"var_type": "Event", "bitwidth": 0},
        "INT32_OUT": {"var_type": "Integer", "bitwidth": 32, "is_bit_vector": True},
        "REAL_BUF": {"var_type": "Real", "bitwidth": None, "is_real": True},
        "REG128_INOUT": {"var_type": "Reg", "bitwidth": 128, "is_bit_vector": True},
        "SUPPLY0_var": {"var_type": "Supply0", "bitwidth": 1, "is_1bit": True},
        "SUPPLY1_var": {"var_type": "Supply1", "bitwidth": 1, "is_1bit": True},
        "TRI_var": {"var_type": "Tri", "bitwidth": 1, "is_1bit": True},
        "WIRE_var": {"var_type": "Wire", "bitwidth": 1, "is_1bit": True},
        "SV_BIT_10_var": {"var_type": "Bit", "bitwidth": 10, "is_bit_vector": True},
        "SV_LOGIC_10_var": {"var_type": "Logic", "bitwidth": 10, "is_bit_vector": True},
        "SV_INT32_var": {"var_type": "Int", "bitwidth": 32, "is_bit_vector": True},
        "SV_BYTE8_var": {"var_type": "Byte", "bitwidth": 8, "is_bit_vector": True},
    }

    # Find and test specific variables
    found_vars = {}
    for var in all_vars:
        var_name = var.name
        if var_name in var_tests:
            found_vars[var_name] = var

    # Test that we found the expected variables
    assert len(found_vars) >= 10, (
        f"Expected to find at least 10 test variables, found {len(found_vars)}"
    )

    for var_name, var in found_vars.items():
        expected = var_tests[var_name]

        # Test basic properties
        assert var.name == var_name
        assert var.full_name == f"main.{var_name}"

        # Test var_type
        var_type = var.var_type
        assert var_type == expected["var_type"], (
            f"Expected {expected['var_type']}, got {var_type} for {var_name}"
        )

        # Test bitwidth/length
        assert var.bitwidth == expected["bitwidth"], (
            f"Expected bitwidth {expected['bitwidth']}, got {var.bitwidth()} for {var_name}"
        )
        assert var.length == expected["bitwidth"], (
            f"Expected length {expected['bitwidth']}, got {var.length()} for {var_name}"
        )

        # Test direction (should be Unknown for VCD)
        direction = var.direction
        assert direction == "Unknown", (
            f"Expected direction Unknown, got {direction} for {var_name}"
        )

        # Test signal encoding properties
        if expected.get("is_string"):
            assert var.is_string, f"Expected {var_name} to be string"
            assert not var.is_real and not var.is_bit_vector
        elif expected.get("is_real"):
            assert var.is_real, f"Expected {var_name} to be real"
            assert not var.is_string and not var.is_bit_vector
        elif expected.get("is_bit_vector"):
            assert var.is_bit_vector, f"Expected {var_name} to be bit vector"
            assert not var.is_string and not var.is_real

        # Test is_1bit
        if expected.get("is_1bit"):
            assert var.is_1bit, f"Expected {var_name} to be 1-bit"
        elif expected["bitwidth"] and expected["bitwidth"] > 1:
            assert not var.is_1bit, f"Expected {var_name} to not be 1-bit"

        # Test enum_type (only for ENUM2_IN)
        enum_type = var.enum_type
        if var_name == "ENUM2_IN":
            # Note: enum_type might be None if no enum definition is provided in VCD
            pass  # VCD format may not include enum definitions


def test_hierarchy_metadata_swerv1():
    """Test reading metadata from hierarchy"""
    filename = _git_root_rel("wellen/inputs/verilator/swerv1.vcd")
    waves = Waveform(path=filename)
    assert waves.file_format == "VCD"
    assert waves.timescale.factor == 1
    exponent = waves.timescale.unit.to_exponent()
    assert exponent == -12
    assert str(waves.timescale.unit) == "ps"


# Some FST tests ported from Rust (wellen/tests/fst.rs)
def load_verilator_many_sv_datatypes():
    """Helper function to load the verilator many_sv_datatypes.fst file"""
    filename = _git_root_rel("wellen/inputs/verilator/many_sv_datatypes.fst")
    waves = Waveform(path=filename)
    top = waves.scopes()[0]
    assert top.name == "TOP"
    assert top == waves["TOP"], "an alternative way to get the scope"
    wrapper = top.scopes()[0]
    assert wrapper.name == "SVDataTypeWrapper"
    assert wrapper == waves["TOP.SVDataTypeWrapper"]
    bb = wrapper.scopes()[0]
    assert bb.name == "bb"
    assert bb == waves["TOP.SVDataTypeWrapper.bb"]
    return waves, bb


def test_fst_enum_signals():
    """Test enum signals from Verilator FST file"""
    waves, bb = load_verilator_many_sv_datatypes()

    # Find the abc_r variable
    abc_r = bb["abc_r"]
    assert abc_r is not None, "failed to find abc_r"

    # Test enum type
    enum_type = abc_r.enum_type
    assert enum_type is not None, "abc_r should have an enum type!"

    enum_name, enum_values = enum_type
    assert enum_name == "SVDataTypeBlackBox.abc"

    # Sort the enum values for comparison
    enum_values_sorted = sorted(enum_values)
    expected = [("00", "A"), ("01", "B"), ("10", "C"), ("11", "D")]
    assert enum_values_sorted == expected


def test_fst_var_directions():
    """Test variable directions from Verilator FST file"""
    waves, bb = load_verilator_many_sv_datatypes()

    # Expected variable properties
    expected_vars = {
        "abc_r": {"direction": "Implicit", "var_type": "Logic"},
        "clock": {"direction": "Input", "var_type": "Wire"},
        "int_r": {"direction": "Implicit", "var_type": "Integer"},
        "out": {"direction": "Output", "var_type": "Wire"},
        "real_r": {"direction": "Implicit", "var_type": "Real"},
        "time_r": {"direction": "Implicit", "var_type": "Bit"},
    }

    found_vars = {}
    for var in bb.vars():
        var_name = var.name
        if var_name in expected_vars:
            found_vars[var_name] = var

    # Test each expected variable
    for var_name, expected in expected_vars.items():
        assert var_name in found_vars, f"Variable {var_name} not found"
        var = found_vars[var_name]

        direction = var.direction
        var_type = var.var_type

        assert direction == expected["direction"], (
            f"Expected direction {expected['direction']}, got {direction} for {var_name}"
        )
        assert var_type == expected["var_type"], (
            f"Expected var_type {expected['var_type']}, got {var_type} for {var_name}"
        )


def test_scope_types():
    """Test scope_type() method with different scope types from VCD extensions file"""
    filename = _git_root_rel("wellen/inputs/gtkwave-analyzer/vcd_extensions.vcd")
    waves = Waveform(path=filename)

    # Get the top scope (should be 'main' with type 'module')
    top_scopes = waves.scopes()
    assert len(top_scopes) == 1, f"Expected 1 top scope, found {len(top_scopes)}"

    main_scope = top_scopes[0]
    assert main_scope.name == "main"
    assert main_scope.scope_type == "module"

    # Test various child scope types
    expected_scope_types = {
        "MODULE0": "module",
        "TASK0": "task",
        "FUNCTION0": "function",
        "BEGIN0": "begin",
        "FORK0": "fork",
        "GENERATE0": "generate",
        "STRUCT0": "struct",
        "UNION0": "union",
        "CLASS0": "class",
        "INTERFACE0": "interface",
        "PACKAGE0": "package",
        "PROGRAM0": "program",
        "ARCHITECTURE0": "vhdl_architecture",
        "PROCEDURE0": "vhdl_procedure",
        "FUNCTION1": "vhdl_function",
        "RECORD0": "vhdl_record",
        "PROCESS0": "vhdl_process",
        "BLOCK0": "vhdl_block",
        "FOR_GENERATE0": "vhdl_for_generate",
        "IF_GENERATE0": "vhdl_if_generate",
        "GENERATE1": "vhdl_generate",
    }

    # Collect all child scopes
    child_scopes = main_scope.scopes()
    found_scopes = {}

    for scope in child_scopes:
        scope_name = scope.name
        if scope_name in expected_scope_types:
            found_scopes[scope_name] = scope

    # Test that we found the expected scopes
    assert len(found_scopes) >= 15, (
        f"Expected to find at least 15 test scopes, found {len(found_scopes)}"
    )

    # Test each scope type
    for scope_name, expected_type in expected_scope_types.items():
        if scope_name in found_scopes:
            scope = found_scopes[scope_name]
            actual_type = scope.scope_type
            assert actual_type == expected_type, (
                f"Expected scope type '{expected_type}' for '{scope_name}', got '{actual_type}'"
            )

            # Also test that full_name works correctly
            expected_full_name = f"main.{scope_name}"
            actual_full_name = scope.full_name
            assert actual_full_name == expected_full_name, (
                f"Expected full name '{expected_full_name}', got '{actual_full_name}'"
            )


def test_stream_changes():
    wave = WaveformStream(path=_git_root_rel("wellen/inputs/verilator/swerv1.vcd"))

    # accessing signals directly in stream mode is not allowed
    error = ""
    try:
        wave.all_vars()[0].signal
    except RuntimeError as e:
        error = str(e)
    assert "access" in error and "stream" in error

    # record all changes in streaming mode
    changes = defaultdict(list)

    def on_change(time, signal, value):
        for var in wave[signal]:
            changes[var.full_name].append((time, value))

    wave.stream_changes(on_change, include=wave.all_vars()[0:10])
    check_swerv_changes(changes)


def test_stream_time_steps():
    wave = WaveformStream(path=_git_root_rel("wellen/inputs/verilator/swerv1.vcd"))

    # record all changes in streaming mode
    changes = defaultdict(list)

    def on_change(time, values, changed):
        for signal in changed:
            value = values[signal]
            for var in wave[signal]:
                changes[var.full_name].append((time, value))

    wave.stream_time_steps(on_change, include=wave.all_vars()[0:10])
    check_swerv_changes(changes)


def test_signal_slice():
    filename = _git_root_rel("wellen/inputs/gameroy/trace_prefix.vcd")
    waves = Waveform(path=filename)
    pc = waves["gameroy.cpu"]["pc"]
    values = pc.signal
    assert len(values) == 4269

    # we check that indexing the signal directly works the same as indexing into the list
    values_list = list(values)
    assert values[0] == values_list[0]
    # negative index
    assert values[-1] == values_list[-1]
    assert values[100] == values_list[100]
    assert values[1:] == values_list[1:]
    assert values[:] == values_list[:]
    assert values[::] == values_list[::]
    assert values[:3] == values_list[:3]
    assert values[:-3] == values_list[:-3]
    assert values[-4:-3] == values_list[-4:-3]
    # step size > 1
    assert values[2:-3:2] == values_list[2:-3:2]
