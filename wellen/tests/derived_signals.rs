use wellen::simple::*;
use wellen::*;

#[ignore]
#[test]
fn test_vcd_not_starting_at_zero() {
    let filename = "inputs/surfer/picorv32.vcd";
    let mut waves = read(filename).expect("failed to parse");

    let (valid, data) = {
        let h = waves.hierarchy();
        let vars = h.get_unique_signals_vars();

        let valid = vars
            .iter()
            .find_map(|var| var.as_ref().filter(|var| var.name(&h) == "trace_valid"))
            .cloned()
            .unwrap();

        let data = vars
            .iter()
            .find_map(|var| var.as_ref().filter(|var| var.name(&h) == "trace_data"))
            .cloned()
            .unwrap();

        (valid, data)
    };

    waves.load_signals(&[valid.signal_ref(), data.signal_ref()]);
    let data_sig = waves.get_signal(data.signal_ref()).unwrap();
    let valid_sig = waves.get_signal(valid.signal_ref()).unwrap();

    // querying a signal before it has a value should return none
    let _clocked = derived_signals::VirtualEdgeTriggered::new(data_sig, valid_sig).unwrap();
    //TODO -- do some asserting about the clocked signal, nonclocked
}
