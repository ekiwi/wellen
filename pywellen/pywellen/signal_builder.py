import abc
from typing import Dict, List, Optional, Union
from pywellen import Signal, Waveform, create_derived_signal


class SignalBuilder(abc.ABC):
    def get_all_signals(self) -> List[Signal]:
        raise NotImplementedError()

    def get_value_at_index(self, time_table_idx: int) -> Optional[Union[int, str]]:
        """
        Returns the value at the index -- wellen interprets this as an unsigned
        width of size `self.width`
        """
        raise NotImplementedError()

    def width(self):
        """
        Width of the generated signal in bits

        It MUST be static -- width is not allowed to change function of
        timetable index
        """

    def to_signal(self) -> Signal:
        return create_derived_signal(self)


class PassThrough(SignalBuilder):
    signal: Signal

    def __init__(self, signal: Signal, lsb: int, msb: int):

        self.signal = signal

    def get_all_signals(self) -> List[Signal]:
        return [self.signal]

    def get_value_at_index(self, time_table_idx: int) -> Optional[Union[int, str]]:
        return self.signal.value_at_idx(time_table_idx)

    def width(self):
        return self.signal.width


class SlicedSignal(SignalBuilder):
    signal: Signal
    lsb: int
    msb: int

    def __init__(self, signal: Signal, lsb: int, msb: int):

        self.signal = signal
        self.lsb = lsb
        self.msb = msb

    def get_all_signals(self) -> List[Signal]:
        return [self.signal]

    def get_value_at_index(self, time_table_idx: int) -> Optional[Union[int, str]]:
        current_value = self.signal.value_at_idx(time_table_idx)
        if isinstance(current_value, int):
            mask = (1 << (self.msb - self.lsb)) - 1
            return (current_value >> self.lsb) & mask
        else:
            return None

    def width(self):
        return self.msb - self.lsb


def get_signals(wave: Waveform) -> Dict[str, SignalBuilder]: ...
