import abc
from typing import List, Optional
from pywellen import Signal


class SignalBuilder(abc.ABC):
    def get_all_signals(self) -> List[Signal]:
        raise NotImplementedError()

    def get_value_at_index(self, time_table_idx: int) -> Optional[int]:
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

    def get_value_at_index(self, time_table_idx: int) -> Optional[int]:
        current_value = self.signal.value_at_idx(time_table_idx)
        if isinstance(current_value, int):
            mask = (1 << (self.msb - self.lsb)) - 1
            return (current_value >> self.lsb) & mask
        else:
            return None

    def width(self):
        return self.msb - self.lsb
