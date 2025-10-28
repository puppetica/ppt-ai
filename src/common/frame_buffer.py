from dataclasses import dataclass, field

import numpy as np
from mcap_protobuf.reader import McapProtobufMessage


@dataclass
class FrameBuffer:
    topic: str
    size: int = 40
    index: int = field(default=0, init=False)  # points to the current index in the buffer
    _data: list[tuple[int, McapProtobufMessage]] = field(default_factory=list, init=False)  # (timestamp_ns, data)

    def add(self, item: tuple[int, McapProtobufMessage]):
        if len(self._data) < self.size:
            self._data.append(item)
        else:
            self._data[self.index] = item
        self.index = (self.index + 1) % self.size  # move index in a circular manner

    def get(self, idx: int) -> tuple[int, McapProtobufMessage]:
        if not self._data:
            raise IndexError("Buffer is empty")
        # interpret idx=0 as latest
        latest_idx = (self.index - 1 - idx) % len(self._data)
        return self._data[latest_idx]

    def get_idx_by_ts(self, ts_us: int) -> int:
        smallest_diff = np.inf
        best_idx = 0
        for idx in range(len(self._data)):
            new_diff = abs(self._data[idx][0] - ts_us)
            if smallest_diff > new_diff:
                best_idx = idx
                smallest_diff = new_diff
        return best_idx

    def get_by_ts(self, ts_ns: int) -> tuple[int, McapProtobufMessage]:
        if len(self._data) == 0:
            raise IndexError("Buffer is emtpy")
        idx = self.get_idx_by_ts(ts_ns)
        return self._data[idx]

    def is_buffer_full(self) -> bool:
        return len(self._data) == self.size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self.get(idx)

    def __iter__(self):
        """Iterate from latest to oldest."""
        for i in range(len(self._data)):
            yield self.get(i)

    def __reversed__(self):
        """Iterate from oldest to newest (chronological order)."""
        for i in range(len(self._data)):
            yield self.get(len(self._data) - 1 - i)
