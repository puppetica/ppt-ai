from typing import Iterator

from mcap_protobuf.reader import McapProtobufMessage, read_protobuf_messages


def message_generator(path: str, topics: list[str] | None = None, start_ts: int | None = None):
    with open(path, "rb") as f:
        for msg in read_protobuf_messages(f, topics=topics, start_time=start_ts):
            yield msg


def merged_messages(
    paths: list[str], topics: list[str] | None = None, start_ts: int | None = None
) -> Iterator[tuple[int, McapProtobufMessage]]:
    gens = [message_generator(p, topics, start_ts) for p in paths]
    current = []

    # prime each generator with file index
    for file_idx, g in enumerate(gens):
        try:
            m = next(g)
            current.append((m.publish_time_ns, file_idx, m, g))
        except StopIteration:
            pass

    while current:
        # sort by ts first, then by file_idx
        ts_min, file_idx, msg, g = min(current, key=lambda x: (x[0], x[1]))
        yield ts_min, msg
        current.remove((ts_min, file_idx, msg, g))
        try:
            m_next = next(g)
            current.append((m_next.publish_time_ns, file_idx, m_next, g))
        except StopIteration:
            pass
