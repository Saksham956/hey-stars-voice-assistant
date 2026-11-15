"""Bounded console logging: audio workers never wait for terminal I/O."""
import logging
import queue
import threading


class AsyncConsoleHandler(logging.Handler):
    def __init__(self, sink=None, capacity=256):
        super().__init__()
        self.sink = sink if sink is not None else logging.StreamHandler()
        self.sink.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.records = queue.Queue(capacity)
        self.dropped_records = 0
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self.run, name='console-log', daemon=True)
        self.worker.start()

    def emit(self, record):
        if self.stopping.is_set(): return
        try: self.records.put_nowait(record)
        except queue.Full: self.dropped_records += 1

    def run(self):
        reported_drops = 0
        while not self.stopping.is_set() or not self.records.empty():
            try: record = self.records.get(timeout=.05)
            except queue.Empty: continue
            try:
                # One consumer owns the sink. Its potentially blocking output
                # never holds the producer handler's logging lock.
                self.sink.emit(record)
                dropped = self.dropped_records
                if dropped > reported_drops:
                    self.sink.emit(logging.LogRecord(
                        'stars', logging.WARNING, __file__, 0,
                        'Console log backlog: %d records dropped; audio was not blocked',
                        (dropped-reported_drops,), None))
                    reported_drops = dropped
            except Exception:
                # A broken/closed terminal must not stop audio or recurse into
                # logging. Remain bounded even if the sink cannot recover.
                pass

    def close(self):
        self.stopping.set()
        if threading.current_thread() is not self.worker:
            self.worker.join(timeout=.25)
        super().close()


def configure_console_logging():
    handler = AsyncConsoleHandler()
    root = logging.getLogger()
    # Core v2 owns console setup. Panel handlers live on the stars logger.
    for previous in root.handlers[:]: root.removeHandler(previous)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler
