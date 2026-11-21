import importlib
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"))
cache = importlib.import_module("cache")


class SlowFirstStatement:
    """A real connection whose first statement (the WAL pragma) is slow, widening the window in which another
    thread could see a connection whose table doesn't exist yet."""

    def __init__(self, conn):
        self._conn, self._first = conn, True

    def execute(self, *args):
        if self._first:
            self._first = False
            time.sleep(0.3)
        return self._conn.execute(*args)

    def commit(self):
        return self._conn.commit()


def test_first_use_from_many_threads_on_a_fresh_cache(tmp_path, monkeypatch):
    real_connect = cache.sqlite3.connect
    monkeypatch.setattr(cache.sqlite3, "connect", lambda *a, **kw: SlowFirstStatement(real_connect(*a, **kw)))
    monkeypatch.setattr(cache, "_conn", None)
    start, errors, results = threading.Barrier(16), [], []

    def worker(i):
        start.wait()
        try:
            cache.put(str(tmp_path), f"k{i}", f"v{i}")
            results.append(cache.get(str(tmp_path), f"k{i}"))
        except Exception as exc:  # the old code raised sqlite3.OperationalError: no such table: kv
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    monkeypatch.setattr(cache, "_conn", None)
    assert not errors
    assert sorted(results) == sorted(f"v{i}" for i in range(16))
