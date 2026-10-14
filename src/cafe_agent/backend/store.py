"""Transactional SQLite persistence, including the deterministic logical clock."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = RLock()
        with self.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, body TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clock "
                "(id INTEGER PRIMARY KEY CHECK(id=1), minute INTEGER)"
            )
            connection.execute("INSERT OR IGNORE INTO clock VALUES (1, 0)")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def close(self) -> None:
        with self.lock:
            self.connection.close()
