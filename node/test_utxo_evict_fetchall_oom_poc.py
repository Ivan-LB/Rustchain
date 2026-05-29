# SPDX-License-Identifier: MIT
"""Regression tests for _evict_stale_data_input_txs() fetchall() OOM fix."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from utxo_db import UtxoDB  # noqa: E402


def _make_utxo_db(path: str) -> UtxoDB:
    db = UtxoDB(path)
    db.init_tables()
    return db


def _seed_mempool(db: UtxoDB, entries: list) -> None:
    """Insert rows into utxo_mempool and optionally utxo_mempool_inputs.

    Each entry is a dict with keys:
      tx_id, data_inputs (list[str]), regular_inputs (list[str] optional)
    """
    conn = db._conn()
    try:
        for entry in entries:
            tx_data = json.dumps({
                "tx_id": entry["tx_id"],
                "data_inputs": entry.get("data_inputs", []),
            })
            conn.execute(
                "INSERT INTO utxo_mempool VALUES (?,?,0,0,9999999999)",
                (entry["tx_id"], tx_data),
            )
            for box_id in entry.get("regular_inputs", []):
                conn.execute(
                    "INSERT INTO utxo_mempool_inputs VALUES (?,?)",
                    (box_id, entry["tx_id"]),
                )
        conn.commit()
    finally:
        conn.close()


def _mempool_tx_ids(db: UtxoDB) -> set:
    conn = db._conn()
    try:
        rows = conn.execute("SELECT tx_id FROM utxo_mempool").fetchall()
        return {r["tx_id"] for r in rows}
    finally:
        conn.close()


class TestEvictStaleDataInputTxs(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = _make_utxo_db(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_evicts_tx_whose_data_input_was_spent(self):
        """Transaction referencing a spent box as a data_input is removed."""
        _seed_mempool(self.db, [
            {"tx_id": "tx_stale", "data_inputs": ["spent_box"]},
            {"tx_id": "tx_clean", "data_inputs": ["other_box"]},
            {"tx_id": "tx_no_di", "data_inputs": []},
        ])

        evicted = self.db._evict_stale_data_input_txs(["spent_box"])

        remaining = _mempool_tx_ids(self.db)
        self.assertEqual(evicted, 1)
        self.assertNotIn("tx_stale", remaining)
        self.assertIn("tx_clean", remaining)
        self.assertIn("tx_no_di", remaining)

    def test_evicts_tx_via_regular_input_path(self):
        """Transaction with a spent box as a regular input is removed."""
        _seed_mempool(self.db, [
            {"tx_id": "tx_reg_stale", "data_inputs": [], "regular_inputs": ["spent_box"]},
            {"tx_id": "tx_unrelated", "data_inputs": [], "regular_inputs": ["other_box"]},
        ])

        evicted = self.db._evict_stale_data_input_txs(["spent_box"])

        remaining = _mempool_tx_ids(self.db)
        self.assertEqual(evicted, 1)
        self.assertNotIn("tx_reg_stale", remaining)
        self.assertIn("tx_unrelated", remaining)

    def test_evicts_across_both_paths_simultaneously(self):
        """Both regular-input and data-input stale txs are evicted in one call."""
        _seed_mempool(self.db, [
            {"tx_id": "tx_di_stale", "data_inputs": ["box_A"]},
            {"tx_id": "tx_ri_stale", "data_inputs": [], "regular_inputs": ["box_A"]},
            {"tx_id": "tx_safe", "data_inputs": ["box_B"], "regular_inputs": ["box_C"]},
        ])

        evicted = self.db._evict_stale_data_input_txs(["box_A"])

        remaining = _mempool_tx_ids(self.db)
        self.assertEqual(evicted, 2)
        self.assertNotIn("tx_di_stale", remaining)
        self.assertNotIn("tx_ri_stale", remaining)
        self.assertIn("tx_safe", remaining)

    def test_empty_spent_ids_returns_zero(self):
        """Calling with no spent box IDs is a no-op."""
        _seed_mempool(self.db, [
            {"tx_id": "tx_a", "data_inputs": ["box_1"]},
        ])

        evicted = self.db._evict_stale_data_input_txs([])

        self.assertEqual(evicted, 0)
        self.assertIn("tx_a", _mempool_tx_ids(self.db))

    def test_empty_mempool_returns_zero(self):
        """No rows to scan means no evictions."""
        evicted = self.db._evict_stale_data_input_txs(["spent_box"])
        self.assertEqual(evicted, 0)

    def test_cursor_iteration_no_fetchall(self):
        """_evict_stale_data_input_txs() must not call fetchall() on the full mempool scan."""
        _seed_mempool(self.db, [
            {"tx_id": "tx_a", "data_inputs": ["box_1"]},
            {"tx_id": "tx_b", "data_inputs": ["box_2"]},
        ])

        fetchall_violations = []

        class _SpyCursor:
            def __init__(self, real_cursor, is_full_scan):
                self._real = real_cursor
                self._is_full_scan = is_full_scan

            def fetchall(self):
                if self._is_full_scan:
                    fetchall_violations.append("fetchall() on full utxo_mempool scan")
                return self._real.fetchall()

            def __iter__(self):
                return iter(self._real)

            def __next__(self):
                return next(self._real)

            def __getattr__(self, name):
                return getattr(self._real, name)

        class _SpyConnection:
            def __init__(self, real_conn):
                self._real = real_conn

            def execute(self, sql, params=()):
                real_cursor = self._real.execute(sql, params)
                is_full_scan = (
                    "FROM utxo_mempool" in sql
                    and "WHERE" not in sql.upper()
                )
                return _SpyCursor(real_cursor, is_full_scan)

            def __getattr__(self, name):
                return getattr(self._real, name)

        real_conn_factory = self.db._conn

        def spy_conn_factory():
            return _SpyConnection(real_conn_factory())

        self.db._conn = spy_conn_factory

        self.db._evict_stale_data_input_txs(["box_1"])

        self.assertEqual(
            fetchall_violations, [],
            "fetchall() called on the full utxo_mempool scan — "
            "OOM regression: the fix must use cursor iteration",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
