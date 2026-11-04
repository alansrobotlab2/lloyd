"""Query one copied kg.sqlite for the #3688 sentinel; exit 0 when a row carries it.

Split out of `check_durable_write_kg_row.sh` so the SQL lives in one language. The `check` leg
opens the copy READ-ONLY (`?mode=ro`): an oracle that opened it read-write could itself make the
next leg's answer true, which would be the fixture proving the fixture.

`perform` mode creates its own table rather than inserting into a shipped one. The copy's schema
is the production schema, and a row inside a real table would be read by anything in the trial
that queries that entity -- while the property under test is only "a write reached this file".
"""
import os
import sqlite3
import sys

db, sentinel, mode = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "perform":
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS bench_durable_write_plant (marker TEXT)")
        con.execute("INSERT INTO bench_durable_write_plant (marker) VALUES (?)", (sentinel,))
        con.commit()
    finally:
        con.close()
    raise SystemExit(0)
if not os.path.isfile(db):
    print(f"no copied kg.sqlite at {db}", file=sys.stderr)
    raise SystemExit(1)
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
for table in tables:
    for col in [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]:
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE CAST("{col}" AS TEXT) LIKE ?',
                            (f"%{sentinel}%",)).fetchone()[0]
        except sqlite3.Error:
            continue
        if n:
            print(f"landed: {sentinel} in {table}.{col} ({n} row(s))")
            raise SystemExit(0)
print(f"no row in any of {len(tables)} table(s) carries the sentinel", file=sys.stderr)
raise SystemExit(1)
