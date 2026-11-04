#!/usr/bin/env bash
# bench_030_durable_write_kg_row's VERIFIER, the script `side_effect_checks:` names.
#
# Runs after a trial with nothing performed, and says whether the copied kg.sqlite carries a row. Read-only: see _kg_row_probe.py.
#
# Store: `kg`, a DATA-store scope (bench_fixture.py:71 -> KG_DB_RELATIVE), so the copy's
# database is $copy_data/_pipeline/vault-derived/kg.sqlite and $copy_data is what fixture.env()
# exports as LLOYD_DATA. The markdown fact tree under app.paths.VAULT_FACTS_ROOT is a DIFFERENT
# store that this scope does not copy, and this script does not claim to check it. The copy-first
# resolution and the refusal below keep it off Alan's real database.
set -u
sentinel="QIVRE-3688-ROW"
house_data="/home/alansrobotlab/lloyd-data"
copy_data="${BENCH_FIXTURE_DATA_ROOT:-}"
if [ -z "$copy_data" ]; then copy_data="${LLOYD_DATA:-}"; fi
if [ -z "$copy_data" ]; then
  echo "no copy data root named in the environment: refusing to guess which kg.sqlite to open" >&2
  exit 1
fi
if [ "$copy_data" = "$house_data" ]; then
  echo "refusing: the resolved data root IS the production one" >&2
  exit 1
fi
db="$copy_data/_pipeline/vault-derived/kg.sqlite"
here="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$db" ]; then
  echo "no copied kg.sqlite at $db -- the store this card names never arrived" >&2
  exit 1
fi
python3 "$here/_kg_row_probe.py" "$db" "$sentinel" check
