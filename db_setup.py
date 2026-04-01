"""
db_setup.py
-----------
Script à exécuter UNE SEULE FOIS pour convertir les fichiers Excel en SQLite.
Usage : python db_setup.py
"""

import pandas as pd
import sqlite3
from config import DB_PATH

EXCEL_FILES = {
    "direction":    "data/DIRECTION.xlsx",
    "departement":  "data/DEPARTEMENT.xlsx",
    "service":      "data/SERVICE.xlsx",
    "poste":        "data/POSTE.xlsx",
}

def build_database():
    conn = sqlite3.connect(DB_PATH)

    for table_name, path in EXCEL_FILES.items():
        df = pd.read_excel(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"✅ {table_name}: {len(df)} lignes importées")

    # Index pour accélérer les recherches
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dir_id    ON direction(ID)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dep_id    ON departement(ID)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_svc_id    ON service(ID)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_poste_id  ON poste(ID)")
    conn.commit()
    conn.close()
    print(f"\n✅ Base SQLite créée : {DB_PATH}")

if __name__ == "__main__":
    build_database()
