"""Load module test data from Excel into TimescaleDB/PostgreSQL.

GitHub-safe version:
- No hard-coded DB host, port, username, password, or DB name.
- No database URL is printed to logs.
- Input path and table name are provided by CLI arguments.

Required environment variables:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

Example:
    python csv_input_moduletest.py \
        --input-file "./data/module_test.xlsx" \
        --table-name "module_snu_1"
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


MODULE_TEST_COLUMNS = [
    "time",
    "module_voltage",
    "core1_voltage",
    "core2_voltage",
    "core3_voltage",
    "core4_voltage",
    "core5_voltage",
    "core6_voltage",
    "core7_voltage",
    "core8_voltage",
    "core9_voltage",
    "core10_voltage",
    "core11_voltage",
    "core12_voltage",
    "current",
    "soc",
    "min_temp",
    "max_temp",
    "min_temp_position",
    "max_temp_position",
    "min_volt",
    "max_volt",
]


@dataclass(frozen=True)
class TimescaleConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "TimescaleConfig":
        missing = [
            name
            for name in ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        return cls(
            host=os.environ["DB_HOST"],
            port=int(os.environ["DB_PORT"]),
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            database=os.environ["DB_NAME"],
        )

    def psycopg_kwargs(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "dbname": self.database,
        }

    def sqlalchemy_url(self) -> URL:
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )


class TimescaleDB:
    def __init__(self, config: TimescaleConfig):
        self.config = config

    def create_engine(self):
        return create_engine(self.config.sqlalchemy_url())

    def create_table(self, table_name: str) -> None:
        query = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                "time" timestamptz NOT NULL,
                "module_voltage" float8 NOT NULL,
                "core1_voltage" float8 NULL,
                "core2_voltage" float8 NULL,
                "core3_voltage" float8 NULL,
                "core4_voltage" float8 NULL,
                "core5_voltage" float8 NULL,
                "core6_voltage" float8 NULL,
                "core7_voltage" float8 NULL,
                "core8_voltage" float8 NULL,
                "core9_voltage" float8 NULL,
                "core10_voltage" float8 NULL,
                "core11_voltage" float8 NULL,
                "core12_voltage" float8 NULL,
                "current" float8 NULL,
                "soc" float8 NULL,
                "min_temp" float8 NULL,
                "max_temp" float8 NULL,
                "min_temp_position" float8 NULL,
                "max_temp_position" float8 NULL,
                "min_volt" float8 NULL,
                "max_volt" float8 NULL
            );
            """
        ).format(table_name=sql.Identifier(table_name))

        with psycopg2.connect(**self.config.psycopg_kwargs()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load module test Excel file into TimescaleDB/PostgreSQL."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the module test Excel file. Do not commit private data files to GitHub.",
    )
    parser.add_argument(
        "--table-name",
        required=True,
        help="Destination table name.",
    )
    parser.add_argument(
        "--schema",
        default="public",
        help="Destination schema name. Default: public",
    )
    parser.add_argument(
        "--create-table",
        action="store_true",
        help="Create the destination table if it does not exist before inserting data.",
    )
    parser.add_argument(
        "--if-exists",
        choices=["fail", "replace", "append"],
        default="append",
        help="pandas.to_sql if_exists option. Default: append",
    )
    return parser.parse_args()


def load_excel(input_file: str) -> pd.DataFrame:
    path = Path(input_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_excel(path)
    if len(data.columns) != len(MODULE_TEST_COLUMNS):
        raise ValueError(
            f"Unexpected column count. Expected {len(MODULE_TEST_COLUMNS)}, "
            f"got {len(data.columns)}."
        )

    data.columns = MODULE_TEST_COLUMNS
    return data


def main() -> None:
    args = parse_args()
    db = TimescaleDB(TimescaleConfig.from_env())

    data = load_excel(args.input_file)
    print(f"Loaded rows: {len(data):,}")

    if args.create_table:
        db.create_table(args.table_name)
        print(f"Table checked/created: {args.schema}.{args.table_name}")

    data.to_sql(
        name=args.table_name,
        con=db.create_engine(),
        schema=args.schema,
        if_exists=args.if_exists,
        index=False,
    )
    print("Upload completed.")


if __name__ == "__main__":
    main()
