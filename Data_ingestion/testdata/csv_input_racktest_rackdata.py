"""Load rack test rack data from Excel into TimescaleDB/PostgreSQL.

GitHub-safe version:
- No hard-coded DB host, port, username, password, or DB name.
- No database URL is printed to logs.
- Input path and table name are provided by CLI arguments.

Required environment variables:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

Example:
    python csv_input_racktest_rackdata.py \
        --input-file "./data/rack_data.xlsx" \
        --table-name "rack_keti_2_rackdata" \
        --create-table
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


RACK_DATA_COLUMNS = [
    "time",
    "min_temp",
    "min_temp_position",
    "max_temp",
    "max_temp_position",
    "min_volt",
    "min_volt_position",
    "max_volt",
    "max_volt_position",
    "rack_voltage",
    "rack_current",
    "rack_soc",
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
                "min_temp" float8 NOT NULL,
                "min_temp_position" float8 NOT NULL,
                "max_temp" float8 NOT NULL,
                "max_temp_position" float8 NOT NULL,
                "min_volt" float8 NOT NULL,
                "min_volt_position" float8 NOT NULL,
                "max_volt" float8 NOT NULL,
                "max_volt_position" float8 NOT NULL,
                "rack_voltage" float8 NOT NULL,
                "rack_current" float8 NOT NULL,
                "rack_soc" float8 NOT NULL
            );
            """
        ).format(table_name=sql.Identifier(table_name))

        with psycopg2.connect(**self.config.psycopg_kwargs()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load rack test rack data Excel file into TimescaleDB/PostgreSQL."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the rack data Excel file. Do not commit private data files to GitHub.",
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
    if len(data.columns) != len(RACK_DATA_COLUMNS):
        raise ValueError(
            f"Unexpected column count. Expected {len(RACK_DATA_COLUMNS)}, "
            f"got {len(data.columns)}."
        )

    data.columns = RACK_DATA_COLUMNS
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
