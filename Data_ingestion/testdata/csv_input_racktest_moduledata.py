"""Load rack test module data from multiple Excel files into TimescaleDB/PostgreSQL.

GitHub-safe version:
- No hard-coded DB host, port, username, password, or DB name.
- No database URL is printed to logs.
- Input path pattern and table name are provided by CLI arguments.

Required environment variables:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

Example:
    python csv_input_racktest_moduledata.py \
        --input-pattern "./data/module{number}.xlsx" \
        --module-count 17 \
        --table-name "rack_keti_2_moduledata" \
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


RACK_MODULE_COLUMNS = [
    "time",
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
    "min_temp",
    "max_temp",
    "min_volt",
    "min_volt_position",
    "max_volt",
    "max_volt_position",
    "module_voltage",
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
                "min_temp" float8 NULL,
                "max_temp" float8 NULL,
                "min_volt" float8 NULL,
                "min_volt_position" float8 NULL,
                "max_volt" float8 NULL,
                "max_volt_position" float8 NULL,
                "module_voltage" float8 NOT NULL,
                "module_num" int4 NOT NULL
            );
            """
        ).format(table_name=sql.Identifier(table_name))

        with psycopg2.connect(**self.config.psycopg_kwargs()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load rack module Excel files into TimescaleDB/PostgreSQL."
    )
    parser.add_argument(
        "--input-pattern",
        required=True,
        help=(
            "Input Excel path pattern. Use {number} where module number should be inserted. "
            "Example: './data/module{number}.xlsx'"
        ),
    )
    parser.add_argument(
        "--module-count",
        type=int,
        default=17,
        help="Number of module files to load. Default: 17",
    )
    parser.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="First module number. Default: 1",
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


def load_module_excel(input_pattern: str, module_number: int) -> pd.DataFrame:
    input_file = input_pattern.format(number=module_number)
    path = Path(input_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_excel(path)
    if len(data.columns) != len(RACK_MODULE_COLUMNS):
        raise ValueError(
            f"Unexpected column count in module {module_number}. "
            f"Expected {len(RACK_MODULE_COLUMNS)}, got {len(data.columns)}."
        )

    data.columns = RACK_MODULE_COLUMNS
    data["module_num"] = module_number
    return data


def main() -> None:
    args = parse_args()
    db = TimescaleDB(TimescaleConfig.from_env())

    if args.module_count < 1:
        raise ValueError("--module-count must be greater than 0.")

    if args.create_table:
        db.create_table(args.table_name)
        print(f"Table checked/created: {args.schema}.{args.table_name}")

    engine = db.create_engine()
    total_rows = 0
    for module_number in range(args.start_number, args.start_number + args.module_count):
        data = load_module_excel(args.input_pattern, module_number)
        total_rows += len(data)
        print(f"Module {module_number}: loaded rows {len(data):,}")

        data.to_sql(
            name=args.table_name,
            con=engine,
            schema=args.schema,
            if_exists=args.if_exists,
            index=False,
        )

    print(f"Upload completed. Total rows: {total_rows:,}")


if __name__ == "__main__":
    main()
