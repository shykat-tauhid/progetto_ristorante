from contextlib import closing, contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Generator


DATABASE_PATH = Path(__file__).resolve().with_name("restaurant.db")


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    with closing(connection):
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                reservation_at TEXT NOT NULL,
                party_size INTEGER NOT NULL CHECK (party_size > 0),
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_reservations_reservation_at
                ON reservations (reservation_at);

            CREATE TABLE IF NOT EXISTS menu_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL,
                price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
                available INTEGER NOT NULL DEFAULT 1 CHECK (available IN (0, 1))
            );

            CREATE TABLE IF NOT EXISTS restaurant_info (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT NOT NULL DEFAULT '',
                address TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                opening_hours TEXT NOT NULL DEFAULT ''
            );
            """
        )
        connection.execute(
            """
            INSERT INTO restaurant_info (id, name, address, phone, email, opening_hours)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = CASE WHEN restaurant_info.name = '' THEN excluded.name
                            ELSE restaurant_info.name END,
                address = CASE WHEN restaurant_info.address = '' THEN excluded.address
                               ELSE restaurant_info.address END,
                phone = CASE WHEN restaurant_info.phone = '' THEN excluded.phone
                             ELSE restaurant_info.phone END,
                email = CASE WHEN restaurant_info.email = '' THEN excluded.email
                             ELSE restaurant_info.email END,
                opening_hours = CASE WHEN restaurant_info.opening_hours = ''
                                     THEN excluded.opening_hours
                                     ELSE restaurant_info.opening_hours END
            """,
            (
                os.getenv("RESTAURANT_NAME", "").strip(),
                os.getenv("RESTAURANT_ADDRESS", "").strip(),
                os.getenv("RESTAURANT_PHONE", "").strip(),
                os.getenv("RESTAURANT_EMAIL", "").strip(),
                os.getenv("RESTAURANT_OPENING_HOURS", "").strip(),
            ),
        )
        has_menu = connection.execute(
            "SELECT 1 FROM menu_items LIMIT 1"
        ).fetchone()
        if has_menu is None:
            connection.executemany(
                """
                INSERT INTO menu_items (name, description, category, price_cents)
                VALUES (?, ?, ?, ?)
                """,
                [
                    ("Pasta del giorno", "Piatto di pasta preparato in giornata", "Primi", 1200),
                    ("Insalata mista", "Insalata fresca di stagione", "Contorni", 700),
                    ("Dolce della casa", "Dessert preparato in casa", "Dessert", 600),
                ],
            )