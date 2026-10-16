from contextlib import closing, contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Generator
import json
import os


DATABASE_PATH = Path(__file__).resolve().with_name("restaurant.db")


MENU_SEED = [
    ("Bruschette al pomodoro", "Pane casereccio tostato, pomodoro fresco, aglio, basilico e olio extravergine", "Antipasti", 700, ["glutine"], True, True),
    ("Burrata pugliese", "Burrata con pomodorini datterini, basilico e olio extravergine", "Antipasti", 1200, ["latte"], True, False),
    ("Insalata di polpo e patate", "Polpo tiepido, patate, sedano, olive taggiasche e limone", "Antipasti", 1400, ["molluschi", "sedano"], False, False),
    ("Parmigiana di melanzane", "Melanzane, passata di pomodoro, mozzarella e parmigiano, cotta al forno", "Antipasti", 1100, ["latte", "glutine", "sedano"], True, False),
    ("Carpaccio di manzo", "Manzo crudo affettato sottile, rucola, scaglie di grana e limone", "Antipasti", 1400, ["latte"], False, False),
    ("Spaghetti alla carbonara", "Guanciale croccante, tuorlo d'uovo, pecorino romano e pepe nero", "Primi", 1300, ["glutine", "uova", "latte"], False, False),
    ("Spaghetti alle vongole", "Vongole veraci, aglio, prezzemolo, vino bianco e peperoncino", "Primi", 1600, ["glutine", "molluschi", "solfiti"], False, False),
    ("Tagliatelle al ragù", "Pasta all'uovo con ragù bolognese cotto lentamente e parmigiano", "Primi", 1300, ["glutine", "uova", "sedano", "latte"], False, False),
    ("Rigatoni all'amatriciana", "Guanciale, pomodoro San Marzano e pecorino romano", "Primi", 1200, ["glutine", "latte"], False, False),
    ("Cacio e pepe", "Tonnarelli con pecorino romano e pepe nero", "Primi", 1200, ["glutine", "latte"], True, False),
    ("Gnocchi al pomodoro e basilico", "Gnocchi di patate fatti in casa con sugo di pomodoro fresco e basilico", "Primi", 1100, ["glutine", "uova", "sedano"], True, False),
    ("Risotto ai funghi porcini", "Riso Carnaroli mantecato al burro e parmigiano con porcini, brodo vegetale", "Primi", 1500, ["latte", "sedano", "solfiti"], True, False),
    ("Lasagna alla bolognese", "Sfoglia all'uovo, ragù, besciamella e parmigiano, gratinata al forno", "Primi", 1300, ["glutine", "uova", "latte", "sedano"], False, False),
    ("Margherita", "Pomodoro, mozzarella fior di latte e basilico", "Pizze", 900, ["glutine", "latte"], True, False),
    ("Marinara", "Pomodoro, aglio, origano e olio extravergine", "Pizze", 800, ["glutine"], True, True),
    ("Verdure grigliate", "Zucchine, melanzane e peperoni alla griglia", "Contorni", 700, [], True, True),
    ("Insalata mista", "Lattuga, pomodori, carote e finocchio, condita con olio e aceto", "Contorni", 600, ["solfiti"], True, True),
    ("Spinaci al burro", "Spinaci saltati al burro", "Contorni", 650, ["latte"], True, False),
    ("Tiramisù", "Savoiardi al caffè, crema al mascarpone e cacao", "Dolci", 700, ["glutine", "uova", "latte"], True, False),
    ("Panna cotta ai frutti di bosco", "Panna cotta con coulis di frutti di bosco", "Dolci", 650, ["latte"], True, False),
    ("Cannolo siciliano", "Cialda croccante con crema di ricotta, gocce di cioccolato e granella di pistacchio", "Dolci", 700, ["glutine", "latte", "frutta a guscio"], True, False),
    ("Torta della nonna", "Frolla con crema pasticcera e pinoli", "Dolci", 700, ["glutine", "uova", "latte", "frutta a guscio"], True, False),
    ("Sorbetto al limone", "Sorbetto artigianale al limone", "Dolci", 500, [], True, True),
    ("Acqua naturale o frizzante (0,75 l)", "", "Bevande", 300, [], True, True),
    ("Bibita in lattina (33 cl)", "Coca-Cola, Fanta o tè freddo", "Bevande", 350, [], True, True),
    ("Birra alla spina (0,4 l)", "Birra chiara", "Bevande", 500, ["glutine"], True, False),
    ("Vino della casa (calice)", "Bianco o rosso", "Bevande", 500, ["solfiti"], True, True),
    ("Chianti Classico (bottiglia)", "Vino rosso toscano", "Bevande", 2200, ["solfiti"], True, True),
    ("Caffè espresso", "", "Bevande", 150, [], True, True),
    ("Limoncello", "", "Bevande", 400, [], True, True),
]


_OLD_PLACEHOLDER_MENU = {"Pasta del giorno", "Insalata mista", "Dolce della casa"}

_MENU_NEW_COLUMNS = {
    "allergens": "TEXT NOT NULL DEFAULT '[]'",
    "is_vegetarian": "INTEGER NOT NULL DEFAULT 0 CHECK (is_vegetarian IN (0, 1))",
    "is_vegan": "INTEGER NOT NULL DEFAULT 0 CHECK (is_vegan IN (0, 1))",
}


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
                available INTEGER NOT NULL DEFAULT 1 CHECK (available IN (0, 1)),
                allergens TEXT NOT NULL DEFAULT '[]',
                is_vegetarian INTEGER NOT NULL DEFAULT 0 CHECK (is_vegetarian IN (0, 1)),
                is_vegan INTEGER NOT NULL DEFAULT 0 CHECK (is_vegan IN (0, 1))
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

        # Migrazione: se menu_items esiste già con il vecchio schema, aggiungo le colonne nuove.
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(menu_items)")
        }
        for column, definition in _MENU_NEW_COLUMNS.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE menu_items ADD COLUMN {column} {definition}"
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

        # Popolo il menu se è vuoto oppure contiene solo il vecchio menu segnaposto.
        current_names = {
            row[0] for row in connection.execute("SELECT name FROM menu_items")
        }
        if not current_names or current_names == _OLD_PLACEHOLDER_MENU:
            connection.execute("DELETE FROM menu_items")
            connection.executemany(
                """
                INSERT INTO menu_items
                    (name, description, category, price_cents,
                     allergens, is_vegetarian, is_vegan)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (name, desc, cat, price, json.dumps(allergens, ensure_ascii=False),
                     int(veg), int(vegan))
                    for name, desc, cat, price, allergens, veg, vegan in MENU_SEED
                ],
            )
