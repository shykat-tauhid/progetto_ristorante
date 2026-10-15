from datetime import datetime
from typing import Any

from database import get_connection


def create_reservation(
    customer_name: str,
    phone: str,
    reservation_at: datetime,
    party_size: int,
    notes: str,
) -> dict[str, Any]:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO reservations
                (customer_name, phone, reservation_at, party_size, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                customer_name,
                phone,
                reservation_at.isoformat(),
                party_size,
                notes,
            ),
        )
        row = connection.execute(
            "SELECT * FROM reservations WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    if row is None:
        raise RuntimeError("La prenotazione è stata inserita ma non è stato possibile rileggerla")
    return dict(row)


def update_reservation(
    reservation_id: int,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    columns = {
        "customer_name": "customer_name",
        "phone": "phone",
        "reservation_at": "reservation_at",
        "party_size": "party_size",
        "notes": "notes",
    }
    values: dict[str, Any] = {}
    for field, value in updates.items():
        if field not in columns:
            raise ValueError(f"Campo di prenotazione non aggiornabile: {field}")
        values[columns[field]] = value.isoformat() if isinstance(value, datetime) else value

    assignments = ", ".join(f"{column} = ?" for column in values)
    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE reservations SET {assignments} WHERE id = ?",
            [*values.values(), reservation_id],
        )
        if cursor.rowcount == 0:
            return None
        row = connection.execute(
            "SELECT * FROM reservations WHERE id = ?",
            (reservation_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def delete_reservation(reservation_id: int) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM reservations WHERE id = ?",
            (reservation_id,),
        )
    return cursor.rowcount > 0


def get_menu() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, description, category, price_cents, available
            FROM menu_items
            WHERE available = 1
            ORDER BY category, name
            """
        ).fetchall()
    return [
        {**dict(row), "available": bool(row["available"])}
        for row in rows
    ]


def get_restaurant_info() -> dict[str, str]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT name, address, phone, email, opening_hours
            FROM restaurant_info
            WHERE id = 1
            """
        ).fetchone()
    if row is None:
        raise RuntimeError("Le informazioni del ristorante non sono inizializzate")
    return dict(row)