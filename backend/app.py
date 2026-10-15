from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Response, status

import repository
from database import initialize_database
from schemas import (
    MenuItemResponse,
    ReservationCreate,
    ReservationResponse,
    ReservationUpdate,
    RestaurantInfoResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    initialize_database()
    yield


app = FastAPI(
    title="Restaurant API",
    description="API per prenotazioni, menu e informazioni del ristorante.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/reservations",
    response_model=ReservationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_reservation(payload: ReservationCreate) -> dict[str, object]:
    return repository.create_reservation(**payload.model_dump())


@app.patch(
    "/reservations/{reservation_id}",
    response_model=ReservationResponse,
)
def update_reservation(
    reservation_id: int,
    payload: ReservationUpdate,
) -> dict[str, object]:
    reservation = repository.update_reservation(
        reservation_id,
        payload.model_dump(exclude_unset=True),
    )
    if reservation is None:
        raise HTTPException(status_code=404, detail="Prenotazione non trovata")
    return reservation


@app.delete(
    "/reservations/{reservation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_reservation(reservation_id: int) -> Response:
    deleted = repository.delete_reservation(reservation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prenotazione non trovata")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/menu", response_model=list[MenuItemResponse])
def get_menu() -> list[dict[str, object]]:
    return repository.get_menu()


@app.get("/restaurant-info", response_model=RestaurantInfoResponse)
def get_restaurant_info() -> dict[str, str]:
    return repository.get_restaurant_info()