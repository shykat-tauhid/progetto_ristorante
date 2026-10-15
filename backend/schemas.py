from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PhoneText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=30)]


class ReservationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: Annotated[NonEmptyText, StringConstraints(max_length=120)]
    phone: PhoneText
    reservation_at: datetime
    party_size: int = Field(ge=1)
    notes: str = Field(default="", max_length=1000)


class ReservationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: Annotated[NonEmptyText, StringConstraints(max_length=120)] | None = None
    phone: PhoneText | None = None
    reservation_at: datetime | None = None
    party_size: int | None = Field(default=None, ge=1)
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_update(self) -> "ReservationUpdate":
        updates = self.model_dump(exclude_unset=True)
        if not updates:
            raise ValueError("Specificare almeno un campo da modificare")
        if any(value is None for value in updates.values()):
            raise ValueError("I campi forniti non possono essere null")
        return self


class ReservationResponse(BaseModel):
    id: int
    customer_name: str
    phone: str
    reservation_at: datetime
    party_size: int
    notes: str


class MenuItemResponse(BaseModel):
    id: int
    name: str
    description: str
    category: str
    price_cents: int
    available: bool


class RestaurantInfoResponse(BaseModel):
    name: str
    address: str
    phone: str
    email: str
    opening_hours: str

