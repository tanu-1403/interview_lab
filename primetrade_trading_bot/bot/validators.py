from typing import Optional
import typer

VALID_SIDES = {"BUY", "SELL"}
VALID_ORDER_TYPES = {"MARKET", "LIMIT", "STOP_MARKET"}
VALID_TIF = {"GTC", "IOC", "FOK"}


def validate_side(side: str) -> str:
    side = side.upper().strip()
    if side not in VALID_SIDES:
        raise typer.BadParameter(
            f"Side must be one of: {', '.join(sorted(VALID_SIDES))}"
        )
    return side


def validate_order_type(order_type: str) -> str:
    order_type = order_type.upper().strip()
    if order_type not in VALID_ORDER_TYPES:
        raise typer.BadParameter(
            f"Order type must be one of: {', '.join(sorted(VALID_ORDER_TYPES))}"
        )
    return order_type


def validate_quantity(quantity: float) -> float:
    if quantity <= 0:
        raise typer.BadParameter("Quantity must be a positive number.")
    return round(quantity, 8)


def validate_price(price: Optional[float], order_type: str) -> Optional[float]:
    if order_type == "LIMIT":
        if price is None or price <= 0:
            raise typer.BadParameter(
                "A positive --price is required for LIMIT orders."
            )
    if order_type == "MARKET" and price is not None:
        raise typer.BadParameter("Do not supply --price for MARKET orders.")
    return round(price, 8) if price else None


def validate_stop_price(stop_price: Optional[float], order_type: str) -> Optional[float]:
    if order_type == "STOP_MARKET":
        if stop_price is None or stop_price <= 0:
            raise typer.BadParameter(
                "A positive --stop-price is required for STOP_MARKET orders."
            )
    return round(stop_price, 8) if stop_price else None


def validate_tif(tif: str) -> str:
    tif = tif.upper().strip()
    if tif not in VALID_TIF:
        raise typer.BadParameter(
            f"Time-in-force must be one of: {', '.join(sorted(VALID_TIF))}"
        )
    return tif
