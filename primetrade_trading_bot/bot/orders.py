from bot.client import BinanceClient, BinanceAPIError
from bot.logging_config import setup_logger

logger = setup_logger("orders")


def _log_result(result: dict, order_type: str) -> None:
    logger.info(
        "%s order OK | orderId=%s symbol=%s side=%s status=%s "
        "origQty=%s executedQty=%s avgPrice=%s",
        order_type,
        result.get("orderId"),
        result.get("symbol"),
        result.get("side"),
        result.get("status"),
        result.get("origQty"),
        result.get("executedQty"),
        result.get("avgPrice", "N/A"),
    )


def place_market_order(
    client: BinanceClient, symbol: str, side: str, quantity: float
) -> dict:
    logger.info("Placing MARKET %s | symbol=%s qty=%s", side, symbol, quantity)
    try:
        result = client.place_order(
            symbol=symbol, side=side, type="MARKET", quantity=quantity
        )
        _log_result(result, "MARKET")
        return result
    except BinanceAPIError:
        logger.error(
            "MARKET order FAILED | symbol=%s side=%s qty=%s", symbol, side, quantity
        )
        raise


def place_limit_order(
    client: BinanceClient,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    time_in_force: str = "GTC",
) -> dict:
    logger.info(
        "Placing LIMIT %s | symbol=%s qty=%s price=%s tif=%s",
        side, symbol, quantity, price, time_in_force,
    )
    try:
        result = client.place_order(
            symbol=symbol,
            side=side,
            type="LIMIT",
            quantity=quantity,
            price=price,
            timeInForce=time_in_force,
        )
        _log_result(result, "LIMIT")
        return result
    except BinanceAPIError:
        logger.error(
            "LIMIT order FAILED | symbol=%s side=%s qty=%s price=%s",
            symbol, side, quantity, price,
        )
        raise


def place_stop_market_order(
    client: BinanceClient,
    symbol: str,
    side: str,
    quantity: float,
    stop_price: float,
) -> dict:
    logger.info(
        "Placing STOP_MARKET %s | symbol=%s qty=%s stopPrice=%s",
        side, symbol, quantity, stop_price,
    )
    try:
        result = client.place_order(
            symbol=symbol,
            side=side,
            type="STOP_MARKET",
            quantity=quantity,
            stopPrice=stop_price,
        )
        _log_result(result, "STOP_MARKET")
        return result
    except BinanceAPIError:
        logger.error(
            "STOP_MARKET order FAILED | symbol=%s side=%s qty=%s stopPrice=%s",
            symbol, side, quantity, stop_price,
        )
        raise
