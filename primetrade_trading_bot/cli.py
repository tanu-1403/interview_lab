#!/usr/bin/env python3
"""Binance Futures Testnet Trading Bot — CLI entry point."""
import os
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bot.client import BinanceAPIError, BinanceClient
from bot.logging_config import setup_logger
from bot.orders import place_limit_order, place_market_order, place_stop_market_order
from bot.validators import (
    validate_order_type,
    validate_price,
    validate_quantity,
    validate_side,
    validate_stop_price,
    validate_tif,
)

load_dotenv()
app = typer.Typer(
    name="trading-bot",
    help="Binance Futures Testnet Trading Bot — place MARKET, LIMIT, and STOP_MARKET orders.",
    add_completion=False,
)
console = Console()
logger = setup_logger("cli")


def _client() -> BinanceClient:
    key = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_API_SECRET", "")
    if not key or not secret:
        console.print(Panel(
            "[bold red]Missing credentials.[/bold red]\n"
            "Copy [cyan].env.example[/cyan] → [cyan].env[/cyan] "
            "and fill in your testnet API key + secret.",
            border_style="red",
        ))
        raise typer.Exit(1)
    return BinanceClient(api_key=key, api_secret=secret)


def _summary_table(symbol, side, order_type, quantity, price=None, stop_price=None):
    t = Table(title="Order Request", show_header=False, border_style="bright_black")
    t.add_column("Field", style="dim", width=16)
    t.add_column("Value")
    t.add_row("Symbol", f"[bold]{symbol}[/bold]")
    color = "green" if side == "BUY" else "red"
    t.add_row("Side", f"[{color} bold]{side}[/{color} bold]")
    t.add_row("Type", order_type)
    t.add_row("Quantity", str(quantity))
    if price:
        t.add_row("Price", str(price))
    if stop_price:
        t.add_row("Stop Price", str(stop_price))
    console.print(t)


def _response_table(result: dict):
    t = Table(title="Order Response", show_header=False, border_style="green")
    t.add_column("Field", style="dim", width=16)
    t.add_column("Value")
    for label, key in [
        ("Order ID", "orderId"),
        ("Symbol", "symbol"),
        ("Status", "status"),
        ("Side", "side"),
        ("Type", "type"),
        ("Orig Qty", "origQty"),
        ("Executed Qty", "executedQty"),
        ("Avg Price", "avgPrice"),
        ("Price", "price"),
        ("Time In Force", "timeInForce"),
    ]:
        val = result.get(key)
        if val not in (None, "", "0"):
            t.add_row(label, str(val))
    console.print(t)


@app.command()
def place_order(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Trading pair, e.g. BTCUSDT"),
    side: str = typer.Option(..., "--side", help="BUY or SELL"),
    order_type: str = typer.Option(..., "--type", "-t", help="MARKET | LIMIT | STOP_MARKET"),
    quantity: float = typer.Option(..., "--qty", "-q", help="Order quantity"),
    price: Optional[float] = typer.Option(None, "--price", "-p", help="Limit price (LIMIT orders only)"),
    stop_price: Optional[float] = typer.Option(None, "--stop-price", help="Stop price (STOP_MARKET only)"),
    tif: str = typer.Option("GTC", "--tif", help="Time in force: GTC | IOC | FOK"),
):
    """Place a futures order on Binance Testnet (MARKET, LIMIT, or STOP_MARKET)."""
    console.rule("[bold]Binance Futures Testnet Bot[/bold]")

    try:
        side = validate_side(side)
        order_type = validate_order_type(order_type)
        quantity = validate_quantity(quantity)
        price = validate_price(price, order_type)
        stop_price = validate_stop_price(stop_price, order_type)
        tif = validate_tif(tif)
    except typer.BadParameter as exc:
        console.print(f"[bold red]Validation error:[/bold red] {exc}")
        logger.warning("Validation failed: %s", exc)
        raise typer.Exit(1)

    symbol = symbol.upper()
    _summary_table(symbol, side, order_type, quantity, price, stop_price)
    logger.info(
        "CLI request | symbol=%s side=%s type=%s qty=%s price=%s stop=%s",
        symbol, side, order_type, quantity, price, stop_price,
    )

    client = _client()
    try:
        if order_type == "MARKET":
            result = place_market_order(client, symbol, side, quantity)
        elif order_type == "LIMIT":
            result = place_limit_order(client, symbol, side, quantity, price, tif)
        else:
            result = place_stop_market_order(client, symbol, side, quantity, stop_price)

        _response_table(result)
        console.print(Panel(
            "[bold green]Order placed successfully![/bold green]",
            border_style="green",
        ))

    except BinanceAPIError as exc:
        console.print(Panel(
            f"[bold red]API error [{exc.code}]:[/bold red] {exc.message}",
            border_style="red",
        ))
        logger.error("Order failed: %s", exc)
        raise typer.Exit(1)
    except Exception as exc:
        console.print(Panel(
            f"[bold red]Unexpected error:[/bold red] {exc}",
            border_style="red",
        ))
        logger.exception("Unexpected error during order placement")
        raise typer.Exit(1)


@app.command()
def account():
    """Show testnet account balances and unrealized PnL."""
    console.rule("[bold]Account Balances[/bold]")
    client = _client()
    try:
        info = client.get_account()
        assets = [a for a in info.get("assets", []) if float(a.get("walletBalance", 0)) > 0]
        if not assets:
            console.print("[yellow]No funded assets found.[/yellow]")
            return
        t = Table(border_style="bright_black")
        t.add_column("Asset", style="bold")
        t.add_column("Wallet Balance", justify="right")
        t.add_column("Unrealized PnL", justify="right")
        for a in assets:
            pnl = float(a.get("unrealizedProfit", 0))
            pnl_color = "green" if pnl >= 0 else "red"
            t.add_row(
                a["asset"],
                a["walletBalance"],
                f"[{pnl_color}]{pnl:.4f}[/{pnl_color}]",
            )
        console.print(t)
    except BinanceAPIError as exc:
        console.print(f"[bold red]API error:[/bold red] {exc.message}")
        raise typer.Exit(1)


@app.command()
def ping():
    """Check connectivity to Binance Futures Testnet."""
    client = _client()
    try:
        ok = client.ping()
        if ok:
            console.print("[bold green]✓ Testnet is reachable.[/bold green]")
        else:
            console.print("[bold red]✗ Testnet unreachable.[/bold red]")
    except Exception as exc:
        console.print(f"[bold red]Ping failed:[/bold red] {exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
