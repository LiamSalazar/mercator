# python -m mercator_data.ingest

from __future__ import annotations
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

from mercator_utils.io import load_config, ensure_dirs, save_parquet
from mercator_data.volatility import add_range_vol_features

SLICKCHARTS_URL = "https://www.slickcharts.com/nasdaq100"

def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def get_nasdaq100_tickers_from_slickcharts() -> list[str]:
    # Nota: requiere internet en tiempo de ejecución.
    tables = pd.read_html(SLICKCHARTS_URL)  # toma la primera tabla
    df = tables[0]
    # Slickcharts usa 'Symbol' con algunos puntos; Yahoo Finance usa '-'
    tickers = (
        df["Symbol"]
        .astype(str)
        .str.replace(r"\.", "-", regex=True)
        .str.strip()
        .tolist()
    )
    # Eliminar duplicados y entradas vacías
    tickers = [t for t in dict.fromkeys(tickers) if t and t.upper() != "N/A"]
    return tickers

def get_tickers_from_file(path: str | Path) -> list[str]:
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return (
        df[col].astype(str).str.replace(r"\.", "-", regex=True).str.strip().tolist()
    )

def load_universe(cfg: dict) -> list[str]:
    src = cfg["tickers_source"]["type"]
    if src == "slickcharts":
        try:
            return get_nasdaq100_tickers_from_slickcharts()
        except Exception as e:
            print(f"[WARN] Slickcharts falló: {e}. Intenta tickers_source.type: file", file=sys.stderr)
            raise
    elif src == "file":
        path = cfg["tickers_source"]["file_path"]
        return get_tickers_from_file(path)
    else:
        raise ValueError(f"tickers_source.type no soportado: {src}")

def download_ohlcv(tickers: list[str], start: str, end: str | None, auto_adjust: bool = True) -> pd.DataFrame:
    """
    Descarga OHLCV diario con yfinance y devuelve un DataFrame con columnas:
    ['date','Ticker','Open','High','Low','Close','Adj Close','Volume'] (las que existan).
    Soporta cambios de pandas (índice sin nombre) y silencia el future warning de stack.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        threads=True,
        progress=False,
    )

    # yfinance devuelve columnas MultiIndex (Field, Ticker). Queremos (Ticker, Field) y apilar por Ticker.
    data = data.swaplevel(axis=1)  # -> (Ticker, Field)
    data = data.sort_index(axis=1)

    # stack con la implementación nueva para evitar el FutureWarning
    df = data.stack(level=0, future_stack=True).reset_index()

    # En algunas versiones el índice no se llama 'Date'. Renombramos de forma robusta.
    # La primera columna tras reset_index() es la fecha, la segunda es el Ticker.
    first_col = df.columns[0]
    second_col = df.columns[1]
    rename_map = {first_col: "date", second_col: "Ticker"}
    df = df.rename(columns=rename_map)

    # Estandarizamos capitalización de campos OHLCV
    df = df.rename(columns=lambda c: c.title() if c not in ("date", "Ticker") else c)

    # Convertimos a datetime naive en UTC (sin tz)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

    # Reordenamos y filtramos columnas esperadas (algunas pueden no existir según el activo)
    keep = ["date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].sort_values(["Ticker", "date"]).reset_index(drop=True)

    return df


def main():
    # 1) Cargar config
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    cfg = load_config(cfg_path)

    start = cfg.get("start_date", "2015-01-01")
    end = cfg.get("end_date")
    if not end:
        end = _today_iso()

    # 2) Cargar universo (Nasdaq-100)
    tickers = load_universe(cfg)
    if len(tickers) == 0:
        raise RuntimeError("No se encontraron tickers para el universo configurado.")
    print(f"[INFO] Universo: {len(tickers)} tickers (ej: {tickers[:5]}...)")

    # 3) Descargar OHLCV
    df = download_ohlcv(tickers, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError("OHLCV vacío. Revisa tu conexión o la lista de tickers.")
    print(f"[INFO] OHLCV: {df.shape}")

    # 4) Agregar features de volatilidad de rango
    window = int(cfg.get("range_vol_window", 20))
    feats = add_range_vol_features(df, window=window)
    print(f"[INFO] Features: {feats.shape}")

    # 5) Guardar datasets
    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    features_dir  = Path(cfg.get("features_dir", "data/features"))
    ensure_dirs(processed_dir, features_dir)

    save_parquet(df, processed_dir / "market_ohlcv.parquet")
    save_parquet(feats, features_dir / "market_features.parquet")
    print(f"[OK] Guardado: {processed_dir/'market_ohlcv.parquet'}")
    print(f"[OK] Guardado: {features_dir/'market_features.parquet'}")

if __name__ == "__main__":
    main()
