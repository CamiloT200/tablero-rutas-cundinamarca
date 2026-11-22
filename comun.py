"""Carga de datos y utilidades compartidas por las páginas del tablero."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
COLOR_MUN = {"Zipaquirá": "#2a78d6", "Nemocón": "#1baf7a", "La Mesa": "#eb6834"}
CENTRO = {"Zipaquirá": "CENTRO_ZIPAQUIRA", "Nemocón": "CENTRO_NEMOCON", "La Mesa": "CENTRO_LAMESA"}
GRUPO_TXT = {"atractivo": "Atractivo", "gastronomia": "Gastronomía", "alojamiento": "Alojamiento", "comercio": "Comercio", "otro": "Otro"}


@st.cache_data
def cargar():
    d = {}
    for f in DATA.glob("*.csv"):
        idx = 0 if f.stem in ("matriz_minutos", "matriz_km", "correlacion_pearson") else None
        d[f.stem] = pd.read_csv(f, index_col=idx)
    d["res"] = json.loads((DATA / "resultados.json").read_text(encoding="utf-8"))
    return d


def cop(v):
    return f"${v:,.0f}".replace(",", ".")


def pie():
    st.caption("Fuentes: Google Maps (Instant Data Scraper, nov. 2026) · OSRM · RNT (datos.gov.co). "
               "Proyecto ML y Algoritmos Genéticos · Universidad de Cundinamarca · Grupo 901.")
