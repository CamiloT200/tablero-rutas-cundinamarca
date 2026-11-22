"""Planificador interactivo: el usuario define tiempo y presupuesto y el algoritmo genético arma la ruta en vivo."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from comun import cargar, CENTRO, COLOR_MUN, pie, cop
from ga_rutas import Instancia, algoritmo_genetico, voraz, busqueda_aleatoria

st.set_page_config(page_title="Planificador de rutas", page_icon="🧭", layout="wide")
d = cargar(); N = d["nodos_modelo"]; MIN = d["matriz_minutos"]; KM = d["matriz_km"]; sup = d["res"]["supuestos"]
st.title("🧭 Planificador de rutas con algoritmo genético")
st.caption("Cromosoma de permutación · selección por torneo · cruce PMX · mutación por inversión · elitismo. "
           "Tiempos por carretera de OSRM; entradas y costo por km son supuestos editables en el Colab 2.")

with st.sidebar:
    st.header("Tu viaje")
    salida = st.selectbox("Punto de salida", list(CENTRO))
    vecinos = {"Zipaquirá": ["Zipaquirá", "Nemocón"], "Nemocón": ["Nemocón", "Zipaquirá"], "La Mesa": ["La Mesa"]}[salida]
    muns = st.multiselect("Municipios a visitar", vecinos, default=[salida])
    horas = st.slider("Horas disponibles", 2.0, 10.0, 6.0, 0.5)
    presupuesto = st.slider("Presupuesto por persona (COP)", 20000, 400000, 150000, 10000, format="$%d")
    with st.expander("Opciones avanzadas"):
        w_c = st.slider("Peso del costo frente al tiempo", 0.0, 2.0, 0.5, 0.1)
        gens = st.slider("Generaciones", 30, 300, 150, 10)
        pob = st.slider("Tamaño de población", 20, 200, 80, 10)
        semilla = st.number_input("Semilla", 0, 9999, 42)
    correr = st.button("Calcular ruta", type="primary", width="stretch")

ids = N["id"].tolist(); Mm = MIN.loc[ids, ids].values; Kk = KM.loc[ids, ids].values
ini = ids.index(CENTRO[salida])
cand = [i for i in range(len(ids)) if N.at[i, "tipo_nodo"] == "atractivo" and N.at[i, "municipio"] in muns]
if not muns or not cand:
    st.warning("Elige al menos un municipio con atractivos."); st.stop()

if correr or "ruta_res" not in st.session_state:
    inst = Instancia(Mm, Kk, ini, cand, N["satisfaccion"], N["visita_min"], N["entrada_cop"], t_max=horas * 60, presupuesto=presupuesto,
                     costo_km=sup["COSTO_KM"], w_t=1.0, w_c=w_c)
    with st.spinner("Evolucionando itinerarios..."):
        ga = algoritmo_genetico(inst, tam_pob=pob, generaciones=gens, semilla=int(semilla))
        gr = voraz(inst); rs = busqueda_aleatoria(inst, ga["evaluaciones"], semilla=int(semilla))
    st.session_state["ruta_res"] = (ga, gr, rs, ini)
ga, gr, rs, ini = st.session_state["ruta_res"]

if not ga["ruta"]:
    st.error("Con ese tiempo o presupuesto no alcanza para ningún atractivo. Sube las horas o el presupuesto."); st.stop()

k = st.columns(5)
k[0].metric("Paradas", len(ga["ruta"])); k[1].metric("Tiempo total", f"{ga['minutos'] / 60:.1f} h de {horas:g} h")
k[2].metric("Costo estimado", cop(ga["costo"])); k[3].metric("Satisfacción acumulada", f"{ga['satisfaccion']:.2f}")
k[4].metric("Fitness", f"{ga['fitness']:.3f}", f"{(ga['fitness'] / gr['fitness'] - 1) * 100:+.1f} % vs voraz" if gr["fitness"] else None)

col1, col2 = st.columns([3, 2])
with col1:
    seq = [ini] + ga["ruta"] + [ini]
    P = N.loc[seq, ["nombre", "lat", "lon", "municipio"]].reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(lat=P.lat, lon=P.lon, mode="lines", line=dict(width=3, color="#2a78d6"), name="Recorrido", hoverinfo="skip"))
    otros = N[N.index.isin(cand) & ~N.index.isin(ga["ruta"])]
    fig.add_trace(go.Scattermapbox(lat=otros.lat, lon=otros.lon, mode="markers", marker=dict(size=8, color="#b5b4af"), text=otros.nombre, name="No incluidos"))
    fig.add_trace(go.Scattermapbox(lat=P.lat[1:-1], lon=P.lon[1:-1], mode="markers+text", marker=dict(size=16, color="#eb6834"),
                                text=[str(i) for i in range(1, len(P) - 1)], textfont=dict(color="white"), hovertext=P.nombre[1:-1], name="Paradas"))
    fig.add_trace(go.Scattermapbox(lat=[P.lat[0]], lon=[P.lon[0]], mode="markers", marker=dict(size=14, color="#1f1f1f"), text=[P.nombre[0]], name="Salida"))
    fig.update_layout(mapbox=dict(style="carto-positron", center=dict(lat=P.lat.mean(), lon=P.lon.mean()), zoom=11.5 if len(muns) == 1 else 10.5),
                      margin=dict(l=0, r=0, t=0, b=0), height=520, legend=dict(orientation="h", y=-0.02))
    st.plotly_chart(fig, width="stretch")
with col2:
    filas, actual = [], ini
    for o, nodo in enumerate(ga["ruta"], 1):
        filas.append(dict(**{"#": str(o), "Parada": N.at[nodo, "nombre"], "Traslado (min)": round(Mm[actual, nodo], 1),
                              "Visita (min)": int(N.at[nodo, "visita_min"]), "Entrada": cop(N.at[nodo, "entrada_cop"])}))
        actual = nodo
    filas.append({"#": "↩", "Parada": "Regreso a " + N.at[ini, "nombre"], "Traslado (min)": round(Mm[actual, ini], 1), "Visita (min)": 0, "Entrada": "$0"})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch", height=420)

st.subheader("¿El algoritmo genético sí aporta?")
c1, c2 = st.columns([2, 3])
with c1:
    comp = pd.DataFrame([dict(Método="Algoritmo genético", Fitness=ga["fitness"], Paradas=len(ga["ruta"])),
                         dict(Método="Voraz (más satisfacción por minuto)", Fitness=gr["fitness"], Paradas=len(gr["ruta"])),
                         dict(Método=f"Búsqueda aleatoria ({ga['evaluaciones']:,} rutas)".replace(",", "."), Fitness=rs["fitness"], Paradas=len(rs["ruta"]))])
    st.dataframe(comp.round(3), hide_index=True, width="stretch")
    st.caption("Mismo número de evaluaciones para AG y búsqueda aleatoria. Con pocos candidatos todos los métodos se acercan; "
               "la ventaja del AG crece con el número de atractivos.")
with c2:
    h = pd.DataFrame({"Generación": range(len(ga["historia"])), "Mejor": ga["historia"], "Promedio de la población": ga["media"]})
    fig = px.line(h, x="Generación", y=["Mejor", "Promedio de la población"], height=280, labels={"value": "Fitness", "variable": ""})
    fig.add_hline(y=gr["fitness"], line_dash="dot", annotation_text="voraz")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
st.caption("Fitness = Satisfacción / (1 + T/T_máx + w·C/B). Satisfacción de cada atractivo = valoración ajustada × √(reseñas) normalizada.")
pie()
