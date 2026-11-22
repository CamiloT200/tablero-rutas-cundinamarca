"""
Tablero gerencial · Segmentación turística y rutas óptimas (Zipaquirá, Nemocón y La Mesa).
Preguntas que responde: ¿cómo es la oferta y cómo se agrupa?, ¿dónde invertir?, ¿qué ruta recomendar?
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from comun import cargar, COLOR_MUN, GRUPO_TXT, pie

st.set_page_config(page_title="Turismo Cundinamarca · ML + AG", page_icon="🧭", layout="wide")
d = cargar(); L = d["lugares_modelo"].copy(); res = d["res"]
L["Tipo"] = L["grupo"].map(GRUPO_TXT)

st.title("Segmentación turística y rutas óptimas")
st.caption(f"{len(L)} lugares de Google Maps en Zipaquirá, Nemocón (Sabana Centro) y La Mesa (Tequendama) · "
           "clustering K-Means, PCA, Random Forest y algoritmo genético.")

with st.sidebar:
    st.header("Filtros")
    muns = st.multiselect("Municipio", list(COLOR_MUN), default=list(COLOR_MUN))
    tipos = st.multiselect("Tipo de lugar", sorted(L["Tipo"].unique()), default=["Atractivo", "Gastronomía", "Alojamiento"])
F = L[L.municipio.isin(muns) & L.Tipo.isin(tipos)]

c = st.columns(5)
c[0].metric("Lugares", len(F))
c[1].metric("Estrellas promedio", f"{F.estrellas.mean():.2f}".replace(".", ","))
c[2].metric("Con 4,0 o más", f"{(F.estrellas >= 4).mean():.0%}")
c[3].metric("Mediana de reseñas", f"{F.n_resenas.median():,.0f}".replace(",", "."))
c[4].metric("Segmentos (K-Means)", res["kmeans"]["k"])

t1, t2, t3, t4 = st.tabs(["🗺️ Mapa de segmentos", "🏗️ Dónde invertir", "🧭 Rutas recomendadas", "🏨 Oferta formal (RNT)"])

with t1:
    st.subheader("¿Cómo se agrupa la oferta?")
    region = st.radio("Región", ["Sabana Centro", "Tequendama"], horizontal=True)
    M = F[F.provincia == region]
    if len(M):
        fig = px.scatter_mapbox(M, lat="lat", lon="lon", color="segmento", size=M["n_resenas"].clip(upper=5000) ** .5 + 3,
                             hover_name="nombre", hover_data={"Tipo": True, "estrellas": True, "n_resenas": True, "lat": False, "lon": False},
                             zoom=11 if region == "Tequendama" else 10.5, height=520, mapbox_style="carto-positron")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.05, title=None))
        st.plotly_chart(fig, width="stretch")
    P = d["clusters_perfil"].copy()
    P = P.rename(columns={"cluster": "Cluster", "segmento": "Segmento", "n": "Lugares", "estrellas": "Estrellas", "n_resenas": "Reseñas (media)",
                          "nivel_precio": "Nivel de precio", "dist_centro_km": "Km al centro", "densidad_1km": "Vecinos a 1 km"})
    st.markdown("**Perfil de cada segmento** (promedios). Tamaño del punto = popularidad.")
    st.dataframe(P[["Cluster", "Segmento", "Lugares", "Estrellas", "Reseñas (media)", "Nivel de precio", "Km al centro", "Vecinos a 1 km",
                    "% atractivo", "% gastronomía", "% alojamiento"]].round(2), hide_index=True, width="stretch")
    st.caption(f"K-Means con k = {res['kmeans']['k']} elegido por silueta ({res['kmeans']['silueta']:.2f}). Segmentación moderada: "
               "los grupos ayudan a describir la oferta, no son fronteras rígidas.")

with t2:
    st.subheader("Alta valoración, baja accesibilidad")
    S = d["segmento_inversion"]
    st.markdown("Atractivos en el **cuartil superior de valoración** (estrellas ajustadas por número de reseñas) que están a **más de 10 minutos** "
                "de su casco urbano. Son candidatos para proyectos de vías, señalización o transporte.")
    st.dataframe(S.rename(columns={"municipio": "Municipio", "nombre": "Atractivo", "estrellas": "Estrellas", "n_resenas": "Reseñas",
                                   "estrellas_bayes": "Valoración ajustada", "min_acceso": "Min. desde el casco urbano",
                                   "dist_centro_km": "Km en línea recta"})[["Municipio", "Atractivo", "Estrellas", "Reseñas", "Valoración ajustada",
                                                                          "Min. desde el casco urbano", "Km en línea recta"]].round(2),
                 hide_index=True, width="stretch")
    A = L[L.grupo == "atractivo"]
    fig = px.scatter(A, x="min_acceso", y="estrellas_bayes", color="municipio", color_discrete_map=COLOR_MUN, hover_name="nombre",
                     size=A["n_resenas"] ** .5 + 2, labels={"min_acceso": "Minutos desde el casco urbano", "estrellas_bayes": "Valoración ajustada"}, height=420)
    fig.add_vline(x=10, line_dash="dot"); fig.add_hline(y=A["estrellas_bayes"].quantile(.75), line_dash="dot")
    st.plotly_chart(fig, width="stretch")
    st.caption("Cuadrante superior derecho = alta valoración y baja accesibilidad.")

with t3:
    st.subheader("Itinerarios óptimos por escenario (algoritmo genético)")
    E = d["ga_escenarios"]; R = d["ga_rutas_detalle"]
    esc = st.selectbox("Escenario", E["escenario"].tolist())
    e = E[E.escenario == esc].iloc[0]; r = R[R.escenario == esc]
    k = st.columns(4)
    k[0].metric("Paradas", int(e.paradas)); k[1].metric("Tiempo total", f"{e.minutos / 60:.1f} h")
    k[2].metric("Costo estimado", f"${e.costo:,.0f}".replace(",", ".")); k[3].metric("Mejora vs. voraz", f"{e.mejora_vs_voraz_pct:+.1f} %")
    st.dataframe(r[["orden", "nombre", "traslado_min", "visita_min", "entrada_cop"]].rename(columns={"orden": "#", "nombre": "Parada",
                 "traslado_min": "Traslado (min)", "visita_min": "Visita (min)", "entrada_cop": "Entrada (COP)"}).round(1), hide_index=True, width="stretch")
    st.info("Para armar una ruta propia con otro tiempo o presupuesto, use la página **Planificador de rutas** (barra lateral).")

with t4:
    st.subheader("Prestadores turísticos activos (RNT)")
    if "rnt_por_categoria" in d:
        Rn = d["rnt_por_categoria"]
        fig = px.bar(Rn[Rn.municipio.isin(muns)], x="prestadores", y="categoria", color="municipio", orientation="h",
                     color_discrete_map=COLOR_MUN, height=420, labels={"prestadores": "Prestadores", "categoria": ""})
        st.plotly_chart(fig, width="stretch")
        tot = Rn.groupby("municipio")["prestadores"].sum()
        lug = L.groupby("municipio").size()
        st.dataframe(pd.DataFrame({"Prestadores RNT": tot, "Lugares en Google Maps (muestra)": lug}), width="stretch")
        st.caption("El RNT mide oferta formal registrada, no calidad. La comparación es descriptiva (3 municipios).")
pie()
