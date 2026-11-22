"""Resultados técnicos: EDA, K-Means, DBSCAN, PCA, Random Forest y validaciones."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from comun import cargar, COLOR_MUN, GRUPO_TXT, pie

st.set_page_config(page_title="Modelos de ML", page_icon="📊", layout="wide")
d = cargar(); L = d["lugares_modelo"]; res = d["res"]
st.title("📊 Modelos de Machine Learning")

t1, t2, t3, t4, t5 = st.tabs(["Exploratorio", "K-Means y DBSCAN", "PCA", "Random Forest", "Algoritmo genético"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(L, x="estrellas", color=L.grupo.map(GRUPO_TXT), nbins=40, barmode="overlay", opacity=.7, height=330,
                           labels={"estrellas": "Estrellas", "color": ""}, title="Distribución de estrellas")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Sesgo positivo: {res['pct_4_o_mas']:.0%} de los lugares tiene 4,0 o más; media {res['estrellas_media']:.2f}.")
    with c2:
        fig = px.histogram(L, x=np.log1p(L.n_resenas), color=L.grupo.map(GRUPO_TXT), nbins=30, barmode="overlay", opacity=.7, height=330,
                           labels={"x": "log(1 + reseñas)", "color": ""}, title="Distribución de la popularidad")
        st.plotly_chart(fig, width="stretch")
    C = d["correlacion_pearson"]
    fig = px.imshow(C, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=430, title="Correlación de Pearson")
    st.plotly_chart(fig, width="stretch")
    st.caption("Lo más fuerte: la distancia al centro baja la densidad de oferta (r = −0,64) y la popularidad sube con la densidad (r = 0,44). "
               "Las estrellas casi no se relacionan con nada: todos los lugares están bien calificados.")
with t2:
    S = d["kmeans_seleccion"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(S, x="k", y="inercia", markers=True, title="Método del codo", height=300), width="stretch")
    fig = px.line(S, x="k", y="silueta", markers=True, title="Coeficiente de silueta", height=300); fig.add_vline(x=res["kmeans"]["k"], line_dash="dot")
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(d["clustering_comparacion"], hide_index=True, width="stretch")
    st.caption("K-Means agrupa por perfil (valoración, popularidad, precio, accesibilidad); DBSCAN agrupa por cercanía física y deja el 30 % "
               "como ruido. Las siluetas no son comparables entre sí porque usan espacios distintos.")
with t3:
    v = res["pca"]["varianza"]
    c1, c2 = st.columns([3, 2])
    fig = px.scatter(L, x="pc1", y="pc2", color="segmento", hover_name="nombre", height=460,
                     labels={"pc1": f"PC1 ({v[0]:.0%})", "pc2": f"PC2 ({v[1]:.0%})"})
    fig.update_layout(legend=dict(orientation="h", y=-0.25, title=None)); c1.plotly_chart(fig, width="stretch")
    c2.markdown(f"**Varianza explicada:** PC1 {v[0]:.0%} · PC2 {v[1]:.0%} · acumulada {res['pca']['acumulada_2']:.0%}")
    c2.dataframe(d["pca_cargas"].round(2), hide_index=True, width="stretch")
    c2.caption("PC1 = eje centro–periferia (densidad y popularidad contra distancia). PC2 = valoración.")
with t4:
    m = d["rf_metricas"]; st.dataframe(m.round(3), hide_index=True, width="stretch")
    st.caption("Objetivo: log(1 + n.º de reseñas) como indicador de popularidad (no hay datos públicos de afluencia). Validación cruzada de 5 pliegues.")
    I = d["rf_importancia"].head(8).iloc[::-1]
    fig = px.bar(I, x="importancia_permutacion", y="variable", error_x="perm_sd", orientation="h", height=330, title="Importancia por permutación")
    st.plotly_chart(fig, width="stretch")
    st.warning("R² ≈ 0,26: el modelo explica una parte modesta de la popularidad. Sirve para identificar factores (estrellas, densidad de oferta, tipo), "
               "no para pronosticar visitantes.")
with t5:
    E = d["ga_escenarios"]
    st.dataframe(E[["escenario", "candidatos", "AG_media", "AG_sd", "aleatoria_media", "voraz", "mejora_vs_aleatoria_pct", "mejora_vs_voraz_pct",
                    "paradas", "minutos", "costo"]].round(3), hide_index=True, width="stretch")
    vg = res["validacion_ga"]
    st.success(f"Validación exacta (7 atractivos, 3 h): el AG encontró el óptimo de fuerza bruta en {vg['aciertos']} de {vg['semillas']} semillas.")
    Cv = d["ga_convergencia"]; esc = st.selectbox("Convergencia del escenario", Cv.escenario.unique())
    c = Cv[Cv.escenario == esc]
    fig = px.line(c, x="generacion", y=["AG", "aleatoria"], height=320, labels={"value": "Mejor fitness", "generacion": "Generación", "variable": ""})
    fig.add_hline(y=c.voraz.iloc[0], line_dash="dot", annotation_text="voraz"); st.plotly_chart(fig, width="stretch")
    Sn = d["ga_sensibilidad"]
    st.plotly_chart(px.line(Sn, x="horas", y="paradas", markers=True, height=300, title="Sensibilidad: paradas según horas disponibles (Sabana Centro)"),
                    width="stretch")
pie()
