"""Datos, supuestos y limitaciones."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from comun import cargar, pie

st.set_page_config(page_title="Datos y supuestos", page_icon="📄", layout="wide")
d = cargar(); L = d["lugares_modelo"]; res = d["res"]; sup = res["supuestos"]
st.title("📄 Datos, supuestos y limitaciones")
st.warning("**Lee esto antes de interpretar.** Muestra no aleatoria de los resultados que muestra Google Maps (hasta ≈ 120 por búsqueda); "
           "3 municipios; las entradas a atractivos y el costo por km son **supuestos**; la popularidad (n.º de reseñas) no es afluencia real.")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Preprocesamiento")
    st.dataframe(d["embudo_preprocesamiento"], hide_index=True, width="stretch")
    st.json({"imputados (%)": {k: round(v * 100, 1) for k, v in res["imputados"].items()}, "atípicos IQR": res["atipicos_iqr"], "matriz de tiempos": res["matriz"]})
with c2:
    st.subheader("Supuestos del algoritmo genético")
    st.write(f"Costo de transporte: **${sup['COSTO_KM']:,} por km**".replace(",", "."))
    st.dataframe(pd.DataFrame(sup["ENTRADAS"].items(), columns=["Atractivo (palabra clave)", "Entrada aprox. (COP)"]), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(sup["VISITA"], columns=["Palabra clave", "Visita (min)"]), hide_index=True, width="stretch")
st.subheader("Lugares procesados")
st.dataframe(L[["municipio", "nombre", "grupo", "categoria_maps", "estrellas", "n_resenas", "nivel_precio", "dist_centro_km", "segmento"]],
             hide_index=True, width="stretch", height=380)
st.download_button("Descargar lugares_modelo.csv", L.to_csv(index=False).encode("utf-8"), "lugares_modelo.csv", "text/csv")
st.subheader("Limitaciones")
st.markdown("""
1. **Cobertura:** Google Maps limita los resultados por búsqueda; la muestra (231 lugares) favorece lo más visible.
2. **Alcance reducido** de 6 a 3 municipios para poder recolectar en el tiempo disponible; las dos provincias siguen representadas.
3. **Sin afluencia real:** el Random Forest predice popularidad digital (reseñas), con R² ≈ 0,26.
4. **Precios:** Google publica precio en la mitad de restaurantes y hoteles; los atractivos usan una tabla de supuestos.
5. **Estrellas con techo:** 88 % de los lugares tiene ≥ 4,0, lo que limita su poder discriminante.
6. **Densidad de oferta** calculada con la propia muestra (OpenStreetMap no respondió durante la recolección).
7. **Instancias pequeñas** para el AG (6–18 candidatos): la mejora frente a búsqueda aleatoria es pequeña; se validó contra el óptimo exacto.
""")
pie()
