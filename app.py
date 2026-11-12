"""
app.py
======
El Santo Grial de los Métodos Numéricos — Entrypoint Principal.
Proyecto Final de Carrera (PFC) — FaCENA - Universidad Nacional del Nordeste (UNNE).

Desarrollado con arquitectura Cloud-Ready para Streamlit Community Cloud utilizando
navegación semántica por ramas matemáticas (st.navigation y st.Page por archivos).
"""

from __future__ import annotations

import streamlit as st

# Configuración global de la página
st.set_page_config(
    page_title="El Santo Grial de los Métodos Numéricos — FaCENA UNNE",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definición de páginas organizadas por Categorías Matemáticas
pg_inicio = st.Page(
    "views/inicio.py",
    title="Inicio (Catálogo General)",
    icon="🏠",
    default=True,
    url_path="inicio",
)

pg_fundamentos = st.Page(
    "views/fundamentos.py",
    title="Fundamentos y Errores",
    icon="🧮",
    url_path="fundamentos",
)

pg_raices = st.Page(
    "views/raices.py",
    title="Raíces de Ecuaciones",
    icon="📉",
    url_path="raices",
)

pg_algebra = st.Page(
    "views/algebra_lineal.py",
    title="Álgebra Lineal Numérica",
    icon="📐",
    url_path="algebra_lineal",
)

pg_interp = st.Page(
    "views/interpolacion.py",
    title="Interpolación Polinomial",
    icon="📈",
    url_path="interpolacion",
)

pg_calculo = st.Page(
    "views/calculo_numerico.py",
    title="Integración y Derivación",
    icon="∬",
    url_path="calculo_numerico",
)

pg_edo = st.Page(
    "views/edo.py",
    title="Ecuaciones Diferenciales (EDO)",
    icon="⏱️",
    url_path="edo",
)

pg_historico = st.Page(
    "views/historico_labs.py",
    title="Histórico de Cursada",
    icon="📚",
    url_path="historico",
)

# Navegación agrupada por categorías temáticas
nav = st.navigation({
    "Portal Principal": [pg_inicio],
    "Métodos Matemáticos": [
        pg_fundamentos,
        pg_raices,
        pg_algebra,
        pg_interp,
        pg_calculo,
        pg_edo,
    ],
    "Anexo Curricular": [
        pg_historico,
    ],
})

# Panel lateral con identidad de marca y contexto académico
with st.sidebar:
    st.markdown("### 🏛️ El Santo Grial")
    st.caption("Plataforma Integral de Métodos Numéricos")
    st.markdown("---")
    st.markdown(
        """
        **Cátedra:** Métodos Computacionales  
        **Institución:** FaCENA - UNNE  
        **Autor:** Jeremías Ezequiel González  
        **Versión:** 2.0.0-pivote-ux
        """
    )
    st.markdown("---")
    st.info("💡 Cada método numérico cuenta con teoría formal, material de cátedra en PDF y workspace interactivo.")

# Ejecutar el ruteo de la página seleccionada
nav.run()
