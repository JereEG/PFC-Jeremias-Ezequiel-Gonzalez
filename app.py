"""
Entrypoint principal de Arandú: Métodos Computacionales (v3.0).
Navegación moderna con st.navigation + st.Page agrupada por metodología y Arena Versus.
100% lista para producción en Streamlit Community Cloud.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Arandú — Métodos Computacionales v3.0",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg_inicio = st.Page(
    "views/inicio.py",
    title="Inicio",
    icon="🏠",
    default=True,
    url_path="inicio",
)
pg_versus = st.Page(
    "views/arena_versus.py",
    title="Arena Versus",
    icon="⚔️",
    url_path="arena_versus",
)
pg_fundamentos = st.Page(
    "views/view_fundamentos.py",
    title="Fundamentos y Errores",
    icon="🧮",
    url_path="fundamentos",
)
pg_raices = st.Page(
    "views/view_raices.py",
    title="Raíces de Ecuaciones",
    icon="📉",
    url_path="raices",
)
pg_sistemas = st.Page(
    "views/view_sistemas.py",
    title="Sistemas y Autovalores",
    icon="📐",
    url_path="sistemas",
)
pg_interpolacion = st.Page(
    "views/view_interpolacion.py",
    title="Interpolación Polinomial",
    icon="📈",
    url_path="interpolacion",
)
pg_integracion = st.Page(
    "views/view_integracion.py",
    title="Integración y Derivación",
    icon="📊",
    url_path="integracion",
)
pg_edo = st.Page(
    "views/view_edo.py",
    title="Ecuaciones Diferenciales",
    icon="⏱️",
    url_path="edo",
)
pg_labs = st.Page(
    "views/view_laboratorios.py",
    title="Histórico de Cursada",
    icon="📚",
    url_path="laboratorios",
)

nav = st.navigation(
    {
        "Principal": [pg_inicio, pg_versus],
        "Métodos Numéricos (Estilo Guru)": [
            pg_fundamentos,
            pg_raices,
            pg_sistemas,
            pg_interpolacion,
            pg_integracion,
            pg_edo,
        ],
        "Anexo Curricular": [pg_labs],
    }
)

with st.sidebar:
    st.markdown("---")
    st.caption("**Arandú v3.0** • FaCENA — UNNE")
    st.caption("Proyecto Final de Carrera • Suite Interactiva Cloud-Ready")

nav.run()
