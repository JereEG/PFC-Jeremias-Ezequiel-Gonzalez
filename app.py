"""
Entrypoint principal de Arandú: Métodos Computacionales (v3.0).
Navegación moderna con st.navigation + st.Page agrupada por metodología y Arena Versus.
100% lista para producción en Streamlit Community Cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from views.theme import (
    LOGO_JPG_PATH,
    LOGO_SVG_PATH,
    aplicar_estilo_arandu,
    obtener_logo_arandu_src,
)
from views.ui_shared import FEEDBACK_FORMS_URL

_SIDEBAR_LOGO = LOGO_JPG_PATH if LOGO_JPG_PATH.is_file() else LOGO_SVG_PATH
_PAGE_ICON = str(LOGO_SVG_PATH) if LOGO_SVG_PATH.is_file() else (str(LOGO_JPG_PATH) if LOGO_JPG_PATH.is_file() else "🌳")

st.set_page_config(
    page_title="Arandú — Métodos Computacionales v3.0",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

if _SIDEBAR_LOGO.is_file():
    _ICON_IMG = LOGO_SVG_PATH if LOGO_SVG_PATH.is_file() else _SIDEBAR_LOGO
    st.logo(str(_SIDEBAR_LOGO), size="large", icon_image=str(_ICON_IMG))

aplicar_estilo_arandu()

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

_sidebar_uri = obtener_logo_arandu_src(prefer_svg=False)

with st.sidebar:
    if _sidebar_uri:
        st.markdown(
            f"""
            <div class="arandu-sidebar-brand">
                <img src="{_sidebar_uri}" alt="Arandú Logo" />
                <p class="arandu-sidebar-brand-name">ARANDÚ</p>
                <p class="arandu-sidebar-brand-sub">Organización de Sabiduría Computacional<br>FaCENA — UNNE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="arandu-sidebar-brand">
                <p class="arandu-sidebar-brand-name">ARANDÚ</p>
                <p class="arandu-sidebar-brand-sub">Organización de Sabiduría Computacional • FaCENA - UNNE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### 💬 Feedback")
        st.caption("¿Probaste la plataforma? Contanos qué te pareció o reportá mejoras:")
        st.link_button(
            "📝 Dar Feedback",
            FEEDBACK_FORMS_URL,
            width="stretch",
        )
    st.caption("**Arandú v3.0** • Dark Botanical & Golden Wisdom")
    st.caption("Proyecto Final de Carrera • FaCENA — UNNE")

nav.run()
