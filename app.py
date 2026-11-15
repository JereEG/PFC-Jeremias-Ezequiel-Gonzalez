"""
Entrypoint principal de Arandú: Métodos Computacionales (v3.0).
Navegación moderna con st.navigation + st.Page agrupada por metodología y Arena Versus.
100% lista para producción en Streamlit Community Cloud.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st


def _sincronizar_modulos_locales() -> None:
    """Recarga módulos locales si el código en disco cambió tras un git pull en caliente."""
    firmas_actuales: dict[str, tuple[int, int]] = {}
    for carpeta in ("core", "views"):
        ruta_carpeta = ROOT_DIR / carpeta
        if ruta_carpeta.is_dir():
            for archivo in ruta_carpeta.glob("*.py"):
                stat = archivo.stat()
                firmas_actuales[f"{carpeta}/{archivo.name}"] = (stat.st_mtime_ns, stat.st_size)

    modulo_tema = sys.modules.get("views.theme")
    falta_simbolo = modulo_tema is not None and any(
        not hasattr(modulo_tema, simbolo)
        for simbolo in ("construir_html_tarjeta_catalogo", "PALETA_ARANDU", "hex_a_rgba")
    )
    if getattr(sys, "_arandu_firmas_modulos", None) != firmas_actuales or falta_simbolo:
        importlib.invalidate_caches()
        for nombre_mod in list(sys.modules):
            if nombre_mod in ("core", "views") or nombre_mod.startswith(("core.", "views.")):
                del sys.modules[nombre_mod]
        try:
            st.cache_data.clear()
        except Exception:
            pass
        setattr(sys, "_arandu_firmas_modulos", firmas_actuales)


_sincronizar_modulos_locales()

from core.version import APP_VERSION, OFFLINE_PWA_URL, es_entorno_offline_wasm
from views.theme import (
    FEEDBACK_FORMS_URL,
    aplicar_estilo_arandu,
    obtener_logo_arandu_src,
    obtener_logos_app,
)

_SIDEBAR_LOGO, _PAGE_ICON = obtener_logos_app()

st.set_page_config(
    page_title=f"Arandú — Métodos Computacionales v{APP_VERSION}",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

if _SIDEBAR_LOGO is not None:
    st.logo(
    _SIDEBAR_LOGO,
    size="large",
    icon_image=_SIDEBAR_LOGO if (isinstance(_PAGE_ICON, str) and _PAGE_ICON.startswith(":material/")) else _PAGE_ICON,
)

aplicar_estilo_arandu()

pg_inicio = st.Page(
    "views/inicio.py",
    title="Inicio",
    icon=":material/home:",
    default=True,
    url_path="inicio",
)
pg_versus = st.Page(
    "views/arena_versus.py",
    title="Arena Versus",
    icon=":material/compare_arrows:",
    url_path="arena_versus",
)
pg_fundamentos = st.Page(
    "views/view_fundamentos.py",
    title="Fundamentos y Errores",
    icon=":material/functions:",
    url_path="fundamentos",
)
pg_raices = st.Page(
    "views/view_raices.py",
    title="Raíces de Ecuaciones",
    icon=":material/show_chart:",
    url_path="raices",
)
pg_sistemas = st.Page(
    "views/view_sistemas.py",
    title="Sistemas y Autovalores",
    icon=":material/grid_on:",
    url_path="sistemas",
)
pg_interpolacion = st.Page(
    "views/view_interpolacion.py",
    title="Interpolación Polinomial",
    icon=":material/timeline:",
    url_path="interpolacion",
)
pg_integracion = st.Page(
    "views/view_integracion.py",
    title="Integración y Derivación",
    icon=":material/area_chart:",
    url_path="integracion",
)
pg_edo = st.Page(
    "views/view_edo.py",
    title="Ecuaciones Diferenciales",
    icon=":material/ssid_chart:",
    url_path="edo",
)
pg_labs = st.Page(
    "views/view_laboratorios.py",
    title="Histórico de Cursada",
    icon=":material/science:",
    url_path="laboratorios",
)

nav = st.navigation(
    {
        "Principal": [pg_inicio, pg_versus],
        "Métodos Numéricos": [
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
                <p class="arandu-sidebar-brand-name">Arandú</p>
                <p class="arandu-sidebar-brand-sub">Organización de Sabiduría Computacional<br>FaCENA — UNNE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="arandu-sidebar-brand">
                <p class="arandu-sidebar-brand-name">Arandú</p>
                <p class="arandu-sidebar-brand-sub">Organización de Sabiduría Computacional • FaCENA - UNNE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Feedback")
        st.caption("¿Probaste la plataforma? Contanos qué te pareció o reportá mejoras:")
        st.link_button(
            "Dar Feedback",
            FEEDBACK_FORMS_URL,
            use_container_width=True,
        )
    if es_entorno_offline_wasm():
        st.markdown(
            """
            <div class="arandu-wasm-banner">
                Motor Local Activo (100% Sin Internet)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.container(border=True):
            st.markdown("#### Modo Offline (PWA)")
            st.caption("Ideal para instalar en notebook y estudiar sin Wi-Fi")
            st.link_button(
                "Versión 100% Sin Conexión (PWA)",
                OFFLINE_PWA_URL,
                use_container_width=True,
            )
    st.caption(f"**Arandú v{APP_VERSION}** • Dark Botanical & Golden Wisdom")
    st.caption("Proyecto Final de Carrera • FaCENA — UNNE")

nav.run()
