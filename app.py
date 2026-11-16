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
        for simbolo in (
            "construir_html_tarjeta_catalogo",
            "construir_html_control_tema_esquina",
            "PALETA_ARANDU",
            "hex_a_rgba",
        )
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
    construir_html_control_tema_esquina,
    obtener_logo_arandu_src,
    obtener_logos_app,
)
from views.ui_shared import mostrar_dialogo_copiar_ia

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

def _al_cambiar_tema_arandu() -> None:
    st.session_state["arandu_tema_modo"] = (
        "claro" if st.session_state.get("ui_tema_claro", False) else "oscuro"
    )


st.session_state.setdefault(
    "ui_tema_claro",
    st.session_state.get("arandu_tema_modo", "oscuro") == "claro",
)

with st.sidebar:
    if _sidebar_uri:
        st.markdown(
            f"""
            <header class="arandu-sidebar-brand">
                <img src="{_sidebar_uri}" alt="Arandú Logo" />
                <p class="arandu-sidebar-brand-name">Arandú</p>
                <p class="arandu-sidebar-brand-sub">Organización de Sabiduría Computacional<br>FaCENA — UNNE</p>
            </header>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <header class="arandu-sidebar-brand">
                <p class="arandu-sidebar-brand-name">Arandú</p>
                <p class="arandu-sidebar-brand-sub">Organización de Sabiduría Computacional • FaCENA - UNNE</p>
            </header>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
    _slot_sidebar_ia = st.container()
    with st.container(border=True):
        st.markdown('<h2 class="arandu-sidebar-heading">Feedback</h2>', unsafe_allow_html=True)
        st.caption("¿Probaste la plataforma? Contanos qué te pareció o reportá mejoras:")
        st.link_button(
            "Dar Feedback",
            FEEDBACK_FORMS_URL,
            use_container_width=True,
        )
    if es_entorno_offline_wasm():
        st.markdown(
            """
            <div class="arandu-wasm-banner" role="status">
                Motor Local Activo (100% Sin Internet)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.container(border=True):
            st.markdown('<h2 class="arandu-sidebar-heading">Modo Offline (PWA)</h2>', unsafe_allow_html=True)
            st.caption("Ideal para instalar en notebook y estudiar sin Wi-Fi")
            st.link_button(
                "Versión 100% Sin Conexión (PWA)",
                OFFLINE_PWA_URL,
                use_container_width=True,
            )
    st.markdown(
        f"""
        <footer class="arandu-footer" aria-label="Pie de página institucional">
            <strong>Arandú v{APP_VERSION}</strong> • Dark Botanical &amp; Golden Wisdom<br/>
            Proyecto Final de Carrera • FaCENA — UNNE
        </footer>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<a class="arandu-skip-link" href="#arandu-contenido-principal">Saltar al contenido principal</a>'
    '<div id="arandu-contenido-principal" tabindex="-1" class="arandu-sr-only">Contenido principal de Arandú</div>',
    unsafe_allow_html=True,
)
_, _col_tema_der = st.columns([5.6, 2.8])
with _col_tema_der:
    st.markdown(
        construir_html_control_tema_esquina(st.session_state.get("arandu_tema_modo", "oscuro")),
        unsafe_allow_html=True,
    )
    st.toggle(
        ":material/dark_mode: / :material/light_mode: Modo claro",
        key="ui_tema_claro",
        on_change=_al_cambiar_tema_arandu,
        help="Alternar entre Modo Oscuro Botánico y Modo Claro Suave.",
    )
_pg_activa = nav.run()
_titulo_pg_ia = getattr(_pg_activa, "title", None) or "Arandú"

with _slot_sidebar_ia:
    with st.container(border=True):
        st.markdown('<h2 class="arandu-sidebar-heading">Asistente IA</h2>', unsafe_allow_html=True)
        st.caption("Copiá teoría, fórmulas LaTeX, datos y resultados de esta página.")
        _clic_ia_sidebar = st.button(
            "Copiar todo en texto para IA",
            key="btn_ia_copiar_sidebar",
            type="primary",
            use_container_width=True,
        )

st.divider()
with st.container(border=True):
    _c_ia_info, _c_ia_btn = st.columns([2.4, 1.0])
    _c_ia_info.markdown(
        "**Exportar vista completa para IA:** generá en un clic el resumen estructurado "
        "(Markdown + LaTeX) con la teoría, entradas, cálculos y gráficos actuales."
    )
    with _c_ia_btn:
        _clic_ia_pie = st.button(
            "Copiar todo en texto para IA",
            key="btn_ia_copiar_pie",
            use_container_width=True,
        )

if _clic_ia_sidebar or _clic_ia_pie:
    mostrar_dialogo_copiar_ia(pagina_actual=_titulo_pg_ia)
