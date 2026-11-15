"""
Componentes UX/UI compartidos para Arandú: Métodos Computacionales (v3.2).
Estructura pedagógica en 3 niveles (estilo Refactoring.Guru), aislamiento de fragmentos,
paginación acotada (> 300 filas) y renderizado Plotly ligero para producción/PWA.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import streamlit as st

from core._cache_utils import limpiar_valores_plotly, muestrear_indices_acotados
from core.catalogo_ejercicios import obtener_metodo
from views.theme import (
    ARANDU_BARK,
    ARANDU_CARD_BORDER,
    ARANDU_GOLD,
    ARANDU_GOLD_FILL,
    ARANDU_GRID,
    ARANDU_IVORY,
    ARANDU_MUTED,
    ARANDU_NODE_IVORY,
    ARANDU_OBSIDIAN,
    ARANDU_PLOT_BG,
    ARANDU_SAGE,
    ARANDU_SAGE_BORDER,
    ARANDU_SAGE_FILL,
    ARANDU_TERRACOTTA,
    ARANDU_TERRACOTTA_DARK,
    ARANDU_TERRACOTTA_FILL,
    COLORWAY_ARANDU,
    FEEDBACK_FORMS_URL,
    LOGO_JPG_PATH,
    LOGO_SVG_PATH,
    PALETA_ARANDU,
    PLOTLY_CONFIG_ARANDU,
    aplicar_estilo_arandu,
    construir_html_mini_tarjetas,
    construir_html_tarjeta_catalogo,
    estilizar_figura_arandu,
    obtener_logo_arandu_src,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


def obtener_logo_data_uri() -> str | None:
    """Retorna el logo SVG/JPEG en base64 cacheado para incrustar en HTML."""
    return obtener_logo_arandu_src(prefer_svg=True)


def inject_arandu_css() -> None:
    """Aplica la hoja CSS cacheada de Arandú sin reconstruir cadenas por interacción."""
    aplicar_estilo_arandu()


def render_bloque_1_teoria_y_ejemplo(clave_metodo: str) -> dict[str, Any]:
    """
    Renderiza la arquitectura de lectura en 3 niveles (Anti-Muro de Texto) para Sistemas:
    - Bloque A: Título + Resumen Ejecutivo (2-3 líneas).
    - Bloque B: Intuición visual/algorítmica + fórmulas st.latex() + Expander de profundización.
    - Bloque C: Veredicto Práctico (✅ Cuándo elegir / ⚠️ Cuándo evitar o tener cuidado).
    """
    meta = obtener_metodo(clave_metodo)
    if not meta:
        st.error(f"No se encontró la ficha '{clave_metodo}' en el catálogo.")
        return {}

    guru = meta.get("guru", {})

    # Bloque A: Título + Resumen Ejecutivo (siempre visible arriba)
    st.subheader(f"{meta.get('icono', '📘')} {meta['nombre']}")
    with st.container(border=True):
        st.markdown(f"**Resumen Ejecutivo:** {meta.get('resumen_corto', '')}")

    # Bloque B: Desarrollo e Intuición (escaneable y progresivo)
    with st.container(border=True):
        st.markdown("#### Cómo funciona en la práctica")
        if guru.get("que_resuelve"):
            st.markdown(f"- **Problema que resuelve:** {guru['que_resuelve']}")
        if guru.get("intuicion_geometrica"):
            st.markdown(f"- **Intuición visual / lógica:** {guru['intuicion_geometrica']}")
        for paso in guru.get("algoritmo_pasos", []):
            st.markdown(f"- {paso}")

        formulas = meta.get("formulas_latex", [])
        if formulas:
            st.markdown("**Expresión matemática del algoritmo:**")
            for formula in formulas:
                st.latex(formula)

    ejemplo = meta.get("ejemplo_teorico", {})
    profundizacion = guru.get("profundizacion", [])
    if profundizacion or ejemplo:
        with st.expander("📖 Ver desarrollo teórico detallado (Para profundizar)", expanded=False):
            for detalle in profundizacion:
                st.markdown(f"- {detalle}")
            if ejemplo:
                st.markdown(f"##### Ejemplo resuelto: {ejemplo.get('titulo', '')}")
                if ejemplo.get("enunciado"):
                    st.markdown(f"**Enunciado:** {ejemplo['enunciado']}")
                pasos = ejemplo.get("pasos", [])
                if pasos:
                    st.markdown("**Pasos:**")
                    for idx, paso in enumerate(pasos, start=1):
                        st.markdown(f"{idx}. {paso}")
                if ejemplo.get("resultado_final"):
                    st.success(f"**Conclusión:** {ejemplo['resultado_final']}")

    # Bloque C: Conclusión / Veredicto Práctico
    col_pro, col_warn = st.columns(2)
    with col_pro:
        with st.container(border=True):
            st.markdown("#### ✅ Cuándo elegir este método")
            for item in guru.get("cuando_usar", []):
                st.markdown(f"- {item}")
    with col_warn:
        with st.container(border=True):
            st.markdown("#### ⚠️ Cuándo evitarlo o tener cuidado")
            for item in guru.get("cuando_falla", []):
                st.markdown(f"- {item}")

    return meta


_PREFIJOS_WIDGETS_PRACTICA: dict[str, tuple[str, ...]] = {
    "fund_taylor": ("ui_taylor_x", "ui_taylor_n", "ui_taylor_scar", "ui_pi_pasos"),
    "fund_asoc": ("ui_asoc_",),
    "fund_canc": ("ui_canc_",),
    "fund_horner": ("ui_horner_",),
    "raiz_bis": ("ui_bis_",),
    "raiz_rf": ("ui_rf_",),
    "raiz_nr": ("ui_nr_",),
    "raiz_pf": ("ui_pf_",),
    "sis_gs": ("ui_gs_",),
    "sis_pot": ("ui_pot_tol",),
    "int_lin": ("ui_lin_yb", "ui_lin_xe"),
    "int_cuad": ("ui_cuad_yb",),
    "int_lag": ("ui_lag_xe", "ui_lag_n"),
    "int_ng": ("ui_ng_xe", "ui_ng_dh"),
    "int_deriv": ("ui_der_xe",),
    "edo_eh": ("ui_eh_corr",),
}


def inicializar_estado_practica(clave_state: str, defaults_dict: dict[str, Any]) -> None:
    """Inicializa en session_state cada parámetro de defaults_dict si aún no existe."""
    for campo, valor in defaults_dict.items():
        key = f"{clave_state}_{campo}"
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(valor)


def restaurar_estado_practica(clave_state: str, defaults_dict: dict[str, Any]) -> None:
    """Restaura parámetros de cátedra y limpia keys de widgets asociados para reflejarlos al re-renderizar."""
    for campo, valor in defaults_dict.items():
        st.session_state[f"{clave_state}_{campo}"] = copy.deepcopy(valor)
    st.session_state.pop(f"{clave_state}__editor_df", None)
    prefijos = (f"ui_{clave_state}_",) + _PREFIJOS_WIDGETS_PRACTICA.get(clave_state, ())
    for key in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefijos)]:
        st.session_state.pop(key, None)


def rerun_fragmento_seguro(scope: str = "fragment") -> None:
    """Ejecuta rerun acotado al fragmento cuando el runtime lo permite, o rerun completo como respaldo."""
    if scope != "fragment":
        st.rerun(scope=scope)
        return
    try:
        st.rerun(scope="fragment")
    except Exception:
        st.rerun()


def render_practica_catedra_info(
    titulo_practica: str | None = None,
    enunciado_practica: str | None = None,
    solucion_referencia: str | None = None,
) -> None:
    """Renderiza únicamente la zona estática del ejercicio de cátedra (fuera de fragmentos)."""
    if titulo_practica or enunciado_practica:
        st.markdown("### 📝 Práctica de Cátedra Interactiva")
        with st.container(border=True):
            if titulo_practica:
                st.markdown(f"**{titulo_practica}**")
            if enunciado_practica:
                st.markdown(enunciado_practica)
            if solucion_referencia:
                with st.expander("✅ Ver solución de referencia esperada", expanded=False):
                    st.success(solucion_referencia)


def render_controles_edicion_practica(
    clave_state: str,
    defaults_dict: dict[str, Any],
    mensaje_edicion: str = "✏️ Parámetros editables en vivo: modificá los valores o probá con un caso propio.",
    scope_rerun: str = "fragment",
) -> str:
    """Renderiza los controles interactivos (modo Cátedra / Manual + Restaurar) dentro de @st.fragment."""
    inicializar_estado_practica(clave_state, defaults_dict)
    modo_key = f"{clave_state}_modo_edicion"
    if modo_key not in st.session_state:
        st.session_state[modo_key] = "catedra"

    c_modo, c_reset = st.columns([2.2, 1.3])
    with c_modo:
        modo = st.radio(
            "Modo de datos de entrada:",
            options=["catedra", "manual"],
            format_func=lambda m: (
                "🔒 Valores oficiales de cátedra"
                if m == "catedra"
                else "✏️ Edición manual en vivo"
            ),
            horizontal=True,
            key=modo_key,
        )
    with c_reset:
        st.markdown("<div style='padding-top: 1.55rem;'></div>", unsafe_allow_html=True)
        if st.button(
            "🔄 Restaurar valores originales de la profesora",
            key=f"{clave_state}_btn_restaurar",
            use_container_width=True,
        ):
            restaurar_estado_practica(clave_state, defaults_dict)
            st.toast("Valores originales restaurados correctamente.", icon="✅")
            rerun_fragmento_seguro(scope_rerun)

    if modo == "manual":
        st.markdown(
            f'<div class="arandu-edit-banner">{mensaje_edicion}</div>',
            unsafe_allow_html=True,
        )
    return modo


def render_banner_edicion_profesora(
    clave_state: str,
    defaults_dict: dict[str, Any],
    label_modo: bool = True,
    titulo_practica: str | None = None,
    enunciado_practica: str | None = None,
    solucion_referencia: str | None = None,
    mensaje_edicion: str = "✏️ Parámetros editables en vivo: modificá los valores o probá con un caso propio.",
    scope_rerun: str = "fragment",
) -> str:
    """Wrapper compatible que combina zona informativa de práctica y controles interactivos."""
    render_practica_catedra_info(titulo_practica, enunciado_practica, solucion_referencia)
    if not label_modo:
        inicializar_estado_practica(clave_state, defaults_dict)
        return "manual"
    return render_controles_edicion_practica(
        clave_state=clave_state,
        defaults_dict=defaults_dict,
        mensaje_edicion=mensaje_edicion,
        scope_rerun=scope_rerun,
    )


def crear_figura_base_plotly(
    titulo: str,
    x_label: str = "x",
    y_label: str = "y",
    altura: int = 430,
) -> go.Figure:
    """Crea una figura Plotly uniforme sobre la paleta Dark Botanical & Golden Wisdom."""
    import plotly.graph_objects as go

    fig = go.Figure()
    estilizar_figura_arandu(fig)
    fig.update_layout(
        title=titulo,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=altura,
        hovermode="x unified",
        margin=dict(l=48, r=24, t=54, b=46),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def render_plotly_arandu(
    fig: go.Figure,
    key: str | None = None,
    **kwargs: Any,
) -> None:
    """Renderiza figuras Plotly optimizando payload de UI (sin logo Plotly y barra al pasar el cursor)."""
    config = kwargs.pop("config", PLOTLY_CONFIG_ARANDU)
    st.plotly_chart(fig, use_container_width=True, config=config, key=key, **kwargs)


def muestrear_para_grafico(
    secuencia: list[Any],
    max_puntos: int = 300,
) -> list[Any]:
    """Submuestrea listas de iteraciones > 300 manteniendo inicio, fin y densidad uniforme."""
    n = len(secuencia)
    if n <= max_puntos:
        return list(secuencia)
    indices = muestrear_indices_acotados(n, max_puntos=max_puntos)
    return [secuencia[i] for i in indices]


def render_dataframe_acotado(
    datos: Any,
    max_filas: int = 300,
    key_prefix: str = "tbl",
    hide_index: bool = True,
    **kwargs: Any,
) -> None:
    """Renderiza tablas grandes paginadas en bloques de <= 300 filas para evitar saturar el navegador."""
    import pandas as pd

    df = datos if isinstance(datos, pd.DataFrame) else pd.DataFrame(datos)
    total = len(df)
    if total <= max_filas:
        st.dataframe(df, use_container_width=True, hide_index=hide_index, **kwargs)
        return

    n_paginas = (total + max_filas - 1) // max_filas
    c_info, c_pag = st.columns([2.5, 1.2])
    with c_info:
        st.caption(
            f"La ejecución generó ${total}$ filas. Por rendimiento se muestran bloques de ${max_filas}$ filas por página."
        )
    with c_pag:
        pagina = int(
            st.number_input(
                "Página de tabla:",
                min_value=1,
                max_value=n_paginas,
                value=1,
                step=1,
                key=f"{key_prefix}_pag",
            )
        )
    ini = (pagina - 1) * max_filas
    fin = min(ini + max_filas, total)
    st.dataframe(df.iloc[ini:fin], use_container_width=True, hide_index=hide_index, **kwargs)


def render_slider_paso_k(
    label: str,
    max_val: int,
    key: str,
    min_val: int = 1,
    default_val: int | None = None,
) -> int:
    """Renderiza de forma segura el slider 'Ver hasta el paso k' incluso si hay 1 paso."""
    if max_val <= min_val:
        st.caption(f"ℹ️ {label}: **{max_val}** (único paso disponible)")
        return max_val
    val_def = max_val if default_val is None else min(max_val, max(min_val, default_val))
    return int(st.slider(label, min_value=min_val, max_value=max_val, value=val_def, key=key))


def render_tarjeta_catalogo(
    clave_svg: str,
    titulo: str,
    descripcion: str,
    ruta: str,
    texto_boton: str,
    key: str,
    subtitulo_tag: str = "",
    primary: bool = False,
) -> None:
    """Renderiza una tarjeta visual Refactoring.Guru (SVG inline + 2 líneas) con navegación."""
    with st.container():
        st.markdown(
            construir_html_tarjeta_catalogo(
                clave_svg=clave_svg,
                titulo=titulo,
                descripcion=descripcion,
                subtitulo_tag=subtitulo_tag,
            ),
            unsafe_allow_html=True,
        )
        if st.button(
            texto_boton,
            key=key,
            type="primary" if primary else "secondary",
            use_container_width=True,
        ):
            st.switch_page(ruta)


def render_mini_catalogo_vista(
    items: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> None:
    """Renderiza tarjetas compactas para distinguir rápidamente los métodos de la vista."""
    st.markdown(construir_html_mini_tarjetas(tuple(items)), unsafe_allow_html=True)


__all__ = [
    "ARANDU_BARK",
    "ARANDU_CARD_BORDER",
    "ARANDU_GOLD",
    "ARANDU_GOLD_FILL",
    "ARANDU_GRID",
    "ARANDU_IVORY",
    "ARANDU_MUTED",
    "ARANDU_NODE_IVORY",
    "ARANDU_OBSIDIAN",
    "ARANDU_PLOT_BG",
    "ARANDU_SAGE",
    "ARANDU_SAGE_BORDER",
    "ARANDU_SAGE_FILL",
    "ARANDU_TERRACOTTA",
    "ARANDU_TERRACOTTA_DARK",
    "ARANDU_TERRACOTTA_FILL",
    "COLORWAY_ARANDU",
    "FEEDBACK_FORMS_URL",
    "LOGO_JPG_PATH",
    "LOGO_SVG_PATH",
    "PALETA_ARANDU",
    "PLOTLY_CONFIG_ARANDU",
    "crear_figura_base_plotly",
    "estilizar_figura_arandu",
    "inicializar_estado_practica",
    "inject_arandu_css",
    "limpiar_valores_plotly",
    "muestrear_para_grafico",
    "obtener_logo_data_uri",
    "render_banner_edicion_profesora",
    "render_bloque_1_teoria_y_ejemplo",
    "render_controles_edicion_practica",
    "render_dataframe_acotado",
    "render_mini_catalogo_vista",
    "render_plotly_arandu",
    "render_practica_catedra_info",
    "render_slider_paso_k",
    "render_tarjeta_catalogo",
    "rerun_fragmento_seguro",
    "restaurar_estado_practica",
]
