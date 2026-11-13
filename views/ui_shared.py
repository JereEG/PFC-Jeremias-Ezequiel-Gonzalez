"""
Componentes UX/UI compartidos para Arandú: Métodos Computacionales (v3.0).
Estética Refactoring.Guru (claridad visual) + Versus.com (comparación técnica).
"""

from __future__ import annotations

import base64
import copy
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from core.catalogo_ejercicios import obtener_metodo

FEEDBACK_FORMS_URL = "https://forms.gle/Sy6M1NWzHRxMbuiBA"
LOGO_SVG_PATH = Path(__file__).resolve().parent.parent / "assets" / "svg" / "arandu.svg"


@st.cache_data(show_spinner=False)
def obtener_logo_data_uri() -> str | None:
    """Retorna el logo SVG codificado en base64 para incrustar de forma segura en HTML."""
    if not LOGO_SVG_PATH.is_file():
        return None
    svg_bytes = LOGO_SVG_PATH.read_bytes()
    encoded = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_arandu_css() -> None:
    """Inyecta estilos CSS limpios y adaptativos (light/dark) para toda la suite Arandú."""
    st.markdown(
        """
        <style>
        .arandu-hero {
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            border: 1px solid rgba(59, 130, 246, 0.35);
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(16, 185, 129, 0.10));
            margin-bottom: 1.2rem;
        }
        .arandu-hero-grid {
            display: flex;
            align-items: center;
            gap: 1.35rem;
            flex-wrap: wrap;
        }
        .arandu-hero-logo {
            width: 96px;
            height: 98px;
            flex-shrink: 0;
            background: transparent;
            border: none;
            filter: drop-shadow(0 4px 10px rgba(15, 23, 42, 0.18));
            object-fit: contain;
        }
        .arandu-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.0rem;
        }
        .arandu-guru-card {
            padding: 0.95rem 1.1rem;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.30);
            background-color: rgba(148, 163, 184, 0.06);
            margin-bottom: 0.75rem;
            height: 100%;
        }
        .arandu-guru-card h4 {
            margin: 0 0 0.45rem 0;
            font-size: 1.0rem;
        }
        .arandu-ejemplo-box {
            padding: 1rem 1.2rem;
            border-radius: 10px;
            border-left: 5px solid #10b981;
            background-color: rgba(16, 185, 129, 0.08);
            margin: 0.6rem 0 1rem 0;
        }
        .arandu-edit-banner {
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: 2px dashed #f59e0b;
            background-color: rgba(245, 158, 11, 0.12);
            font-weight: 600;
            margin: 0.6rem 0 0.75rem 0;
        }
        .arandu-versus-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.40);
            background-color: rgba(99, 102, 241, 0.07);
            margin-bottom: 0.85rem;
        }
        .arandu-badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background-color: rgba(59, 130, 246, 0.18);
            border: 1px solid rgba(59, 130, 246, 0.40);
            margin-right: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bloque_1_teoria_y_ejemplo(clave_metodo: str) -> dict[str, Any]:
    """
    Bloque 1 (Estilo Refactoring.Guru):
    - Qué resuelve / Intuición geométrica / Cuándo usar / Cuándo falla
    - Fórmulas oficiales en LaTeX
    - Ejemplo Teórico Oficial resuelto paso a paso
    Retorna la entrada del catálogo para reutilizar sus datos de cátedra en Bloque 2.
    """
    meta = obtener_metodo(clave_metodo)
    if not meta:
        st.error(f"No se encontró la ficha '{clave_metodo}' en el catálogo.")
        return {}

    st.subheader(f"{meta.get('icono', '📘')} {meta['nombre']}")
    st.caption(meta.get("resumen_corto", ""))

    guru = meta.get("guru", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='arandu-guru-card'><h4>🎯 ¿Qué resuelve?</h4><p>{guru.get('que_resuelve', '')}</p></div>",
            unsafe_allow_html=True,
        )
        usar_html = "".join(f"<li>{item}</li>" for item in guru.get("cuando_usar", []))
        st.markdown(
            f"<div class='arandu-guru-card'><h4>✅ ¿Cuándo usar?</h4><ul>{usar_html}</ul></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='arandu-guru-card'><h4>📐 Intuición geométrica</h4><p>{guru.get('intuicion_geometrica', '')}</p></div>",
            unsafe_allow_html=True,
        )
        falla_html = "".join(f"<li>{item}</li>" for item in guru.get("cuando_falla", []))
        st.markdown(
            f"<div class='arandu-guru-card'><h4>⚠️ ¿Cuándo falla?</h4><ul>{falla_html}</ul></div>",
            unsafe_allow_html=True,
        )

    formulas = meta.get("formulas_latex", [])
    if formulas:
        with st.container(border=True):
            st.markdown("**Fórmulas Oficiales de Cátedra:**")
            for formula in formulas:
                st.latex(formula)

    ejemplo = meta.get("ejemplo_teorico", {})
    if ejemplo:
        with st.expander(f"📖 Ejemplo Teórico Oficial: {ejemplo.get('titulo', '')}", expanded=False):
            st.markdown(f"**Enunciado:** {ejemplo.get('enunciado', '')}")
            pasos = ejemplo.get("pasos", [])
            if pasos:
                st.markdown("**Desarrollo paso a paso:**")
                for idx, paso in enumerate(pasos, start=1):
                    st.markdown(f"{idx}. {paso}")
            if ejemplo.get("resultado_final"):
                st.success(f"**Conclusión:** {ejemplo['resultado_final']}")

    return meta


def inicializar_estado_practica(clave_state: str, defaults_dict: dict[str, Any]) -> None:
    """Inicializa en session_state cada parámetro de defaults_dict si aún no existe."""
    for campo, valor in defaults_dict.items():
        key = f"{clave_state}_{campo}"
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(valor)


def restaurar_estado_practica(clave_state: str, defaults_dict: dict[str, Any]) -> None:
    """Restaura los parámetros de la práctica a los valores originales de la profesora."""
    for campo, valor in defaults_dict.items():
        st.session_state[f"{clave_state}_{campo}"] = copy.deepcopy(valor)


def render_banner_edicion_profesora(
    clave_state: str,
    defaults_dict: dict[str, Any],
    label_modo: bool = True,
    titulo_practica: str | None = None,
    enunciado_practica: str | None = None,
    solucion_referencia: str | None = None,
    mensaje_edicion: str = "✏️ Tabla/Matriz Interactiva: Haz doble clic en cualquier celda para modificar los valores en vivo",
) -> str:
    """
    Bloque 2 (Práctica de Cátedra Interactiva):
    Renderiza el enunciado oficial, banner destacado de edición en vivo, toggle
    Ejercicio de Cátedra vs Edición Manual, y botón de restauración inmediata.
    Retorna el modo seleccionado ("catedra" o "manual").
    """
    inicializar_estado_practica(clave_state, defaults_dict)

    st.markdown("### 🧪 Bloque 2 — Práctica de Cátedra Interactiva")
    if titulo_practica or enunciado_practica:
        with st.container(border=True):
            if titulo_practica:
                st.markdown(f"**{titulo_practica}**")
            if enunciado_practica:
                st.write(enunciado_practica)
            if solucion_referencia:
                with st.expander("📌 Ver respuesta de referencia de la profesora"):
                    st.info(solucion_referencia)

    col_modo, col_btn = st.columns([3, 1.5])
    modo = "manual"
    if label_modo:
        with col_modo:
            opcion = st.radio(
                "Modo de trabajo:",
                options=["📘 Ejercicio de Cátedra", "✍️ Edición Manual"],
                horizontal=True,
                key=f"{clave_state}__modo_ui",
            )
            modo = "catedra" if opcion.startswith("📘") else "manual"
    with col_btn:
        if st.button(
            "🔄 Restaurar valores originales de la profesora",
            key=f"{clave_state}__btn_reset",
            use_container_width=True,
        ):
            restaurar_estado_practica(clave_state, defaults_dict)
            if f"{clave_state}__editor_df" in st.session_state:
                del st.session_state[f"{clave_state}__editor_df"]
            st.rerun()

    st.markdown(f"<div class='arandu-edit-banner'>{mensaje_edicion}</div>", unsafe_allow_html=True)
    return modo


def crear_figura_base_plotly(
    titulo: str,
    x_label: str = "x",
    y_label: str = "y",
    altura: int = 430,
) -> go.Figure:
    """Crea una figura Plotly limpia, responsive y lista para visualización geométrica."""
    fig = go.Figure()
    fig.update_layout(
        title=titulo,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=altura,
        hovermode="x unified",
        margin=dict(l=40, r=25, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(showgrid=True, zeroline=True, zerolinewidth=1.2)
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinewidth=1.2)
    return fig


def render_slider_paso_k(
    label: str,
    max_val: int,
    key: str,
    min_val: int = 1,
    default_val: int | None = None,
) -> int:
    """Renderiza st.slider evitando StreamlitInvalidMinMaxError cuando max_val <= min_val."""
    if max_val <= min_val:
        st.caption(f"{label} **{max_val}** (paso único)")
        return int(max_val)
    val_def = max_val if default_val is None else min(max_val, max(min_val, default_val))
    return int(st.slider(label, min_value=min_val, max_value=max_val, value=val_def, key=key))
