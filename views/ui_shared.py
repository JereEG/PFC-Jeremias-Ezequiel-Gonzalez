"""
Componentes UX/UI compartidos para Arandú: Métodos Computacionales (v3.2).
Estructura pedagógica en 3 niveles (estilo Refactoring.Guru), aislamiento de fragmentos,
paginación acotada (> 300 filas) y renderizado Plotly ligero para producción/PWA.
"""

from __future__ import annotations

import copy
import csv
import html
import io
import math
import re
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
    construir_html_ficha_metodo,
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


_IA_CTX_KEY = "_arandu_ia_ctx"


def _obtener_ctx_ia() -> dict[str, Any]:
    """Obtiene o inicializa el registro del contexto actual para exportar en texto hacia IA."""
    ctx = st.session_state.get(_IA_CTX_KEY)
    if not isinstance(ctx, dict):
        ctx = {
            "pagina": "",
            "metodo": {},
            "teoria": None,
            "practica": {},
            "controles": {},
            "editores": {},
            "tablas": {},
            "resultados": {},
            "graficos": {},
            "errores": {},
        }
        st.session_state[_IA_CTX_KEY] = ctx
    return ctx


def reiniciar_contexto_ia(pagina: str | None = None) -> None:
    """Reinicia el colector estructurado al comenzar un render completo de página."""
    st.session_state[_IA_CTX_KEY] = {
        "pagina": pagina or "",
        "metodo": {},
        "teoria": None,
        "practica": {},
        "controles": {},
        "editores": {},
        "tablas": {},
        "resultados": {},
        "graficos": {},
        "errores": {},
    }


_TITULOS_VISTA_IA: dict[str, str] = {
    "fundamentos": "Fundamentos y Errores",
    "raices": "Raíces de Ecuaciones",
    "sistemas": "Álgebra Lineal y Autovalores",
    "interpolacion": "Interpolación Polinomial",
    "integracion": "Integración y Derivación",
    "edo": "Ecuaciones Diferenciales",
}


def registrar_pagina_ia(titulo: str) -> None:
    """Fija el título descriptivo de la página actual para exportación hacia IA."""
    ctx = _obtener_ctx_ia()
    ctx["pagina"] = str(titulo).strip()


def registrar_contexto_ia(
    titulo: str,
    contenido_md: str,
    clave: str | None = None,
    categoria: str = "resultados",
) -> None:
    """Registra un bloque Markdown/LaTeX relevante dentro de la vista activa para exportación IA."""
    ctx = _obtener_ctx_ia()
    bucket = ctx.setdefault(categoria, {})
    if not isinstance(bucket, dict):
        bucket = {}
        ctx[categoria] = bucket
    bucket[clave or titulo] = (titulo, str(contenido_md).strip())


_RE_SR_ONLY = re.compile(r"<span[^>]*class=['\"][^'\"]*arandu-sr-only[^'\"]*['\"][^>]*>.*?</span>", re.DOTALL)
_RE_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_RE_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL | re.IGNORECASE)


def html_tabla_a_markdown(html_str: str) -> str:
    """Convierte una tabla HTML accesible en una tabla Markdown legible por modelos de lenguaje."""
    limpio_sr = _RE_SR_ONLY.sub("", str(html_str))
    filas: list[list[str]] = []
    for match_tr in _RE_TR.finditer(limpio_sr):
        celdas = []
        for match_td in _RE_CELL.finditer(match_tr.group(1)):
            txt = html.unescape(_HTML_TAG_RE.sub(" ", match_td.group(1)))
            txt = re.sub(r"\s+", " ", txt).strip().replace("|", "\\|")
            celdas.append(txt)
        if celdas:
            filas.append(celdas)
    if not filas:
        txt = html.unescape(_HTML_TAG_RE.sub(" ", limpio_sr))
        return re.sub(r"\s+", " ", txt).strip()

    ancho = max(len(f) for f in filas)
    norm = [f + [""] * (ancho - len(f)) for f in filas]
    lineas = [
        "| " + " | ".join(norm[0]) + " |",
        "| " + " | ".join("---" for _ in norm[0]) + " |",
    ]
    for fila in norm[1:]:
        lineas.append("| " + " | ".join(fila) + " |")
    return "\n".join(lineas)


def _formatear_valor_ia(valor: Any) -> str:
    if isinstance(valor, float):
        return f"{valor:.8g}"
    if isinstance(valor, list):
        if len(valor) > 20:
            items = [_formatear_valor_ia(v) for v in valor[:12]]
            return f"[{', '.join(items)}, ... ({len(valor)} elementos)]"
        return "[" + ", ".join(_formatear_valor_ia(v) for v in valor) + "]"
    return str(valor).replace("\n", " ").strip()


def _df_a_markdown(datos: Any, max_filas: int = 80) -> str:
    import pandas as pd

    df = datos if isinstance(datos, pd.DataFrame) else pd.DataFrame(datos)
    if df.empty:
        return "_(Tabla vacía)_"
    cols = [str(c).replace("|", "\\|") for c in df.columns]
    total = len(df)
    omitidas = 0
    if total > max_filas:
        n_head = max(max_filas // 2, 10)
        n_tail = max(max_filas - n_head, 10)
        df_head = df.iloc[:n_head]
        df_tail = df.iloc[-n_tail:]
        omitidas = total - n_head - n_tail
        bloques = [df_head, df_tail]
    else:
        bloques = [df]

    lineas = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for idx_b, sub_df in enumerate(bloques):
        if idx_b == 1 and omitidas > 0:
            lineas.append(f"| ... _({omitidas} filas intermedias omitidas de {total} totales)_ | " + " | ".join("" for _ in cols[1:]) + " |")
        for _, row in sub_df.iterrows():
            vals = [_formatear_valor_ia(v).replace("|", "\\|") for v in row.tolist()]
            lineas.append("| " + " | ".join(vals) + " |")
    return "\n".join(lineas)


def inject_arandu_css() -> None:
    """Aplica la hoja CSS cacheada de Arandú y reinicia el colector estructurado para IA."""
    aplicar_estilo_arandu()
    reiniciar_contexto_ia()


def render_bloque_1_teoria_y_ejemplo(clave_metodo: str) -> dict[str, Any]:
    """
    Renderiza la arquitectura de lectura en 3 niveles (Anti-Muro de Texto) para Sistemas:
    - Bloque A: Título + Resumen Ejecutivo (2-3 líneas).
    - Bloque B: Intuición visual/algorítmica + fórmulas st.latex() + Expander de profundización.
    - Bloque C: Veredicto Práctico (Cuándo elegirlo / Cuándo evitarlo o tener cuidado).
    """
    meta = obtener_metodo(clave_metodo)
    if not meta:
        st.error(f"No se encontró la ficha '{clave_metodo}' en el catálogo.")
        return {}

    ctx = _obtener_ctx_ia()
    ctx["teoria"] = copy.deepcopy(meta)

    guru = meta.get("guru", {})

    # Bloque A: Título + Resumen Ejecutivo (siempre visible arriba)
    st.subheader(meta["nombre"])
    with st.container(border=True):
        st.markdown(f"**Resumen Ejecutivo:** {meta.get('resumen_corto', '')}")

    # Bloque B: Desarrollo e Intuición (escaneable y progresivo)
    with st.container(border=True):
        st.markdown("### Cómo funciona en la práctica")
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
        with st.expander("Ver desarrollo teórico detallado (Para profundizar)", expanded=False):
            for detalle in profundizacion:
                st.markdown(f"- {detalle}")
            if ejemplo:
                st.markdown(f"#### Ejemplo resuelto: {ejemplo.get('titulo', '')}")
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
            st.markdown(
                '<div class="arandu-verdict-head arandu-verdict-head-pro">Cuándo elegirlo</div>',
                unsafe_allow_html=True,
            )
            for item in guru.get("cuando_usar", []):
                st.markdown(f"- {item}")
    with col_warn:
        with st.container(border=True):
            st.markdown(
                '<div class="arandu-verdict-head arandu-verdict-head-warn">Cuándo evitarlo o tener cuidado</div>',
                unsafe_allow_html=True,
            )
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
    "int_ng": ("ui_ng_", "vis_ng_"),
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
        ctx = _obtener_ctx_ia()
        clave_p = titulo_practica or "practica"
        ctx["practica"][clave_p] = {
            "titulo": titulo_practica or "",
            "enunciado": enunciado_practica or "",
            "solucion_referencia": solucion_referencia or "",
        }
        st.markdown("### Práctica de Cátedra Interactiva")
        with st.container(border=True):
            if titulo_practica:
                st.markdown(f"**{titulo_practica}**")
            if enunciado_practica:
                st.markdown(enunciado_practica)
            if solucion_referencia:
                with st.expander("Ver solución de referencia esperada", expanded=False):
                    st.success(solucion_referencia)


def render_subtitulo_controles_vivo(
    texto: str = "Ajustá los controles para recalcular el gráfico en tiempo real",
) -> None:
    """Muestra un subtítulo claro sobre paneles de parámetros reactivos en vivo."""
    st.markdown(f'<p class="arandu-ctrl-hint">{html.escape(texto)}</p>', unsafe_allow_html=True)


def render_controles_edicion_practica(
    clave_state: str,
    defaults_dict: dict[str, Any],
    mensaje_edicion: str = "Parámetros editables en vivo: modificá los valores o probá con un caso propio.",
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
                "Valores oficiales de cátedra"
                if m == "catedra"
                else "Edición manual en vivo"
            ),
            horizontal=True,
            key=modo_key,
        )
    with c_reset:
        st.markdown("<div style='padding-top: 1.55rem;'></div>", unsafe_allow_html=True)
        if st.button(
            "Restaurar valores originales de la profesora",
            key=f"{clave_state}_btn_restaurar",
            use_container_width=True,
        ):
            restaurar_estado_practica(clave_state, defaults_dict)
            st.toast("Valores originales restaurados correctamente.")
            rerun_fragmento_seguro(scope_rerun)

    if modo == "manual":
        st.info(mensaje_edicion)
    render_subtitulo_controles_vivo()

    params_vivos: dict[str, str] = {}
    prefijo_state = f"{clave_state}_"
    prefijos_ui = (f"ui_{clave_state}_",) + _PREFIJOS_WIDGETS_PRACTICA.get(clave_state, ())
    for k_ss, val_ss in st.session_state.items():
        if not isinstance(k_ss, str) or k_ss.endswith("__editor_df") or k_ss == modo_key:
            continue
        if k_ss.startswith(prefijo_state) or k_ss.startswith(prefijos_ui):
            if hasattr(val_ss, "to_dict"):
                continue
            params_vivos[k_ss] = _formatear_valor_ia(val_ss)
    ctx = _obtener_ctx_ia()
    ctx["controles"][clave_state] = {
        "modo": "Edición manual en vivo" if modo == "manual" else "Valores oficiales de cátedra",
        "mensaje": mensaje_edicion,
        "parametros": params_vivos,
    }
    return modo


def render_banner_edicion_profesora(
    clave_state: str,
    defaults_dict: dict[str, Any],
    label_modo: bool = True,
    titulo_practica: str | None = None,
    enunciado_practica: str | None = None,
    solucion_referencia: str | None = None,
    mensaje_edicion: str = "Parámetros editables en vivo: modificá los valores o probá con un caso propio.",
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
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _limpiar_texto_plotly(valor: Any, defecto: str) -> str:
    if not valor:
        return defecto
    limpio = _HTML_TAG_RE.sub(" ", str(valor))
    limpio = re.sub(r"\s+", " ", limpio).strip()
    return limpio or defecto


def _extraer_accesibilidad_plotly(
    fig: go.Figure,
    resumen_accesible: str | None = None,
) -> tuple[str, str, str, str, list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Extrae título, ejes, resumen textual, filas HTML muestreadas y filas CSV de una figura Plotly."""
    titulo = _limpiar_texto_plotly(getattr(getattr(fig.layout, "title", None), "text", None), "Gráfico numérico")
    ejex = _limpiar_texto_plotly(getattr(getattr(fig.layout.xaxis, "title", None), "text", None), "x")
    ejey = _limpiar_texto_plotly(getattr(getattr(fig.layout.yaxis, "title", None), "text", None), "y")

    resumen_series: list[str] = []
    filas_html: list[tuple[str, str, str]] = []
    filas_csv: list[tuple[str, str, str]] = []

    trazas = list(getattr(fig, "data", ()) or ())
    for idx_t, traza in enumerate(trazas):
        nombre = _limpiar_texto_plotly(getattr(traza, "name", None), f"Serie {idx_t + 1}")
        ty = getattr(traza, "y", None)
        if ty is None:
            continue
        ys = list(ty)
        tx = getattr(traza, "x", None)
        xs = list(tx) if tx is not None else list(range(len(ys)))

        pares: list[tuple[str, str, float]] = []
        for xv, yv in zip(xs, ys):
            if yv is None:
                continue
            try:
                yf = float(yv)
                if not math.isfinite(yf):
                    continue
            except (TypeError, ValueError):
                continue
            try:
                xf = float(xv)
                x_str = f"{xf:.6g}" if math.isfinite(xf) else str(xv)
            except (TypeError, ValueError):
                x_str = str(xv)
            pares.append((x_str, f"{yf:.6g}", yf))

        if not pares:
            continue

        ys_num = [p[2] for p in pares]
        if len(resumen_series) < 5:
            resumen_series.append(
                f"{nombre} ({len(pares)} pts, mín={min(ys_num):.4g}, máx={max(ys_num):.4g})"
            )

        idx_csv = muestrear_indices_acotados(len(pares), max_puntos=600)
        for i in idx_csv:
            filas_csv.append((nombre, pares[i][0], pares[i][1]))

        if len(filas_html) < 48:
            cupo = min(12, 48 - len(filas_html))
            idx_html = muestrear_indices_acotados(len(pares), max_puntos=cupo)
            for i in idx_html:
                filas_html.append((nombre, pares[i][0], pares[i][1]))

    if len(trazas) > 5:
        resumen_series.append(f"+{len(trazas) - 5} series adicionales")

    partes_desc = []
    if resumen_accesible:
        partes_desc.append(resumen_accesible.strip())
    if resumen_series:
        partes_desc.append(
            f"Gráfico '{titulo}' (horizontal: {ejex}; vertical: {ejey}). Series: "
            + "; ".join(resumen_series)
            + "."
        )
    else:
        partes_desc.append(f"Gráfico '{titulo}' sin puntos finitos disponibles.")

    resumen_final = " ".join(partes_desc)
    return titulo, ejex, ejey, resumen_final, filas_html, filas_csv


def render_plotly_arandu(
    fig: go.Figure,
    key: str | None = None,
    resumen_accesible: str | None = None,
    **kwargs: Any,
) -> None:
    """Renderiza figuras Plotly con resumen accesible, tabla semántica HTML y descarga CSV."""
    estilizar_figura_arandu(fig)
    config = kwargs.pop("config", PLOTLY_CONFIG_ARANDU)
    theme_arg = kwargs.pop("theme", None)
    st.plotly_chart(fig, use_container_width=True, config=config, theme=theme_arg, key=key, **kwargs)

    titulo, ejex, ejey, resumen_final, filas_html, filas_csv = _extraer_accesibilidad_plotly(
        fig, resumen_accesible=resumen_accesible
    )
    ctx = _obtener_ctx_ia()
    graf_key = str(key or titulo)
    md_graf = [f"- **Título:** {titulo}", f"- **Ejes:** `{ejex}` vs `{ejey}`", f"- **Resumen:** {resumen_final}"]
    if filas_html:
        muestra_pts = filas_html[:36]
        tbl_graf = [f"| Serie | {ejex} | {ejey} |", "| --- | --- | --- |"]
        for s_g, x_g, y_g in muestra_pts:
            tbl_graf.append(f"| {s_g} | {x_g} | {y_g} |")
        md_graf.append("\n".join(tbl_graf))
    ctx["graficos"][graf_key] = (titulo, "\n".join(md_graf))
    st.markdown(
        f'<p class="arandu-chart-desc" role="region" aria-live="polite" '
        f'aria-label="Resumen accesible: {html.escape(titulo)}">'
        f"<strong>Descripción accesible:</strong> {html.escape(resumen_final)}</p>",
        unsafe_allow_html=True,
    )

    if not filas_csv:
        return

    with st.expander("Datos del gráfico en tabla accesible y descarga CSV", expanded=False):
        filas_tr = "".join(
            f"<tr><td>{html.escape(s)}</td><td>{html.escape(x)}</td><td>{html.escape(y)}</td></tr>"
            for s, x, y in filas_html
        )
        st.markdown(
            '<div class="arandu-chart-table-wrap">'
            '<table class="arandu-chart-table">'
            f"<caption>Datos numéricos de {html.escape(titulo)}</caption>"
            "<thead><tr>"
            '<th scope="col">Serie</th>'
            f'<th scope="col">{html.escape(ejex)}</th>'
            f'<th scope="col">{html.escape(ejey)}</th>'
            "</tr></thead>"
            f"<tbody>{filas_tr}</tbody>"
            "</table>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Vista rápida de {len(filas_html)} puntos representativos ({len(filas_csv)} exportables en CSV)."
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Serie", ejex, ejey])
        writer.writerows(filas_csv)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", titulo.lower()).strip("_") or "grafico_arandu"
        dl_key = f"csv_dl_{key}" if key else f"csv_dl_{abs(hash((titulo, ejex, ejey, len(filas_csv))))}"
        st.download_button(
            "Descargar datos del gráfico (CSV)",
            data=buf.getvalue().encode("utf-8-sig"),
            file_name=f"{slug[:48]}.csv",
            mime="text/csv",
            key=dl_key,
        )


def render_error_validacion(
    campo: str,
    detalle: str,
    sugerencia: str,
    ejemplo: str | None = None,
) -> None:
    """Presenta errores de validación accesibles con diagnóstico y corrección sugerida (WCAG 3.3.3)."""
    ctx = _obtener_ctx_ia()
    ej_txt = f"\n- **Ejemplo válido:** `{ejemplo}`" if ejemplo else ""
    ctx["errores"][str(campo)] = (
        f"Error de validación ({campo})",
        f"- **Qué ocurrió:** {detalle}\n- **Cómo corregirlo:** {sugerencia}{ej_txt}",
    )
    campo_esc = html.escape(str(campo))
    detalle_esc = html.escape(str(detalle))
    sugerencia_esc = html.escape(str(sugerencia))
    ejemplo_html = (
        f"<br/><strong>Ejemplo válido:</strong> <code>{html.escape(str(ejemplo))}</code>"
        if ejemplo
        else ""
    )
    st.markdown(
        f'<div class="arandu-error-callout" role="alert" aria-live="assertive" data-arandu-invalid-field="{campo_esc}">'
        f'<span class="arandu-error-tag">CORRECCIÓN REQUERIDA · {campo_esc}</span>'
        f"<strong>Qué ocurrió:</strong> {detalle_esc}<br/>"
        f"<strong>Cómo corregirlo:</strong> {sugerencia_esc}"
        f"{ejemplo_html}"
        "</div>",
        unsafe_allow_html=True,
    )


_REGLAS_ERROR_VALIDACION: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (
        ("signo", "bolzano", "mismo signo"),
        "Intervalo [a, b] o función f(x)",
        "Elegí extremos a < b donde f(a) y f(b) tengan signos opuestos (f(a)*f(b) < 0).",
        "f(x) = x**3 - 2*x - 5 con a = 2.0, b = 3.0",
    ),
    (
        ("singular", "pivote", "determinante", "división por cero", "division por cero", "dividir por cero", "cero"),
        "Sistema o denominador numérico",
        "Verificá que no haya ceros en el denominador ni filas repetidas, nulas o colineales.",
        "A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]",
    ),
    (
        ("equidistant", "equiespaciad", "duplicad", "orden", "paso h"),
        "Tabla de nodos (xᵢ, yᵢ) / paso Δh",
        "Ingresá valores de x estrictamente crecientes y sin repetir; en paso fijo usá espaciado h > 0 constante.",
        "x = [0.0, 0.2, 0.4, 0.6]",
    ),
    (
        ("sintaxis", "expresi", "sympy", "parse", "vací", "vaci"),
        "Expresión matemática",
        "Usá sintaxis Python explícita con * para multiplicar y ** para potencias.",
        "2*x**3 - 5*sin(x) + 1",
    ),
)


def render_excepcion_validacion(
    exc: Exception,
    campo: str,
    sugerencia_defecto: str,
    ejemplo_defecto: str | None = None,
) -> None:
    """Clasifica una excepción de validación numérica y muestra guía clara de corrección."""
    detalle = str(exc).strip() or "Entrada fuera del dominio esperado."
    texto_low = detalle.lower()
    for palabras, campo_regla, sug_regla, ej_regla in _REGLAS_ERROR_VALIDACION:
        if any(p in texto_low for p in palabras):
            render_error_validacion(
                campo=campo_regla,
                detalle=detalle,
                sugerencia=sug_regla,
                ejemplo=ej_regla,
            )
            return
    render_error_validacion(
        campo=campo,
        detalle=detalle,
        sugerencia=sugerencia_defecto,
        ejemplo=ejemplo_defecto,
    )


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


def render_data_editor_arandu(datos: Any, **kwargs: Any) -> Any:
    """Envuelve st.data_editor con encabezado técnico y borde punteado para evidenciar edición en vivo."""
    st.markdown(
        '<div class="arandu-tbl-tag-edit">[ TABLA INTERACTIVA — Hacé doble clic en cualquier celda para editar los valores en vivo ]</div>',
        unsafe_allow_html=True,
    )
    resultado = st.data_editor(datos, **kwargs)
    ctx = _obtener_ctx_ia()
    ed_key = str(kwargs.get("key") or f"tabla_{len(ctx['editores']) + 1}")
    try:
        ctx["editores"][ed_key] = (f"Tabla de entrada ({ed_key})", _df_a_markdown(resultado))
    except Exception:
        pass
    return resultado


def render_dataframe_acotado(
    datos: Any,
    max_filas: int = 300,
    key_prefix: str = "tbl",
    hide_index: bool = True,
    **kwargs: Any,
) -> None:
    """Renderiza tablas grandes paginadas en bloques de <= 300 filas para evitar saturar el navegador."""
    import pandas as pd

    st.markdown(
        '<div class="arandu-tbl-tag-readonly">[ RESULTADOS CALCULADOS (SOLO LECTURA) ]</div>',
        unsafe_allow_html=True,
    )
    df = datos if isinstance(datos, pd.DataFrame) else pd.DataFrame(datos)
    ctx = _obtener_ctx_ia()
    try:
        ctx["tablas"][key_prefix] = (f"Tabla de resultados ({key_prefix})", _df_a_markdown(df, max_filas=80))
    except Exception:
        pass
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
        st.caption(f"{label}: **{max_val}** (único paso disponible)")
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


def _activar_ficha_metodo(state_key: str, idx: int) -> None:
    st.session_state[state_key] = idx


def render_selector_fichas_metodos(
    vista_id: str,
    fichas: list[dict[str, str]] | tuple[dict[str, str], ...],
    default_idx: int = 0,
) -> int:
    """Renderiza un selector interactivo de fichas de estudio y retorna el índice activo."""
    if not fichas:
        return 0
    state_key = f"arandu_ficha_{vista_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_idx
    activo = max(0, min(len(fichas) - 1, int(st.session_state[state_key])))
    st.session_state[state_key] = activo

    n = len(fichas)
    cols_por_fila = 4 if n == 4 else (3 if n >= 3 else n)
    for ini in range(0, n, cols_por_fila):
        fila = fichas[ini : ini + cols_por_fila]
        cols = st.columns(cols_por_fila)
        for offset, ficha in enumerate(fila):
            idx = ini + offset
            es_activa = idx == activo
            with cols[offset]:
                st.markdown(
                    construir_html_ficha_metodo(
                        numero=ficha["numero"],
                        clave_svg=ficha["clave_svg"],
                        titulo=ficha["titulo"],
                        subtitulo=ficha["subtitulo"],
                        resumen=ficha["resumen"],
                        pista=ficha["pista"],
                        activa=es_activa,
                    ),
                    unsafe_allow_html=True,
                )
                st.button(
                    "Método activo" if es_activa else f"Seleccionar método {ficha['numero']}",
                    key=f"ficha_{vista_id}_{idx}",
                    type="primary" if es_activa else "secondary",
                    use_container_width=True,
                    on_click=_activar_ficha_metodo,
                    args=(state_key, idx),
                )

    ficha_activa = fichas[activo]
    ctx = _obtener_ctx_ia()
    if not ctx.get("pagina"):
        ctx["pagina"] = _TITULOS_VISTA_IA.get(vista_id, vista_id.capitalize())
    ctx["metodo"] = {
        "vista_id": vista_id,
        "numero": str(ficha_activa["numero"]),
        "titulo": str(ficha_activa["titulo"]),
        "subtitulo": str(ficha_activa["subtitulo"]),
        "resumen": str(ficha_activa["resumen"]),
        "pista": str(ficha_activa["pista"]),
        "disponibles": [f"{f['numero']}. {f['titulo']}" for f in fichas],
    }
    titulo_esc = html.escape(f"Método {ficha_activa['numero']} — {ficha_activa['titulo']}")
    pista_esc = html.escape(ficha_activa["pista"])
    st.markdown(
        '<div class="arandu-ficha-activa-banner" role="status" aria-live="polite">'
        '<span class="arandu-eyebrow">NOTA DE CÁTEDRA</span>'
        f'<span class="arandu-ficha-activa-titulo">{titulo_esc}</span>'
        f'<span class="arandu-ficha-activa-pista">{pista_esc}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    return activo


def _seccion_teoria_markdown(teoria: dict[str, Any] | None) -> list[str]:
    if not teoria:
        return []
    partes = ["## 1. Marco Teórico y Fórmulas Oficiales"]
    partes.append(f"- **Método:** {teoria.get('nombre', '')}")
    if teoria.get("resumen_corto"):
        partes.append(f"- **Resumen:** {teoria['resumen_corto']}")
    guru = teoria.get("guru", {})
    if guru.get("que_resuelve"):
        partes.append(f"- **Qué resuelve:** {guru['que_resuelve']}")
    if guru.get("intuicion_geometrica"):
        partes.append(f"- **Intuición:** {guru['intuicion_geometrica']}")
    pasos_alg = guru.get("algoritmo_pasos", [])
    if pasos_alg:
        partes.append("**Algoritmo:**")
        partes.extend(f"- {p}" for p in pasos_alg)
    formulas = teoria.get("formulas_latex", [])
    if formulas:
        partes.append("**Fórmulas oficiales (LaTeX):**")
        for f_tex in formulas:
            partes.append(f"$$\n{f_tex}\n$$")
    if guru.get("cuando_usar"):
        partes.append("**Cuándo elegirlo:** " + " | ".join(guru["cuando_usar"]))
    if guru.get("cuando_falla"):
        partes.append("**Cuándo evitarlo / tener cuidado:** " + " | ".join(guru["cuando_falla"]))
    ej = teoria.get("ejemplo_teorico", {})
    if ej:
        partes.append(f"**Ejemplo teórico ({ej.get('titulo', '')}):** {ej.get('enunciado', '')}")
        for idx, paso in enumerate(ej.get("pasos", []), start=1):
            partes.append(f"{idx}. {paso}")
        if ej.get("resultado_final"):
            partes.append(f"- **Conclusión:** {ej['resultado_final']}")
    return partes


def construir_markdown_contexto_ia(
    pagina_actual: str | None = None,
    incluir_teoria: bool = True,
) -> tuple[str, int]:
    """Construye un documento Markdown + LaTeX completo con el estado actual de la vista para una IA."""
    ctx = _obtener_ctx_ia()
    pagina = ctx.get("pagina") or pagina_actual or "Vista actual de Arandú"
    metodo = ctx.get("metodo") or {}
    partes = [
        f"# Contexto completo de Arandú — {pagina}",
        "- **Plataforma:** Arandú: Métodos Computacionales (FaCENA — UNNE)",
        f"- **Página activa:** {pagina}",
    ]
    if metodo:
        partes.append(
            f"- **Método seleccionado:** {metodo.get('numero', '')} — {metodo.get('titulo', '')} ({metodo.get('pista', '')})"
        )
        if metodo.get("resumen"):
            partes.append(f"- **Descripción del método:** {metodo['resumen']}")
    partes.append("")

    secciones_contadas = 1
    if incluir_teoria and ctx.get("teoria"):
        partes.extend(_seccion_teoria_markdown(ctx["teoria"]))
        partes.append("")
        secciones_contadas += 1

    practicas = ctx.get("practica") or {}
    if practicas:
        partes.append("## 2. Consigna de Práctica de Cátedra")
        for p_item in practicas.values():
            if p_item.get("titulo"):
                partes.append(f"### {p_item['titulo']}")
            if p_item.get("enunciado"):
                partes.append(p_item["enunciado"])
            if p_item.get("solucion_referencia"):
                partes.append(f"- **Solución de referencia:** {p_item['solucion_referencia']}")
        partes.append("")
        secciones_contadas += 1

    controles = ctx.get("controles") or {}
    editores = ctx.get("editores") or {}
    if controles or editores:
        partes.append("## 3. Parámetros y Datos de Entrada Activos en Vivo")
        for clave_st, info_c in controles.items():
            partes.append(f"### Controles (`{clave_st}` — {info_c.get('modo', '')})")
            if info_c.get("mensaje"):
                partes.append(f"- **Nota activa:** {info_c['mensaje']}")
            for pk, pv in (info_c.get("parametros") or {}).items():
                partes.append(f"- `{pk}` = `{pv}`")
        for tit_e, md_e in editores.values():
            partes.append(f"### {tit_e}")
            partes.append(md_e)
        partes.append("")
        secciones_contadas += 1

    resultados = ctx.get("resultados") or {}
    tablas = ctx.get("tablas") or {}
    if resultados or tablas:
        partes.append("## 4. Resultados Calculados, Fórmulas Evaluadas y Tablas")
        for tit_r, md_r in resultados.values():
            partes.append(f"### {tit_r}")
            partes.append(md_r)
        for tit_t, md_t in tablas.values():
            partes.append(f"### {tit_t}")
            partes.append(md_t)
        partes.append("")
        secciones_contadas += 1

    errores = ctx.get("errores") or {}
    if errores:
        partes.append("## Alertas y Errores de Validación Activos")
        for tit_err, md_err in errores.values():
            partes.append(f"### {tit_err}")
            partes.append(md_err)
        partes.append("")
        secciones_contadas += 1

    graficos = ctx.get("graficos") or {}
    if graficos:
        partes.append("## 5. Gráficos Interactivos y Datos Muestreados")
        for tit_g, md_g in graficos.values():
            partes.append(f"### {tit_g}")
            partes.append(md_g)
        partes.append("")
        secciones_contadas += 1

    return "\n".join(partes).strip(), secciones_contadas


def _render_contenido_exportador_ia(pagina_actual: str | None = None) -> None:
    incluir_teoria = st.checkbox(
        "Incluir marco teórico completo y fórmulas oficiales de cátedra",
        value=True,
        key="ui_ia_incluir_teoria",
    )
    texto_ia, n_sec = construir_markdown_contexto_ia(
        pagina_actual=pagina_actual,
        incluir_teoria=incluir_teoria,
    )
    st.caption(
        f"{n_sec} secciones capturadas · {len(texto_ia)} caracteres en formato Markdown + LaTeX. "
        "Hacé clic en el ícono de copiar en la esquina superior derecha del recuadro o descargá el archivo."
    )
    st.code(texto_ia, language="markdown", wrap_lines=True, height=380)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (pagina_actual or "arandu").lower()).strip("_") or "arandu"
    st.download_button(
        "Descargar archivo .md para IA",
        data=texto_ia.encode("utf-8"),
        file_name=f"contexto_ia_{slug[:40]}.md",
        mime="text/markdown",
        key=f"dl_ia_{slug[:24]}",
        use_container_width=True,
    )


if hasattr(st, "dialog"):

    @st.dialog("Copiar todo en texto para IA (Markdown + LaTeX)", width="large")
    def mostrar_dialogo_copiar_ia(pagina_actual: str | None = None) -> None:
        """Muestra en una ventana accesible todo el contexto actual listo para copiar hacia una IA."""
        _render_contenido_exportador_ia(pagina_actual=pagina_actual)

else:

    def mostrar_dialogo_copiar_ia(pagina_actual: str | None = None) -> None:
        with st.expander("Copiar todo en texto para IA (Markdown + LaTeX)", expanded=True):
            _render_contenido_exportador_ia(pagina_actual=pagina_actual)


def render_boton_copiar_ia(
    pagina_actual: str | None = None,
    key: str = "btn_ia_copiar",
    label: str = "Copiar todo en texto para IA",
) -> None:
    """Renderiza el botón que sincroniza los datos actuales y abre el texto completo para copiar."""
    if st.button(label, key=key, type="primary", use_container_width=True):
        mostrar_dialogo_copiar_ia(pagina_actual=pagina_actual)


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
    "construir_markdown_contexto_ia",
    "crear_figura_base_plotly",
    "estilizar_figura_arandu",
    "html_tabla_a_markdown",
    "inicializar_estado_practica",
    "inject_arandu_css",
    "limpiar_valores_plotly",
    "mostrar_dialogo_copiar_ia",
    "muestrear_para_grafico",
    "obtener_logo_data_uri",
    "registrar_contexto_ia",
    "registrar_pagina_ia",
    "reiniciar_contexto_ia",
    "render_banner_edicion_profesora",
    "render_bloque_1_teoria_y_ejemplo",
    "render_boton_copiar_ia",
    "render_controles_edicion_practica",
    "render_data_editor_arandu",
    "render_dataframe_acotado",
    "render_error_validacion",
    "render_excepcion_validacion",
    "render_mini_catalogo_vista",
    "render_plotly_arandu",
    "render_practica_catedra_info",
    "render_selector_fichas_metodos",
    "render_slider_paso_k",
    "render_subtitulo_controles_vivo",
    "render_tarjeta_catalogo",
    "rerun_fragmento_seguro",
    "restaurar_estado_practica",
]
