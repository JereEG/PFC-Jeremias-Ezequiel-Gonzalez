"""
Vista de Sistemas de Ecuaciones Lineales y Autovalores - Arandú.
Gauss, Gauss-Jordan, LU (Doolittle), Gauss-Seidel, Potencias (Max/Min) y Faddeev-Leverrier.
Separación Static-First con aislamiento mediante @st.fragment.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.sistemas_autovalores import (
    eliminacion_gauss,
    factorizacion_lu,
    gauss_jordan,
    gauss_seidel,
    metodo_faddeev_leverrier,
    metodo_potencias_max,
    metodo_potencias_min,
    verificar_dominancia_diagonal,
)
from views.theme import PALETA_ARANDU
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    limpiar_valores_plotly,
    muestrear_para_grafico,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_data_editor_arandu,
    render_dataframe_acotado,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
)


def _editor_sistema(clave_state: str, solo_lectura: bool, incluir_x0: bool = False) -> tuple[list[list[float]], list[float], list[float] | None]:
    A = st.session_state[f"{clave_state}_A"]
    b = st.session_state[f"{clave_state}_b"]
    n = len(A)
    cols_A = [f"a_{i+1}" for i in range(n)]
    df_mat = pd.DataFrame(A, columns=cols_A)
    df_mat["b"] = b
    if incluir_x0:
        x0 = st.session_state[f"{clave_state}_x0"]
        df_mat["x0"] = x0

    df_edit = render_data_editor_arandu(
        df_mat,
        use_container_width=True,
        hide_index=True,
        disabled=solo_lectura,
        key=f"{clave_state}__editor_df",
    )
    if not solo_lectura:
        A = df_edit[cols_A].astype(float).values.tolist()
        b = df_edit["b"].astype(float).tolist()
        st.session_state[f"{clave_state}_A"] = A
        st.session_state[f"{clave_state}_b"] = b
        if incluir_x0:
            x0 = df_edit["x0"].astype(float).tolist()
            st.session_state[f"{clave_state}_x0"] = x0
            return A, b, x0
    return A, b, (st.session_state[f"{clave_state}_x0"] if incluir_x0 else None)


@st.fragment
def _frag_gauss(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="sis_gauss",
        defaults_dict=defaults,
        mensaje_edicion=r"Edita la matriz de coeficientes $A$ y el vector independiente $b$ directamente en la tabla.",
    )
    solo_lectura = modo == "catedra"
    A, b, _ = _editor_sistema("sis_gauss", solo_lectura)

    st.markdown("### Bloque 3 — Escalonamiento y Sustitución Regresiva")
    try:
        res = eliminacion_gauss(A, b)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A y vector b (Gauss)",
            sugerencia_defecto="Verificá que A sea cuadrada no singular y sin ceros insalvables tras pivoteo parcial.",
            ejemplo_defecto="A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]], b = [2, 6, 2]",
        )
        return

    cols = st.columns(len(res["solucion"]) + 1)
    for i, val in enumerate(res["solucion"]):
        cols[i].metric(f"x_{i+1}", f"{val:.6f}")
    cols[-1].metric("‖Ax - b‖₂", f"{res['norma_error']:.3e}")

    pasos = res["pasos_escalonamiento"]
    if pasos:
        st.markdown("### Bloque 4 — Matriz Aumentada Paso a Paso")
        k_vis = render_slider_paso_k("Ver paso de escalonamiento $k$:", len(pasos), key="vis_gauss_k")
        paso = pasos[k_vis - 1]
        st.info(paso["descripcion"])
        render_dataframe_acotado(paso["matriz"], max_filas=300, key_prefix="tbl_sis_gauss", hide_index=False)

    fig = crear_figura_base_plotly("Solución xᵢ y Magnitud de Componentes", "Componente", "Valor", altura=320)
    fig.add_trace(
        go.Bar(
            x=[f"x_{i+1}" for i in range(len(res["solucion"]))],
            y=res["solucion"],
            name="Solución x",
            marker_color=PALETA_ARANDU["oro"],
            marker_line_color=PALETA_ARANDU["terracota_oscuro"],
            marker_line_width=1.2,
        )
    )
    render_plotly_arandu(fig, key="plot_sis_gauss")


def _tab_gauss() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("eliminacion_gauss")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
        },
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_gauss(defaults)


@st.fragment
def _frag_gauss_jordan(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="sis_gj",
        defaults_dict=defaults,
        mensaje_edicion=r"Edita $A$ y $b$, o conmuta entre resolver $Ax = b$ e invertir $A$.",
    )
    solo_lectura = modo == "catedra"
    A, b, _ = _editor_sistema("sis_gj", solo_lectura)
    modo_gj = st.radio(
        "Objetivo de Gauss-Jordan:",
        ["Resolver Ax = b", "Calcular Inversa A⁻¹"],
        horizontal=True,
        key="ui_gj_modo",
    )

    st.markdown("### Bloque 3 — Diagonalización Completa en Vivo")
    try:
        res = gauss_jordan(A, b if modo_gj.startswith("Resolver") else None)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A (Gauss-Jordan)",
            sugerencia_defecto="La matriz A debe ser cuadrada e invertible (determinante distinto de cero).",
            ejemplo_defecto="A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]",
        )
        return

    if res["modo"] == "sistema":
        cols = st.columns(len(res["solucion"]) + 1)
        for i, val in enumerate(res["solucion"]):
            cols[i].metric(f"x_{i+1}", f"{val:.6f}")
        cols[-1].metric("‖Ax - b‖₂", f"{res['norma_error']:.3e}")
    else:
        st.metric("‖A·A⁻¹ - I‖₂", f"{res['norma_error']:.3e}")
        render_dataframe_acotado(res["inversa"], max_filas=300, key_prefix="tbl_sis_gj_inv", hide_index=False)

    pasos = res["pasos"]
    if pasos:
        st.markdown("### Bloque 4 — Transformación a Identidad Paso a Paso")
        k_vis = render_slider_paso_k("Ver paso de pivoteo $k$:", len(pasos), key="vis_gj_k")
        paso = pasos[k_vis - 1]
        st.info(paso["descripcion"])
        render_dataframe_acotado(paso["matriz"], max_filas=300, key_prefix="tbl_sis_gj_paso", hide_index=False)


def _tab_gauss_jordan() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("gauss_jordan")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
        },
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_gauss_jordan(defaults)


@st.fragment
def _frag_lu(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="sis_lu",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica $A$ y $b$ para inspeccionar los factores triangulares $L$ y $U$.",
    )
    solo_lectura = modo == "catedra"
    A, b, _ = _editor_sistema("sis_lu", solo_lectura)

    st.markdown("### Bloque 3 — Factores $L$, $U$ y Solución ($Ly=b$, $Ux=y$)")
    try:
        res = factorizacion_lu(A, b)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A (Factorización LU)",
            sugerencia_defecto="Doolittle sin pivoteo requiere menores principales no nulos (u_kk != 0); reordená filas si u_kk = 0.",
            ejemplo_defecto="A = [[4, -2, 1], [20, -7, 12], [-8, 13, 17]]",
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matriz Triangular Inferior $L$:**")
        render_dataframe_acotado(res["L"], max_filas=300, key_prefix="tbl_sis_lu_l", hide_index=False)
    with c2:
        st.markdown("**Matriz Triangular Superior $U$:**")
        render_dataframe_acotado(res["U"], max_filas=300, key_prefix="tbl_sis_lu_u", hide_index=False)

    cols = st.columns(len(res["solucion"]) + 1)
    for i, val in enumerate(res["solucion"]):
        cols[i].metric(f"x_{i+1}", f"{val:.6f}")
    cols[-1].metric("Error ‖LU - A‖", f"{res['error_LU']:.2e}")

    pasos = res["pasos_eliminacion"]
    if pasos:
        st.markdown("### Bloque 4 — Construcción Paso a Paso de $L$ y $U$")
        k_vis = render_slider_paso_k("Ver columna eliminada $k$:", len(pasos), key="vis_lu_k")
        paso = pasos[k_vis - 1]
        c_l, c_u = st.columns(2)
        with c_l:
            st.caption(f"$L$ tras paso ${k_vis}$")
            render_dataframe_acotado(paso["L_actual"], max_filas=300, key_prefix="tbl_sis_lu_kl", hide_index=False)
        with c_u:
            st.caption(f"$U$ tras paso ${k_vis}$")
            render_dataframe_acotado(paso["U_actual"], max_filas=300, key_prefix="tbl_sis_lu_ku", hide_index=False)


def _tab_lu() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("factorizacion_lu")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"A": [[4.0, -2.0, 1.0], [20.0, -7.0, 12.0], [-8.0, 13.0, 17.0]], "b": [11.0, 70.0, 17.0]},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_lu(defaults)


@st.fragment
def _frag_gauss_seidel(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="sis_gs",
        defaults_dict=defaults,
        mensaje_edicion=r"Ajusta la matriz $A$, el vector $b$, la semilla $x^{(0)}$ o los criterios del bucle.",
    )
    solo_lectura = modo == "catedra"
    A, b, x0 = _editor_sistema("sis_gs", solo_lectura, incluir_x0=True)

    c1, c2 = st.columns(2)
    tol_val = c1.number_input("Tolerancia $E$:", value=float(st.session_state["sis_gs_tol"]), min_value=1e-12, format="%.6f", disabled=solo_lectura, key="ui_gs_tol")
    max_it = c2.slider("Iteraciones máx:", 5, 100, int(st.session_state["sis_gs_max_iter"]), disabled=solo_lectura, key="ui_gs_max")
    if not solo_lectura:
        st.session_state["sis_gs_tol"] = tol_val
        st.session_state["sis_gs_max_iter"] = max_it

    dom = verificar_dominancia_diagonal(A)
    if dom["es_dominante"]:
        st.success(
            f"**{dom['diagnostico']}** Cada fila tiene peso dominante en la diagonal "
            r"($|a_{ii}| > \sum_{j\neq i}|a_{ij}|$): garantiza que el bucle reduzca el error en cada ciclo sin divergir."
        )
    else:
        st.warning(
            f"**{dom['diagnostico']}** Sin dominancia diagonal estricta, el lazo de actualización "
            "podría amplificar el error en lugar de amortiguarlo."
        )

    st.markdown("### Bloque 3 — Iteraciones de Gauss-Seidel en Vivo")
    try:
        res = gauss_seidel(A, b, x0=x0, tol=tol_val, max_iter=max_it)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A, vector b o tolerancia (Gauss-Seidel)",
            sugerencia_defecto="Asegurate de que ningún elemento diagonal a_ii sea 0; preferentemente usá una matriz diagonalmente dominante.",
            ejemplo_defecto="Diagonal dominante: |a_ii| > suma |a_ij|",
        )
        return

    cols = st.columns(len(res["solucion"]) + 2)
    for i, val in enumerate(res["solucion"]):
        cols[i].metric(f"x_{i+1}", f"{val:.6f}")
    cols[-2].metric("Iteraciones", res["iteraciones"])
    cols[-1].metric("Error final", f"{res['error_final']:.3e}")

    hist = res["historial"]
    filas_hist = [
        {"iteracion": h["iteracion"], **h["x_dict"], "error_max": h["error_max"], "residuo": h["norma_residuo"]}
        for h in hist
    ]
    render_dataframe_acotado(filas_hist, max_filas=300, key_prefix="tbl_sis_gs")

    if filas_hist:
        st.markdown("### Bloque 4 — Decaimiento Logarítmico del Error Paso a Paso")
        k_vis = render_slider_paso_k("Ver hasta la iteración $k$:", len(filas_hist), key="vis_gs_k")
        sub_hist = muestrear_para_grafico(filas_hist[:k_vis], max_puntos=220)
        fig = crear_figura_base_plotly(f"Convergencia de Gauss-Seidel hasta k = {k_vis}", "Iteración", "Error máximo (escala log)")
        fig.add_trace(
            go.Scatter(
                x=[r["iteracion"] for r in sub_hist],
                y=limpiar_valores_plotly([r["error_max"] for r in sub_hist]),
                mode="lines+markers",
                name="max |Δx|",
                line=dict(color=PALETA_ARANDU["oro"], width=2.5),
                marker=dict(size=8, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.6)),
            )
        )
        fig.update_yaxes(type="log")
        render_plotly_arandu(fig, key="plot_sis_gs")


def _tab_gauss_seidel() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("gauss_seidel")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
            "x0": [1.0, 1.0, 1.0],
            "tol": 1e-4,
            "max_iter": 50,
        },
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_gauss_seidel(defaults)


@st.fragment
def _frag_potencias(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="sis_pot",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica la matriz $A$, el vector semilla $x^{(0)}$ o selecciona el autovalor objetivo.",
    )
    solo_lectura = modo == "catedra"

    A = st.session_state["sis_pot_A"]
    x0 = st.session_state["sis_pot_x0"]
    cols_A = [f"a_{i+1}" for i in range(len(A))]
    df_in = pd.DataFrame(A, columns=cols_A)
    df_in["x0"] = x0
    df_edit = render_data_editor_arandu(df_in, use_container_width=True, hide_index=True, disabled=solo_lectura, key="sis_pot__editor_df")
    if not solo_lectura:
        A = df_edit[cols_A].astype(float).values.tolist()
        x0 = df_edit["x0"].astype(float).tolist()
        st.session_state["sis_pot_A"] = A
        st.session_state["sis_pot_x0"] = x0

    c1, c2 = st.columns(2)
    tol_pct = c1.number_input(r"Tolerancia $\varepsilon$ (%):", value=float(st.session_state["sis_pot_tol_pct"]), min_value=1e-6, format="%.5f", disabled=solo_lectura, key="ui_pot_tol")
    tipo = c2.radio("Autovalor buscado:", ["Mayor |λ_max| (Potencias Directo)", "Menor |λ_min| (Potencia Inversa)"], horizontal=True, key="ui_pot_tipo")
    if not solo_lectura:
        st.session_state["sis_pot_tol_pct"] = tol_pct

    st.markdown("### Bloque 3 — Iteración de Autovalores en Vivo")
    try:
        if tipo.startswith("Mayor"):
            res = metodo_potencias_max(A, x0=x0, tol=tol_pct, max_iter=int(st.session_state["sis_pot_max_iter"]))
            lam_final = res["autovalor_max"]
        else:
            res = metodo_potencias_min(A, x0=x0, tol=tol_pct, max_iter=int(st.session_state["sis_pot_max_iter"]))
            lam_final = res["autovalor_min"]
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A o vector inicial v0 (Potencias)",
            sugerencia_defecto="Ingresá una matriz cuadrada y un vector semilla no nulo; en Potencias Inversa, A debe ser invertible.",
            ejemplo_defecto="v0 = [1.0, 1.0, 1.0]",
        )
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Autovalor estimado λ", f"{lam_final:.6f}")
    m2.metric("Iteraciones", res["iteraciones"])
    m3.metric("Error % final", f"{res['error_final_pct']:.4f}%")

    hist = res["historial"]
    render_dataframe_acotado(hist, max_filas=300, key_prefix="tbl_sis_pot")

    if hist:
        st.markdown("### Bloque 4 — Estabilización de $\\lambda_k$ Paso a Paso")
        k_vis = render_slider_paso_k("Ver hasta la iteración $k$:", len(hist), key="vis_pot_k")
        sub_h = muestrear_para_grafico(hist[:k_vis], max_puntos=220)
        col_lam = "lambda" if "lambda" in sub_h[0] else "lambda_min"
        fig = crear_figura_base_plotly(f"Convergencia de λ_k hasta k = {k_vis}", "Iteración", "λ estimado")
        fig.add_trace(
            go.Scatter(
                x=[r["iteracion"] for r in sub_h],
                y=limpiar_valores_plotly([r[col_lam] for r in sub_h]),
                mode="lines+markers",
                name="λ_k",
                line=dict(color=PALETA_ARANDU["salvia"], width=2.5),
                marker=dict(size=8, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.6)),
            )
        )
        render_plotly_arandu(fig, key="plot_sis_pot")


def _tab_potencias() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("potencias")
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    defaults = {
        "A": raw_def.get("A", [[3.556, -1.778, 0.0], [-1.778, 3.556, -1.778], [0.0, -1.778, 3.556]]),
        "x0": raw_def.get("x0", [1.0, 1.0, 1.0]),
        "tol_pct": float(raw_def.get("tol", raw_def.get("tol_pct", 0.01))),
        "max_iter": int(raw_def.get("max_iter", 50)),
    }
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_potencias(defaults)


@st.fragment
def _frag_faddeev(defaults: dict) -> None:
    modo = render_controles_edicion_practica(
        clave_state="sis_fad",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica los elementos de $A$ para recalcular el polinomio característico, autovalores e inversa.",
    )
    solo_lectura = modo == "catedra"

    A = st.session_state["sis_fad_A"]
    cols_A = [f"c_{i+1}" for i in range(len(A))]
    df_edit = render_data_editor_arandu(pd.DataFrame(A, columns=cols_A), use_container_width=True, hide_index=True, disabled=solo_lectura, key="sis_fad__editor_df")
    if not solo_lectura:
        A = df_edit[cols_A].astype(float).values.tolist()
        st.session_state["sis_fad_A"] = A

    st.markdown("### Bloque 3 — Polinomio Característico, Autovalores e Inversa")
    try:
        res = metodo_faddeev_leverrier(A)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A (Faddeev-Leverrier)",
            sugerencia_defecto="Ingresá una matriz cuadrada numérica finita de orden n >= 2.",
            ejemplo_defecto="A = [[4, 1, 0], [1, 4, 1], [0, 1, 4]]",
        )
        return

    st.info(f"**Polinomio característico:** `{res['polinomio_texto']}`")
    m1, m2, m3 = st.columns(3)
    m1.metric("det(A)", f"{res['determinante']:.6f}")
    m2.metric("Autovalores", ", ".join(f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in res["autovalores"]))
    m3.metric("Error A⁻¹", f"{res['error_inversa']:.2e}" if res["error_inversa"] is not None else "Singular")

    if res["inversa"] is not None:
        with st.expander(r"Ver Matriz Inversa $A^{-1}$ obtenida por Faddeev"):
            render_dataframe_acotado(res["inversa"], max_filas=300, key_prefix="tbl_sis_fad_inv", hide_index=False)

    st.markdown(r"### Bloque 4 — Raíces del Polinomio Característico $P(\lambda) = 0$")
    curva = res["curva_polinomio"]
    fig = crear_figura_base_plotly("Polinomio Característico P(λ)", "λ", "P(λ)")
    fig.add_hline(y=0, line_dash="dash", line_color=PALETA_ARANDU["terracota_oscuro"])
    fig.add_trace(
        go.Scatter(
            x=curva["lambda"],
            y=limpiar_valores_plotly(curva["p_lambda"], limite_abs=1e9),
            mode="lines",
            name="P(λ)",
            line=dict(color=PALETA_ARANDU["oro"], width=2.5),
        )
    )
    reales = res["autovalores_reales"]
    if reales:
        k_vis = render_slider_paso_k("Destacar hasta el autovalor real $k$:", len(reales), key="vis_fad_k")
        fig.add_trace(
            go.Scatter(
                x=reales[:k_vis],
                y=[0.0] * k_vis,
                mode="markers",
                name="Autovalores reales",
                marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
            )
        )
    render_plotly_arandu(fig, key="plot_sis_fad")


def _tab_faddeev() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("faddeev_leverrier")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"A": [[3.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 3.0]]},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_faddeev(defaults)


def vista_sistemas() -> None:
    inject_arandu_css()
    st.title("Sistemas Lineales y Autovalores")
    st.markdown(
        r"Métodos directos ($O(n^3)$ por triangulación), esquemas iterativos condicionados por dominancia diagonal "
        r"y extracción del espectro característico."
    )
    activo = render_selector_fichas_metodos(
        "sistemas",
        (
            {
                "numero": "1",
                "clave_svg": "gauss",
                "titulo": "Eliminación de Gauss",
                "subtitulo": "Triangulación · Costo O(n³)",
                "resumen": "Triangula la matriz aumentada con pivoteo parcial y despeja incógnitas mediante sustitución regresiva.",
                "pista": "Triangular superior · Pivoteo parcial",
            },
            {
                "numero": "2",
                "clave_svg": "gauss_jordan",
                "titulo": "Gauss-Jordan",
                "subtitulo": "Matriz diagonal · Inversa",
                "resumen": "Reduce la matriz directamente a la forma identidad I para obtener soluciones y matriz inversa.",
                "pista": "Identidad I · Matriz inversa",
            },
            {
                "numero": "3",
                "clave_svg": "lu",
                "titulo": "Factorización LU",
                "subtitulo": "Descomposición · A = L·U",
                "resumen": "Descompone la matriz en factores triangulares L y U para resolver múltiples vectores independientes.",
                "pista": "Factores reutilizables · A = L·U",
            },
            {
                "numero": "4",
                "clave_svg": "gauss_seidel",
                "titulo": "Gauss-Seidel",
                "subtitulo": "Iterativo · Dominancia diagonal",
                "resumen": "Actualiza cada incógnita al instante asegurando convergencia cuando la matriz es diagonalmente dominante.",
                "pista": "Iterativo · Dominancia diagonal",
            },
            {
                "numero": "5",
                "clave_svg": "potencias",
                "titulo": "Método de Potencias",
                "subtitulo": "Autovalores · Cociente de Rayleigh",
                "resumen": "Extrae el autovalor dominante |λ_max| o el menor |λ_min| mediante multiplicación iterativa de vectores.",
                "pista": "Cociente de Rayleigh · Espectro",
            },
            {
                "numero": "6",
                "clave_svg": "faddeev",
                "titulo": "Faddeev-Leverrier",
                "subtitulo": "Espectro · Polinomio característico",
                "resumen": "Construye los coeficientes del polinomio característico completo usando potencias y trazas matriciales sucesivas.",
                "pista": "Espectro completo · Trazas matriciales",
            },
        ),
    )
    (
        _tab_gauss,
        _tab_gauss_jordan,
        _tab_lu,
        _tab_gauss_seidel,
        _tab_potencias,
        _tab_faddeev,
    )[activo]()


vista_sistemas()
