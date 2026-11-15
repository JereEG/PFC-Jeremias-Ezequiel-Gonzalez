"""
Arena Versus - Arandú: Métodos Computacionales.
Comparador directo restringido estrictamente a métodos de la misma familia matemática.
Muestra fichas técnicas en Markdown/KaTeX nativo, expectativas teóricas, alertas sobre casos límite
y simulación en vivo aislada con @st.fragment.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core._cache_utils import evaluar_fn_vectorizado
from core.catalogo_ejercicios import FICHAS_VERSUS, obtener_expectativa_versus
from core.edo_pvi import (
    compilar_exacta_edo,
    compilar_funcion_edo,
    metodo_euler,
    metodo_euler_modificado,
    metodo_milne,
    metodo_runge_kutta_2,
    metodo_runge_kutta_4,
)
from core.integracion_derivacion import (
    integrar_adaptativo_en_vivo,
    remuestrear_tabla_integracion,
)
from core.interpolacion import (
    diagnosticar_espaciamiento_nodos,
    evaluar_curva_gregory_parcial,
    evaluar_curva_lagrange,
    interpolar_lagrange,
    interpolar_newton_gregory_ascendente,
    interpolar_newton_gregory_descendente,
)
from core.raices import (
    compilar_funcion_raices,
    compilar_g_punto_fijo,
    metodo_biseccion,
    metodo_interpolacion_lineal_bolzano,
    metodo_newton_raphson,
    metodo_punto_fijo_aitken,
)
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
    render_plotly_arandu,
)


_RE_EMOJIS = re.compile(r"[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u20E3\uFE0F\u200D]")


def _sin_emojis(texto: str) -> str:
    """Elimina emojis y selectores de variación preservando texto, acentos y notación matemática."""
    return re.sub(r"\s{2,}", " ", _RE_EMOJIS.sub("", str(texto))).strip()


def _render_ficha_tecnica(titulo_slot: str, ficha: dict, color_acento: str) -> None:
    titulo_html = html.escape(f"{titulo_slot}: {ficha['nombre']}")
    with st.container(border=True):
        st.markdown(
            f'<div style="border-left: 3px solid {color_acento}; padding-left: 0.65rem; '
            f'margin-bottom: 0.6rem; font-size: 1.05rem; font-weight: 700; color: var(--arandu-text);">'
            f"{titulo_html}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Orden de convergencia / error:** {ficha['orden']}")
        st.markdown(f"**Costo computacional:** {ficha['costo']}")
        st.markdown(f"**Estabilidad:** {ficha['estabilidad']}")
        st.markdown(f"**Requisitos:** {ficha['requisitos']}")
        pros = ficha.get("pros", [])
        if pros:
            st.markdown('<div class="arandu-verdict-head arandu-verdict-head-pro">Ventajas</div>', unsafe_allow_html=True)
            for p in pros:
                st.markdown(f"- {p}")
        contras = ficha.get("contras", [])
        if contras:
            st.markdown('<div class="arandu-verdict-head arandu-verdict-head-warn">Limitaciones</div>', unsafe_allow_html=True)
            for c in contras:
                st.markdown(f"- {c}")


@st.fragment
def _duelo_raices(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown(r"## Arena en Vivo — Raíces sobre una misma Ecuación $f(x) = 0$")
    c1, c2 = st.columns(2)
    f_str = c1.text_input("Ecuación $f(x) = 0$:", value="x**3 - 2*x - 5", key="vs_raiz_f")
    g_str = c2.text_input("Despeje $x = g(x)$ (si participa Aitken):", value="(2*x + 5)**(1/3)", key="vs_raiz_g")

    c3, c4, c5, c6 = st.columns(4)
    a_val = c3.number_input("Extremo $a$:", value=2.0, format="%.4f", key="vs_raiz_a")
    b_val = c4.number_input("Extremo $b$:", value=2.5, format="%.4f", key="vs_raiz_b")
    x0_val = c5.number_input("Semilla $x_0$ (Newton/Aitken):", value=2.5, format="%.4f", key="vs_raiz_x0")
    tol_val = c6.number_input("Tolerancia $E$:", value=1e-4, min_value=1e-10, format="%.6f", key="vs_raiz_tol")

    comp_f = compilar_funcion_raices(f_str)
    comp_g = compilar_g_punto_fijo(g_str) if ("aitken" in (m_a, m_b)) else None

    def _correr(clave: str):
        if clave == "biseccion":
            r = metodo_biseccion(comp_f["fn"], a_val, b_val, tol=tol_val, max_iter=80)
            errs = [h["error"] for h in r["historial"] if h.get("error") is not None]
            return r["raiz"], r["iteraciones"], r["error_final"], r["duracion"], errs
        if clave == "regula_falsi":
            r = metodo_interpolacion_lineal_bolzano(comp_f["fn"], a_val, b_val, tol=tol_val, max_iter=80)
            errs = [h["error"] for h in r["historial"] if h.get("error") is not None]
            return r["raiz"], r["iteraciones"], r["error_final"], r["duracion"], errs
        if clave == "newton_raphson":
            r = metodo_newton_raphson(comp_f["fn"], comp_f["d1"], x0_val, tol=tol_val, max_iter=50)
            errs = [h["error"] for h in r["historial"] if h.get("error") is not None]
            return r["raiz"], r["iteraciones"], r["error_final"], r["duracion"], errs
        r_pf = metodo_punto_fijo_aitken(comp_g["gn"], x0_val, tol=tol_val, max_iter=50)["aitken"]
        errs = [h["error"] for h in r_pf["historial"] if h.get("error") is not None]
        return r_pf["raiz"], r_pf["ciclos"], r_pf["error_final"], r_pf["tiempo"], errs

    try:
        ra, ita, ea, _ta, err_a = _correr(m_a)
        rb, itb, eb, _tb, err_b = _correr(m_b)
    except Exception as exc:
        st.error(f"Error ejecutando el duelo de raíces: {exc}")
        return

    ca, cb = st.columns(2)
    ca.metric(f"A: {nombre_a}", f"{ra:.7f}", f"{ita} pasos{' (menor)' if ita < itb else ''} | Err {ea:.2e}")
    cb.metric(f"B: {nombre_b}", f"{rb:.7f}", f"{itb} pasos{' (menor)' if itb < ita else ''} | Err {eb:.2e}")

    fig = crear_figura_base_plotly("Velocidad de Convergencia (Error Logarítmico por Paso)", "Iteración / Ciclo", "Error |Δx| (escala log)")
    if err_a:
        err_a_s = muestrear_para_grafico([max(e, 1e-16) for e in err_a], max_puntos=220)
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(err_a_s) + 1)),
                y=err_a_s,
                mode="lines+markers",
                name=nombre_a,
                line=dict(color=PALETA_ARANDU["oro"], width=2.5),
                marker=dict(size=7, color=PALETA_ARANDU["oro"]),
            )
        )
    if err_b:
        err_b_s = muestrear_para_grafico([max(e, 1e-16) for e in err_b], max_puntos=220)
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(err_b_s) + 1)),
                y=err_b_s,
                mode="lines+markers",
                name=nombre_b,
                line=dict(color=PALETA_ARANDU["terracota"], width=2.5, dash="dot"),
                marker=dict(size=7, color=PALETA_ARANDU["terracota"]),
            )
        )
    fig.update_yaxes(type="log")
    render_plotly_arandu(fig, key="plot_vs_raices")


@st.fragment
def _duelo_sistemas(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown(r"## Arena en Vivo — Sistemas Lineales $Ax = b$")
    default_mat = pd.DataFrame(
        [[18.72, 8.20, 8.76, 121.280], [6.40, 15.90, 7.18, 126.321], [5.22, 6.40, 14.31, 118.522]],
        columns=["a_1", "a_2", "a_3", "b"],
    )
    st.caption(r"Modifica la matriz $A$ o el vector $b$ para ensayar matrices con y sin dominancia diagonal:")
    df_ed = st.data_editor(default_mat, use_container_width=True, hide_index=True, key="vs_sis_mat")
    A = df_ed[["a_1", "a_2", "a_3"]].astype(float).values.tolist()
    b = df_ed["b"].astype(float).tolist()
    tol = st.number_input("Tolerancia $E$ (si participa Gauss-Seidel):", value=1e-4, format="%.6f", key="vs_sis_tol")

    dom = verificar_dominancia_diagonal(A)
    if not dom["es_dominante"] and ("gauss_seidel" in (m_a, m_b)):
        st.warning(dom["diagnostico"])

    def _correr(clave: str):
        if clave == "gauss":
            r = eliminacion_gauss(A, b)
            return r["solucion"], r["norma_error"], r["duracion"], "Directo (O(n³))"
        if clave == "gauss_jordan":
            r = gauss_jordan(A, b)
            return r["solucion"], r["norma_error"], r["duracion"], "Directo (O(n³))"
        if clave == "lu":
            r = factorizacion_lu(A, b)
            return r["solucion"], r["norma_error"], r["duracion"], "Factorización LU"
        r = gauss_seidel(A, b, x0=[1.0] * len(b), tol=tol, max_iter=50)
        return r["solucion"], r["norma_error"], r["duracion"], f"Iterativo ({r['iteraciones']} iters)"

    try:
        sa, na, _ta, d_a = _correr(m_a)
        sb, nb, _tb, d_b = _correr(m_b)
    except Exception as exc:
        st.error(str(exc))
        return

    ca, cb = st.columns(2)
    ca.metric(f"A: {nombre_a}", d_a, f"‖Ax-b‖ = {na:.2e}{' (menor)' if na < nb else ''}")
    cb.metric(f"B: {nombre_b}", d_b, f"‖Ax-b‖ = {nb:.2e}{' (menor)' if nb < na else ''}")

    fig = crear_figura_base_plotly("Comparación de Soluciones xᵢ", "Incógnita", "Valor xᵢ", altura=340)
    etiquetas = [f"x_{i+1}" for i in range(len(sa))]
    fig.add_trace(go.Bar(x=etiquetas, y=sa, name=nombre_a, marker_color=PALETA_ARANDU["oro"]))
    fig.add_trace(go.Bar(x=etiquetas, y=sb, name=nombre_b, marker_color=PALETA_ARANDU["terracota"]))
    fig.update_layout(barmode="group")
    render_plotly_arandu(fig, key="plot_vs_sistemas")


@st.fragment
def _duelo_autovalores(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("## Arena en Vivo — Autovalores de una Matriz Cuadrada")
    default_a = pd.DataFrame(
        [[3.556, -1.778, 0.0], [-1.778, 3.556, -1.778], [0.0, -1.778, 3.556]],
        columns=["c_1", "c_2", "c_3"],
    )
    df_ed = st.data_editor(default_a, use_container_width=True, hide_index=True, key="vs_aut_mat")
    A = df_ed.astype(float).values.tolist()

    def _correr(clave: str):
        if clave == "potencias_max":
            r = metodo_potencias_max(A, tol=0.01, max_iter=60)
            return f"λ_max = {r['autovalor_max']:.5f}", r["iteraciones"], [h["lambda"] for h in r["historial"]]
        if clave == "potencias_min":
            r = metodo_potencias_min(A, tol=0.01, max_iter=60)
            return f"λ_min = {r['autovalor_min']:.5f}", r["iteraciones"], [h["lambda"] for h in r["historial"]]
        r = metodo_faddeev_leverrier(A)
        reales = sorted(r["autovalores_reales"])
        txt = ", ".join(f"{v:.4f}" for v in reales) if reales else str(r["autovalores"])
        return f"Espectro: [{txt}]", len(A), reales

    try:
        va, ita, sa = _correr(m_a)
        vb, itb, sb = _correr(m_b)
    except Exception as exc:
        st.error(str(exc))
        return

    ca, cb = st.columns(2)
    ca.metric(f"A: {nombre_a}", va, f"{ita} pasos")
    cb.metric(f"B: {nombre_b}", vb, f"{itb} pasos")

    if "faddeev" in (m_a, m_b):
        fad = metodo_faddeev_leverrier(A)
        curva = fad["curva_polinomio"]
        fig = crear_figura_base_plotly("Polinomio Característico P(λ) y Autovalores Estimados", "λ", "P(λ)")
        fig.add_hline(y=0, line_dash="dash", line_color=PALETA_ARANDU["terracota_oscuro"])
        fig.add_trace(
            go.Scatter(
                x=curva["lambda"],
                y=limpiar_valores_plotly(curva["p_lambda"], limite_abs=1e9),
                mode="lines",
                name="P(λ) Faddeev",
                line=dict(color=PALETA_ARANDU["oro"], width=2.5),
            )
        )
        if fad["autovalores_reales"]:
            fig.add_trace(
                go.Scatter(
                    x=fad["autovalores_reales"],
                    y=[0] * len(fad["autovalores_reales"]),
                    mode="markers",
                    name="Raíces exactas",
                    marker=dict(size=13, color=PALETA_ARANDU["marfil_nodo"], symbol="star", line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
                )
            )
        render_plotly_arandu(fig, key="plot_vs_aut_fad")
    else:
        fig = crear_figura_base_plotly("Trayectoria de Potencias λ_k", "Iteración", "λ_k")
        fig.add_trace(go.Scatter(x=list(range(1, len(sa) + 1)), y=limpiar_valores_plotly(sa), mode="lines+markers", name=nombre_a, line=dict(color=PALETA_ARANDU["salvia"], width=2.5)))
        fig.add_trace(go.Scatter(x=list(range(1, len(sb) + 1)), y=limpiar_valores_plotly(sb), mode="lines+markers", name=nombre_b, line=dict(color=PALETA_ARANDU["terracota"], width=2.5)))
        render_plotly_arandu(fig, key="plot_vs_aut_pot")


@st.fragment
def _duelo_interpolacion(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("## Arena en Vivo — Interpolación sobre Tabla Discreta")
    df_def = pd.DataFrame({"x": [0.2, 0.5, 0.8, 1.1, 1.4], "y": [0.258, 2.535, 4.902, 6.722, 10.704]})
    st.caption(r"Modifica cualquier nodo $x_i$ para comprobar qué ocurre con Newton-Gregory cuando se rompe la equidistancia:")
    df_ed = st.data_editor(df_def, use_container_width=True, hide_index=True, num_rows="dynamic", key="vs_int_tbl")
    df_c = df_ed.dropna()
    xs, ys = df_c["x"].astype(float).tolist(), df_c["y"].astype(float).tolist()
    x_eval = st.number_input("Punto $x^*$ a evaluar:", value=0.35, format="%.4f", key="vs_int_xe")

    diag = diagnosticar_espaciamiento_nodos(xs)
    if not diag["es_equiespaciado"]:
        st.warning(diag["mensaje"])

    def _correr(clave: str):
        if clave == "lagrange":
            r = interpolar_lagrange(xs, ys, x_eval)
            return r["y_interpolado"], r["duracion"], "Válido siempre", None
        if clave == "gregory_asc":
            r = interpolar_newton_gregory_ascendente(xs, ys, x_eval)
            return r["y_interpolado"], r["duracion"], f"u = {r['u']:.4f}", "ascendente"
        r = interpolar_newton_gregory_descendente(xs, ys, x_eval)
        return r["y_interpolado"], r["duracion"], f"v = {r['v']:.4f}", "descendente"

    ca, cb = st.columns(2)
    x_grid = np.linspace(min(xs), max(xs), 220)
    fig = crear_figura_base_plotly("Superposición de Polinomios Interpolantes", "x", "P(x)")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Nodos tabla",
            marker=dict(size=10, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )

    for col, clave, nombre, color in ((ca, m_a, nombre_a, PALETA_ARANDU["salvia"]), (cb, m_b, nombre_b, PALETA_ARANDU["terracota"])):
        try:
            val, _dur, nota, modo_g = _correr(clave)
            col.metric(nombre, f"{val:.6f}", nota)
            if modo_g:
                yg_raw = evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=len(xs) - 1, modo=modo_g)
            else:
                yg_raw = evaluar_curva_lagrange(xs, ys, x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid.tolist(),
                    y=limpiar_valores_plotly(yg_raw, limite_abs=1e9),
                    mode="lines",
                    name=nombre,
                    line=dict(color=color, width=2.4),
                )
            )
        except Exception as exc:
            col.error(f"{nombre}: {exc}")

    render_plotly_arandu(fig, key="plot_vs_interp")


@st.fragment
def _duelo_integracion(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown(r"## Arena en Vivo — Cuadratura Numérica con Paso $\Delta h$ Variable")
    x_base = [0.5, 5.5, 10.5, 15.5, 20.5, 25.5]
    y_base = [0.05, 6.05, 22.05, 48.05, 84.05, 130.05]
    c1, c2 = st.columns(2)
    f_str = c1.text_input("Función exacta $f(x)$:", value="x**2 / 5", key="vs_ing_f")
    n_sub = c2.slider("Cantidad de subintervalos $n$ (prueba $n$ impar vs par):", 2, 15, 5, key="vs_ing_n")

    xs, ys, _ = remuestrear_tabla_integracion(x_base, y_base, nuevo_n=n_sub, f_str=f_str.strip() or None)

    mapa_clave = {"trapecio": "trapecio", "simpson_1_3": "simpson_1_3", "simpson_3_8": "simpson_3_8"}
    res_a = integrar_adaptativo_en_vivo(xs, ys, metodo_solicitado=mapa_clave[m_a], f_str=f_str.strip() or None)
    res_b = integrar_adaptativo_en_vivo(xs, ys, metodo_solicitado=mapa_clave[m_b], f_str=f_str.strip() or None)

    ca, cb = st.columns(2)
    err_a = res_a["exacta"]["error_absoluto"] if res_a["exacta"] else float("inf")
    err_b = res_b["exacta"]["error_absoluto"] if res_b["exacta"] else float("inf")
    ca.metric(f"A: {nombre_a}", f"{res_a['area_total']:.6f}", f"{res_a['reglas_aplicadas']} | Err {err_a:.2e}{' (menor)' if err_a < err_b else ''}")
    cb.metric(f"B: {nombre_b}", f"{res_b['area_total']:.6f}", f"{res_b['reglas_aplicadas']} | Err {err_b:.2e}{' (menor)' if err_b < err_a else ''}")

    if res_a["hubo_adaptacion"] or res_b["hubo_adaptacion"]:
        st.info(" | ".join(res_a["avisos_pedagogicos"] + res_b["avisos_pedagogicos"]))

    fig = crear_figura_base_plotly(f"Comparación de Reglas de Integración (n = {n_sub})", "x", "f(x)")
    for b in res_a["bloques"]:
        fig.add_trace(
            go.Scatter(
                x=b["x_curva"],
                y=limpiar_valores_plotly(b["y_curva"], limite_abs=1e9),
                mode="lines",
                name=f"A: {b['regla']}",
                line=dict(color=PALETA_ARANDU["salvia"], width=2.4),
            )
        )
    for b in res_b["bloques"]:
        fig.add_trace(
            go.Scatter(
                x=b["x_curva"],
                y=limpiar_valores_plotly(b["y_curva"], limite_abs=1e9),
                mode="lines",
                name=f"B: {b['regla']}",
                line=dict(color=PALETA_ARANDU["terracota"], width=2.0, dash="dash"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Nodos",
            marker=dict(size=9, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8)),
        )
    )
    render_plotly_arandu(fig, key="plot_vs_integ")


@st.fragment
def _duelo_edo(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown(r"## Arena en Vivo — Trayectorias de EDO ($y' = f(x, y)$)")
    c1, c2 = st.columns(2)
    f_str = c1.text_input("EDO $y' = f(x, y)$:", value="6*x**2 / y", key="vs_edo_f")
    y_ex_str = c2.text_input("Exacta $y(x)$:", value="2*x*sqrt(x)", key="vs_edo_yex")
    c3, c4, c5, c6 = st.columns(4)
    x0 = c3.number_input("$x_0$:", value=4.0, format="%.4f", key="vs_edo_x0")
    y0 = c4.number_input("$y_0$:", value=16.0, format="%.4f", key="vs_edo_y0")
    xf = c5.number_input(r"$x_f$:", value=4.4, format="%.4f", key="vs_edo_xf")
    h = c6.number_input(r"Paso $\Delta h$:", value=0.1, min_value=0.01, step=0.05, format="%.4f", key="vs_edo_h")

    comp_f = compilar_funcion_edo(f_str)
    comp_ex = compilar_exacta_edo(y_ex_str) if y_ex_str.strip() else None
    y_ex_fn = comp_ex["fn"] if comp_ex else None

    runners = {
        "euler": lambda: metodo_euler(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn),
        "heun": lambda: metodo_euler_modificado(comp_f["fn"], x0, y0, h, xf=xf, iterar_corrector=True, y_exacta_fn=y_ex_fn),
        "rk2": lambda: metodo_runge_kutta_2(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn),
        "rk4": lambda: metodo_runge_kutta_4(comp_f["fn"], x0, y0, h, xf=xf, y_exacta_fn=y_ex_fn),
        "milne": lambda: metodo_milne(comp_f["fn"], x0, y0, h, xf=xf, n_correcciones=2, y_exacta_fn=y_ex_fn),
    }

    try:
        ra = runners[m_a]()
        rb = runners[m_b]()
    except Exception as exc:
        st.error(str(exc))
        return

    ea = ra["tabla"][-1].get("error_abs") or 0.0
    eb = rb["tabla"][-1].get("error_abs") or 0.0
    ca, cb = st.columns(2)
    ca.metric(f"A: {nombre_a}", f"{ra['y_final']:.6f}", f"Err final {ea:.2e}{' (menor)' if ea < eb else ''}")
    cb.metric(f"B: {nombre_b}", f"{rb['y_final']:.6f}", f"Err final {eb:.2e}{' (menor)' if eb < ea else ''}")

    fig = crear_figura_base_plotly("Superposición de Trayectorias PVI", "x", "y(x)")
    if comp_ex is not None:
        xg = np.linspace(x0, ra["xf_real"], 220)
        fn_np_ex = comp_ex.get("fn_np")
        yg = evaluar_fn_vectorizado(fn_np_ex if fn_np_ex is not None else np.vectorize(y_ex_fn), xg, limite_abs=1e6)
        fig.add_trace(
            go.Scatter(
                x=xg.tolist(),
                y=yg,
                mode="lines",
                name="Solución Exacta",
                line=dict(color=PALETA_ARANDU["oro"], width=2.8, dash="dash"),
            )
        )
    pares_a = muestrear_para_grafico(list(zip(ra["x"], ra["y"])), max_puntos=300)
    pares_b = muestrear_para_grafico(list(zip(rb["x"], rb["y"])), max_puntos=300)
    fig.add_trace(go.Scatter(x=[p[0] for p in pares_a], y=[p[1] for p in pares_a], mode="lines+markers", name=nombre_a, line=dict(color=PALETA_ARANDU["salvia"], width=2.4)))
    fig.add_trace(go.Scatter(x=[p[0] for p in pares_b], y=[p[1] for p in pares_b], mode="lines+markers", name=nombre_b, line=dict(color=PALETA_ARANDU["terracota"], width=2.4)))
    render_plotly_arandu(fig, key="plot_vs_edo")


def vista_arena_versus() -> None:
    inject_arandu_css()
    st.title("Arena Versus — Comparador Directo de Métodos")
    st.markdown(
        "Compara dos algoritmos únicamente dentro de su **misma familia matemática**, "
        "contrastando comportamiento esperado, casos límite y resultados en vivo."
    )

    familias = list(FICHAS_VERSUS.keys())
    fam_sel = st.selectbox(
        "1. Seleccioná la Familia Matemática (comparación estricta):",
        options=familias,
        format_func=lambda k: _sin_emojis(FICHAS_VERSUS[k]["titulo"]),
        key="vs_fam",
    )
    ficha_fam = FICHAS_VERSUS[fam_sel]
    st.info(ficha_fam["descripcion"])

    metodos_fam = list(ficha_fam["metodos"].keys())
    col_a, col_vs, col_b = st.columns([2, 0.4, 2])
    with col_a:
        m_a = st.selectbox(
            "Método A:",
            options=metodos_fam,
            index=0,
            format_func=lambda k: _sin_emojis(ficha_fam["metodos"][k]["nombre"]),
            key=f"vs_ma_{fam_sel}",
        )
    with col_vs:
        st.markdown('<p class="arandu-vs-divider" aria-hidden="true">VS</p>', unsafe_allow_html=True)
    with col_b:
        default_b = 1 if len(metodos_fam) > 1 else 0
        m_b = st.selectbox(
            "Método B:",
            options=metodos_fam,
            index=default_b,
            format_func=lambda k: _sin_emojis(ficha_fam["metodos"][k]["nombre"]),
            key=f"vs_mb_{fam_sel}",
        )

    ficha_a = ficha_fam["metodos"][m_a]
    ficha_b = ficha_fam["metodos"][m_b]

    cf_a, cf_b = st.columns(2)
    with cf_a:
        _render_ficha_tecnica("Contendiente A", ficha_a, "var(--arandu-gold)")
    with cf_b:
        _render_ficha_tecnica("Contendiente B", ficha_b, "var(--arandu-terracotta)")

    exp = obtener_expectativa_versus(fam_sel, m_a, m_b)
    ce1, ce2 = st.columns(2)
    with ce1:
        st.success(f"**¿Qué esperamos ver en ejecución?**\n\n{exp['que_esperamos']}")
    with ce2:
        st.warning(f"**Nota sobre Excepciones y Casos Límite**\n\n{exp['excepciones']}")

    st.divider()
    if m_a == m_b:
        st.warning("Elegiste el mismo método en ambos lados; selecciona dos métodos distintos de la familia para un duelo real.")

    if fam_sel == "raices":
        _duelo_raices(m_a, m_b, ficha_a["nombre"], ficha_b["nombre"])
    elif fam_sel == "sistemas":
        _duelo_sistemas(m_a, m_b, ficha_a["nombre"], ficha_b["nombre"])
    elif fam_sel == "autovalores":
        _duelo_autovalores(m_a, m_b, ficha_a["nombre"], ficha_b["nombre"])
    elif fam_sel == "interpolacion":
        _duelo_interpolacion(m_a, m_b, ficha_a["nombre"], ficha_b["nombre"])
    elif fam_sel == "integracion":
        _duelo_integracion(m_a, m_b, ficha_a["nombre"], ficha_b["nombre"])
    elif fam_sel == "edo":
        _duelo_edo(m_a, m_b, ficha_a["nombre"], ficha_b["nombre"])


vista_arena_versus()

