"""
Arena Versus - Arandú: Métodos Computacionales (v3.0).
Comparador directo estilo Versus.com restringido estrictamente a métodos de la misma familia matemática.
Muestra fichas técnicas, expectativas teóricas, advertencias sobre casos patológicos y duelo en vivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from views.ui_shared import crear_figura_base_plotly, inject_arandu_css


def _render_ficha_tecnica(titulo_slot: str, ficha: dict, color_borde: str) -> None:
    pros_html = "".join(f"<li>✅ {p}</li>" for p in ficha.get("pros", []))
    contras_html = "".join(f"<li>❌ {c}</li>" for c in ficha.get("contras", []))
    st.markdown(
        f"""
        <div class="arandu-versus-card" style="border-left: 5px solid {color_borde};">
            <h4 style="margin-top:0;">{titulo_slot}: {ficha['nombre']}</h4>
            <p style="margin:0.2rem 0;"><b>Orden:</b> {ficha['orden']}</p>
            <p style="margin:0.2rem 0;"><b>Costo:</b> {ficha['costo']}</p>
            <p style="margin:0.2rem 0;"><b>Estabilidad:</b> {ficha['estabilidad']}</p>
            <p style="margin:0.2rem 0;"><b>Requisitos:</b> {ficha['requisitos']}</p>
            <ul style="margin:0.45rem 0 0 0; padding-left:1.1rem;">{pros_html}{contras_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _duelo_raices(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("#### 📉 Arena en Vivo — Raíces sobre una misma Ecuación f(x) = 0")
    c1, c2 = st.columns(2)
    f_str = c1.text_input("Ecuación f(x) = 0:", value="x**3 - 2*x - 5", key="vs_raiz_f")
    g_str = c2.text_input("Despeje g(x) = x (si participa Aitken):", value="(2*x + 5)**(1/3)", key="vs_raiz_g")

    c3, c4, c5, c6 = st.columns(4)
    a_val = c3.number_input("a:", value=2.0, format="%.4f", key="vs_raiz_a")
    b_val = c4.number_input("b:", value=2.5, format="%.4f", key="vs_raiz_b")
    x0_val = c5.number_input("x0 (Newton/Aitken):", value=2.5, format="%.4f", key="vs_raiz_x0")
    tol_val = c6.number_input("Tolerancia E:", value=1e-4, min_value=1e-10, format="%.6f", key="vs_raiz_tol")

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
        ra, ita, ea, ta, err_a = _correr(m_a)
        rb, itb, eb, tb, err_b = _correr(m_b)
    except Exception as exc:
        st.error(f"Error ejecutando el duelo de raíces: {exc}")
        return

    ca, cb = st.columns(2)
    ca.metric(f"🔵 {nombre_a}", f"{ra:.7f}", f"{ita} pasos {'🏆' if ita < itb else ''} | Err {ea:.2e}")
    cb.metric(f"🟠 {nombre_b}", f"{rb:.7f}", f"{itb} pasos {'🏆' if itb < ita else ''} | Err {eb:.2e}")

    fig = crear_figura_base_plotly("Velocidad de Convergencia (Error Logarítmico por Paso)", "Iteración / Ciclo", "Error |Δx| (escala log)")
    if err_a:
        fig.add_trace(go.Scatter(x=list(range(1, len(err_a) + 1)), y=[max(e, 1e-16) for e in err_a], mode="lines+markers", name=nombre_a, line=dict(color="#2563eb", width=2.5)))
    if err_b:
        fig.add_trace(go.Scatter(x=list(range(1, len(err_b) + 1)), y=[max(e, 1e-16) for e in err_b], mode="lines+markers", name=nombre_b, line=dict(color="#f97316", width=2.5)))
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)


def _duelo_sistemas(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("#### 📐 Arena en Vivo — Sistemas Lineales Ax = b")
    default_mat = pd.DataFrame(
        [[18.72, 8.20, 8.76, 121.280], [6.40, 15.90, 7.18, 126.321], [5.22, 6.40, 14.31, 118.522]],
        columns=["a_1", "a_2", "a_3", "b"],
    )
    st.caption("✏️ Modifica la matriz A o el vector b para probar matrices con y sin dominancia diagonal:")
    df_ed = st.data_editor(default_mat, use_container_width=True, hide_index=True, key="vs_sis_mat")
    A = df_ed[["a_1", "a_2", "a_3"]].astype(float).values.tolist()
    b = df_ed["b"].astype(float).tolist()
    tol = st.number_input("Tolerancia (si participa Gauss-Seidel):", value=1e-4, format="%.6f", key="vs_sis_tol")

    dom = verificar_dominancia_diagonal(A)
    if not dom["es_dominante"] and ("gauss_seidel" in (m_a, m_b)):
        st.warning(f"⚠️ {dom['diagnostico']}")

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
        sa, na, ta, d_a = _correr(m_a)
        sb, nb, tb, d_b = _correr(m_b)
    except Exception as exc:
        st.error(str(exc))
        return

    ca, cb = st.columns(2)
    ca.metric(f"🔵 {nombre_a}", d_a, f"‖Ax-b‖ = {na:.2e} {'🏆' if na <= nb else ''}")
    cb.metric(f"🟠 {nombre_b}", d_b, f"‖Ax-b‖ = {nb:.2e} {'🏆' if nb < na else ''}")

    fig = crear_figura_base_plotly("Comparación de Soluciones xᵢ", "Incógnita", "Valor xᵢ", altura=340)
    etiquetas = [f"x_{i+1}" for i in range(len(sa))]
    fig.add_trace(go.Bar(x=etiquetas, y=sa, name=nombre_a, marker_color="#2563eb"))
    fig.add_trace(go.Bar(x=etiquetas, y=sb, name=nombre_b, marker_color="#f97316"))
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True)


def _duelo_autovalores(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("#### 🎯 Arena en Vivo — Autovalores de una Matriz Cuadrada")
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
            return f"λ_min = {r['autovalor_min']:.5f}", r["iteraciones"], [h["lambda_min"] for h in r["historial"]]
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
    ca.metric(f"🔵 {nombre_a}", va, f"{ita} pasos")
    cb.metric(f"🟠 {nombre_b}", vb, f"{itb} pasos")

    if "faddeev" in (m_a, m_b):
        fad = metodo_faddeev_leverrier(A)
        curva = fad["curva_polinomio"]
        fig = crear_figura_base_plotly("Polinomio Característico P(λ) y Autovalores Estimados", "λ", "P(λ)")
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig.add_trace(go.Scatter(x=curva["lambda"], y=curva["p_lambda"], mode="lines", name="P(λ) Faddeev", line=dict(color="#8b5cf6")))
        if fad["autovalores_reales"]:
            fig.add_trace(go.Scatter(x=fad["autovalores_reales"], y=[0] * len(fad["autovalores_reales"]), mode="markers", name="Raíces exactas", marker=dict(size=12, color="#10b981")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = crear_figura_base_plotly("Trayectoria de Potencias λ_k", "Iteración", "λ_k")
        fig.add_trace(go.Scatter(x=list(range(1, len(sa) + 1)), y=sa, mode="lines+markers", name=nombre_a, line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=list(range(1, len(sb) + 1)), y=sb, mode="lines+markers", name=nombre_b, line=dict(color="#f97316")))
        st.plotly_chart(fig, use_container_width=True)


def _duelo_interpolacion(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("#### 📈 Arena en Vivo — Interpolación sobre Tabla Discreta")
    df_def = pd.DataFrame({"x": [0.2, 0.5, 0.8, 1.1, 1.4], "y": [0.258, 2.535, 4.902, 6.722, 10.704]})
    st.caption("✏️ Modifica cualquier nodo x_i para comprobar qué pasa con Newton-Gregory cuando se rompe la equidistancia:")
    df_ed = st.data_editor(df_def, use_container_width=True, hide_index=True, num_rows="dynamic", key="vs_int_tbl")
    df_c = df_ed.dropna()
    xs, ys = df_c["x"].astype(float).tolist(), df_c["y"].astype(float).tolist()
    x_eval = st.number_input("Punto x* a evaluar:", value=0.35, format="%.4f", key="vs_int_xe")

    diag = diagnosticar_espaciamiento_nodos(xs)
    if not diag["es_equiespaciado"]:
        st.warning(f"⚠️ {diag['mensaje']}")

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
    x_grid = np.linspace(min(xs), max(xs), 140)
    fig = crear_figura_base_plotly("Superposición de Polinomios Interpolantes", "x", "P(x)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos tabla", marker=dict(size=9, color="#0f172a")))

    for col, clave, nombre, color in ((ca, m_a, nombre_a, "#2563eb"), (cb, m_b, nombre_b, "#f97316")):
        try:
            val, dur, nota, modo_g = _correr(clave)
            col.metric(nombre, f"{val:.6f}", nota)
            if modo_g:
                yg = evaluar_curva_gregory_parcial(xs, ys, x_grid, grado=len(xs) - 1, modo=modo_g)
            else:
                yg = np.array([interpolar_lagrange(xs, ys, float(xg))["y_interpolado"] for xg in x_grid])
            fig.add_trace(go.Scatter(x=x_grid, y=yg, mode="lines", name=nombre, line=dict(color=color, width=2.4)))
        except Exception as exc:
            col.error(f"{nombre}: {exc}")

    st.plotly_chart(fig, use_container_width=True)


def _duelo_integracion(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("#### 📊 Arena en Vivo — Cuadratura Numérica con Paso Δh Variable")
    x_base = [0.5, 5.5, 10.5, 15.5, 20.5, 25.5]
    y_base = [0.05, 6.05, 22.05, 48.05, 84.05, 130.05]
    c1, c2 = st.columns(2)
    f_str = c1.text_input("Función exacta f(x):", value="x**2 / 5", key="vs_ing_f")
    n_sub = c2.slider("Cantidad de subintervalos n (prueba n impar vs par):", 2, 15, 5, key="vs_ing_n")

    xs, ys, _ = remuestrear_tabla_integracion(x_base, y_base, nuevo_n=n_sub, f_str=f_str.strip() or None)

    mapa_clave = {"trapecio": "trapecio", "simpson_1_3": "simpson_1_3", "simpson_3_8": "simpson_3_8"}
    res_a = integrar_adaptativo_en_vivo(xs, ys, metodo_solicitado=mapa_clave[m_a], f_str=f_str.strip() or None)
    res_b = integrar_adaptativo_en_vivo(xs, ys, metodo_solicitado=mapa_clave[m_b], f_str=f_str.strip() or None)

    ca, cb = st.columns(2)
    err_a = res_a["exacta"]["error_absoluto"] if res_a["exacta"] else float("inf")
    err_b = res_b["exacta"]["error_absoluto"] if res_b["exacta"] else float("inf")
    ca.metric(f"🔵 {nombre_a}", f"{res_a['area_total']:.6f}", f"{res_a['reglas_aplicadas']} | Err {err_a:.2e} {'🏆' if err_a <= err_b else ''}")
    cb.metric(f"🟠 {nombre_b}", f"{res_b['area_total']:.6f}", f"{res_b['reglas_aplicadas']} | Err {err_b:.2e} {'🏆' if err_b < err_a else ''}")

    if res_a["hubo_adaptacion"] or res_b["hubo_adaptacion"]:
        st.info("ℹ️ " + " | ".join(res_a["avisos_pedagogicos"] + res_b["avisos_pedagogicos"]))

    fig = crear_figura_base_plotly(f"Comparación de Reglas de Integración (n = {n_sub})", "x", "f(x)")
    for b in res_a["bloques"]:
        fig.add_trace(go.Scatter(x=b["x_curva"], y=b["y_curva"], mode="lines", name=f"A: {b['regla']}", line=dict(color="#2563eb", width=2.4)))
    for b in res_b["bloques"]:
        fig.add_trace(go.Scatter(x=b["x_curva"], y=b["y_curva"], mode="lines", name=f"B: {b['regla']}", line=dict(color="#f97316", width=2.0, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodos", marker=dict(size=8, color="#0f172a")))
    st.plotly_chart(fig, use_container_width=True)


def _duelo_edo(m_a: str, m_b: str, nombre_a: str, nombre_b: str) -> None:
    st.markdown("#### ⏱️ Arena en Vivo — Trayectorias de EDO (y' = f(x, y))")
    c1, c2 = st.columns(2)
    f_str = c1.text_input("EDO y' = f(x, y):", value="6*x**2 / y", key="vs_edo_f")
    y_ex_str = c2.text_input("Exacta y(x):", value="2*x*sqrt(x)", key="vs_edo_yex")
    c3, c4, c5, c6 = st.columns(4)
    x0 = c3.number_input("x0:", value=4.0, format="%.4f", key="vs_edo_x0")
    y0 = c4.number_input("y0:", value=16.0, format="%.4f", key="vs_edo_y0")
    xf = c5.number_input("xf:", value=4.4, format="%.4f", key="vs_edo_xf")
    h = c6.number_input("Paso Δh:", value=0.1, min_value=0.01, step=0.05, format="%.4f", key="vs_edo_h")

    comp_f = compilar_funcion_edo(f_str)
    y_ex_fn = compilar_exacta_edo(y_ex_str)["fn"] if y_ex_str.strip() else None

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
    ca.metric(f"🔵 {nombre_a}", f"{ra['y_final']:.6f}", f"Err final {ea:.2e} {'🏆' if ea <= eb else ''}")
    cb.metric(f"🟠 {nombre_b}", f"{rb['y_final']:.6f}", f"Err final {eb:.2e} {'🏆' if eb < ea else ''}")

    fig = crear_figura_base_plotly("Superposición de Trayectorias PVI", "x", "y(x)")
    if y_ex_fn is not None:
        xg = np.linspace(x0, ra["xf_real"], 150)
        fig.add_trace(go.Scatter(x=xg.tolist(), y=[y_ex_fn(float(v)) for v in xg], mode="lines", name="Solución Exacta", line=dict(color="#10b981", width=2.8, dash="dash")))
    fig.add_trace(go.Scatter(x=ra["x"], y=ra["y"], mode="lines+markers", name=nombre_a, line=dict(color="#2563eb", width=2.4)))
    fig.add_trace(go.Scatter(x=rb["x"], y=rb["y"], mode="lines+markers", name=nombre_b, line=dict(color="#f97316", width=2.4)))
    st.plotly_chart(fig, use_container_width=True)


def vista_arena_versus() -> None:
    inject_arandu_css()
    st.title("⚔️ Arena Versus — Comparador Directo de Métodos")
    st.caption(
        "Compara dos algoritmos únicamente dentro de su **misma familia matemática**, "
        "contrastando teoría esperada, excepciones patológicas y resultados en vivo."
    )

    familias = list(FICHAS_VERSUS.keys())
    fam_sel = st.selectbox(
        "1️⃣ Selecciona la Familia Matemática (comparación estricta):",
        options=familias,
        format_func=lambda k: FICHAS_VERSUS[k]["titulo"],
        key="vs_fam",
    )
    ficha_fam = FICHAS_VERSUS[fam_sel]
    st.info(ficha_fam["descripcion"])

    metodos_fam = list(ficha_fam["metodos"].keys())
    col_a, col_vs, col_b = st.columns([2, 0.4, 2])
    with col_a:
        m_a = st.selectbox(
            "🔵 Método A:",
            options=metodos_fam,
            index=0,
            format_func=lambda k: ficha_fam["metodos"][k]["nombre"],
            key=f"vs_ma_{fam_sel}",
        )
    with col_vs:
        st.markdown("<h2 style='text-align:center; padding-top:1.1rem;'>VS</h2>", unsafe_allow_html=True)
    with col_b:
        default_b = 1 if len(metodos_fam) > 1 else 0
        m_b = st.selectbox(
            "🟠 Método B:",
            options=metodos_fam,
            index=default_b,
            format_func=lambda k: ficha_fam["metodos"][k]["nombre"],
            key=f"vs_mb_{fam_sel}",
        )

    ficha_a = ficha_fam["metodos"][m_a]
    ficha_b = ficha_fam["metodos"][m_b]

    cf_a, cf_b = st.columns(2)
    with cf_a:
        _render_ficha_tecnica("🔵 Contendiente A", ficha_a, "#2563eb")
    with cf_b:
        _render_ficha_tecnica("🟠 Contendiente B", ficha_b, "#f97316")

    exp = obtener_expectativa_versus(fam_sel, m_a, m_b)
    ce1, ce2 = st.columns(2)
    with ce1:
        st.success(f"**🧠 ¿Qué esperamos ver teóricamente?**\n\n{exp['que_esperamos']}")
    with ce2:
        st.warning(f"**⚠️ Nota sobre Excepciones y Casos Patológicos**\n\n{exp['excepciones']}")

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

