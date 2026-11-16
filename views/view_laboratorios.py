"""
Vista Histórica de Laboratorios (Labs 1 al 7) - Arandú.
Anexo curricular con experimentos introductorios del TP1 aislados con @st.fragment
y mapa completo hacia las vistas temáticas.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.fundamentos_errores import (
    generar_fibonacci,
    sumar_n_positivos,
    sumar_pares,
)
from core.version import APP_VERSION
from views.theme import PALETA_ARANDU
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    muestrear_para_grafico,
    registrar_contexto_ia,
    registrar_pagina_ia,
    render_dataframe_acotado,
    render_deduccion_paso_a_paso_viva,
    render_plotly_arandu,
    render_subtitulo_controles_vivo,
)


@st.fragment
def _frag_lab1_sumatorias() -> None:
    st.markdown(
        r"Compara la acumulación iterativa mediante un bucle `for` en memoria frente a la fórmula cerrada "
        r"$\sum_{i=1}^{N} i = \frac{N(N+1)}{2}$ (complejidad $O(N)$ vs. $O(1)$)."
    )
    render_subtitulo_controles_vivo()
    n_lim = st.slider(r"Cantidad límite $N$:", 5, 500, 100, key="lab1_n_sum")
    enteros = list(range(1, n_lim + 1))
    res_n = sumar_n_positivos([float(v) for v in enteros])
    res_p = sumar_pares(enteros)
    c1, c2 = st.columns(2)
    c1.metric(f"Suma 1..{n_lim}", f"{res_n['total']:.0f}", f"Cerrada N(N+1)/2 = {n_lim * (n_lim + 1) // 2}")
    c2.metric(f"Suma pares ≤ {n_lim}", str(res_p["suma_pares"]), f"Pares sumados: {res_p['cant_pares']}")
    registrar_contexto_ia(
        "Lab 1 — Sumatorias",
        f"- **Límite $N = {n_lim}$** | **Suma $1..N$:** `{res_n['total']:.0f}` | **Suma pares:** `{res_p['suma_pares']}` (`{res_p['cant_pares']}` pares)",
        clave="lab1_sum_res",
    )
    m_par = res_p["cant_pares"]
    render_deduccion_paso_a_paso_viva(
        f"Sumatorias Lab 1 (N={n_lim})",
        [
            "**Deducción de la suma cerrada de Gauss:** Sumando $S = 1 + 2 + \\dots + N$ con su orden inverso $S = N + (N-1) + \\dots + 1$, cada par vertical suma $(N + 1)$ un total de $N$ veces: $2S = N(N+1) \\implies S = \\frac{N(N+1)}{2}$.",
            f"**Sustitución para $N = {n_lim}$:** $\\frac{{{n_lim}\\cdot({n_lim} + 1)}}{{2}} = {n_lim * (n_lim + 1) // 2}$, idéntico al bucle acumulativo (`{res_n['total']:.0f}`).",
            f"**Deducción de suma de pares:** Los pares positivos hasta $N$ son $2(1 + 2 + \\dots + m)$ con $m = \\lfloor N/2 \\rfloor = {m_par}$, luego $S_{{\\text{{pares}}}} = 2 \\cdot \\frac{{m(m+1)}}{{2}} = m(m+1)$.",
            f"**Sustitución para $m = {m_par}$:** ${m_par} \\times ({m_par} + 1) = {res_p['suma_pares']}$.",
        ],
        clave_ia="lab1_sum_viva",
    )


@st.fragment
def _frag_lab1_fibonacci() -> None:
    st.markdown(
        r"Evalúa la recurrencia $F_k = F_{k-1} + F_{k-2}$ en aritmética entera exacta y verifica cómo el "
        r"cociente sucesivo $\frac{F_k}{F_{k-1}}$ converge a la razón áurea $\varphi = \frac{1 + \sqrt{5}}{2}$."
    )
    render_subtitulo_controles_vivo()
    n_fib = st.slider(r"Cantidad de términos $N$ (TP1 solicita $N = 200$):", 5, 200, 30, key="lab1_n_fib")
    res_f = generar_fibonacci(n_fib)
    hist_f = res_f["tabla"]
    razones = [h for h in hist_f if h.get("cociente") is not None]
    ultima_razon = razones[-1]["cociente"] if razones else None

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Término F_{n_fib}", f"{res_f['ultimo_termino']:.4e}" if n_fib > 25 else str(res_f["ultimo_termino"]))
    c2.metric("Razón F_n / F_{n-1}", f"{ultima_razon:.12f}" if ultima_razon else "-")
    c3.metric("φ = (1 + √5)/2 exacto", f"{res_f['phi_exacto']:.12f}")
    registrar_contexto_ia(
        "Lab 1 — Sucesión de Fibonacci y Razón Áurea",
        f"- **Términos $N = {n_fib}$** | **$F_N$:** `{res_f['ultimo_termino']}` | **Razón $F_n/F_{{n-1}}$:** `{ultima_razon}` | **$\\varphi$ exacto:** `{res_f['phi_exacto']:.12f}`",
        clave="lab1_fib_res",
    )
    err_phi_txt = (
        f"con cociente $F_{{{n_fib}}}/F_{{{n_fib - 1}}} = {ultima_razon:.12f}$ (distancia $|r - \\varphi| = {abs(ultima_razon - res_f['phi_exacto']):.3e}$)"
        if ultima_razon is not None
        else "sin cociente disponible"
    )
    render_deduccion_paso_a_paso_viva(
        f"Fibonacci y Razón Áurea φ (N={n_fib})",
        [
            "**Relación de recurrencia lineal:** Definimos $F_k = F_{k-1} + F_{k-2}$ partiendo de $F_1 = 0$, $F_2 = 1$ (generando $0, 1, 1, 2, 3, 5, 8, \\dots$).",
            "**Paso al límite para el cociente sucesivo:** Dividiendo la recurrencia por $F_{k-1}$ resulta $\\frac{F_k}{F_{k-1}} = 1 + \\frac{F_{k-2}}{F_{k-1}}$. Cuando $k \\to \\infty$, el límite $\\varphi$ satisface $\\varphi = 1 + \\frac{1}{\\varphi} \\implies \\varphi^2 - \\varphi - 1 = 0$.",
            f"**Resolución de la ecuación característica:** Aplicando la cuadrática con raíz positiva obtenemos $\\varphi = \\frac{{1 + \\sqrt{{5}}}}{{2}} \\approx {res_f['phi_exacto']:.12f}$.",
            f"**Verificación en $N = {n_fib}$:** El término exacto es $F_{{{n_fib}}} = {res_f['ultimo_termino']}$ ({res_f['cifras_ultimo']} cifras), {err_phi_txt}.",
        ],
        clave_ia="lab1_fib_viva",
    )

    render_dataframe_acotado(
        [
            {
                "orden": h["orden"],
                "valor_str": h["valor_str"],
                "cifras": h["cifras"],
                "cociente": h["cociente"],
                "distancia_phi": h["distancia_phi"],
            }
            for h in hist_f
        ],
        max_filas=300,
        key_prefix="lab1_fib_tbl",
    )

    if razones:
        muestra_r = muestrear_para_grafico(razones[:120], max_puntos=120)
        fig = crear_figura_base_plotly("Convergencia de F_k / F_{k-1} hacia φ", "Índice k", "Cociente", altura=340)
        fig.add_hline(y=res_f["phi_exacto"], line_dash="dash", line_color=PALETA_ARANDU["oro"], annotation_text="φ")
        fig.add_trace(
            go.Scatter(
                x=[r["orden"] for r in muestra_r],
                y=[r["cociente"] for r in muestra_r],
                mode="lines+markers",
                name="F_k / F_{k-1}",
                line=dict(color=PALETA_ARANDU["salvia"], width=2.4),
                marker=dict(size=7, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.4)),
            )
        )
        render_plotly_arandu(fig, key="plot_lab1_fib")


def vista_laboratorios() -> None:
    inject_arandu_css()
    registrar_pagina_ia("Histórico de Cursada — Laboratorios 1 al 7")
    st.title("Histórico de Cursada — Laboratorios 1 al 7")
    st.caption(
        "Anexo curricular de Métodos Computacionales (FaCENA - UNNE). "
        "Incluye los experimentos iterativos del Lab 1 y accesos directos a todos los TPs."
    )

    with st.expander(f"Mapa Curricular: Laboratorios → Módulos Arandú v{APP_VERSION}", expanded=True):
        mapa = [
            ("Lab 1 y 2", r"Algoritmos iterativos, Fibonacci, series ($e^x$, $\pi$), error de máquina ($k$ dígitos), cancelación y Horner", "views/view_fundamentos.py"),
            ("Lab 3", r"Descartes, Tanteos, Bisección, Regula Falsi, Newton-Raphson y Punto Fijo + $\Delta^2$ de Aitken", "views/view_raices.py"),
            ("Lab 4", r"Gauss, Gauss-Jordan, factorización $LU$ (Doolittle), Gauss-Seidel, Potencias y Faddeev-Leverrier", "views/view_sistemas.py"),
            ("Lab 5", "Interpolación Lineal, Cuadrática Inversa, Lagrange y Newton-Gregory Ascendente/Descendente", "views/view_interpolacion.py"),
            ("Lab 6", r"Derivación con Gregory + Trapecio, Simpson $1/3$, Simpson $3/8$ y combinaciones por paso $\Delta h$", "views/view_integracion.py"),
            ("Lab 7", "PVI: Euler, Heun (Euler Modificado), RK2, RK4 y Predictor-Corrector de Milne", "views/view_edo.py"),
        ]
        for lab, desc, ruta in mapa:
            c1, c2, c3 = st.columns([1.0, 3.3, 1.2])
            c1.markdown(f"**{lab}**")
            c2.markdown(desc)
            if c3.button("Abrir módulo", key=f"labmap_{lab}", use_container_width=True):
                st.switch_page(ruta)

    st.divider()
    st.subheader("Experimentos Clásicos del Laboratorio 1")

    t_sum, t_fib = st.tabs(["Sumatorias (N y Pares)", "Sucesión de Fibonacci y Razón Áurea φ"])
    with t_sum:
        _frag_lab1_sumatorias()

    with t_fib:
        _frag_lab1_fibonacci()


vista_laboratorios()
