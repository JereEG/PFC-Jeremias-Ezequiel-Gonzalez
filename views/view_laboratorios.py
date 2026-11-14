"""
Vista Histórica de Laboratorios (Labs 1 al 7) - Arandú v3.0.
Anexo curricular con experimentos introductorios del TP1 y mapa completo hacia las vistas temáticas.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.fundamentos_errores import (
    generar_fibonacci,
    sumar_n_positivos,
    sumar_pares,
)
from views.ui_shared import crear_figura_base_plotly, inject_arandu_css


def vista_laboratorios() -> None:
    inject_arandu_css()
    st.title("📚 Histórico de Cursada — Laboratorios 1 al 7")
    st.caption(
        "Anexo curricular de la cátedra Métodos Computacionales (FaCENA - UNNE). "
        "Incluye los experimentos iniciales de algoritmos iterativos del Lab 1 y accesos directos al resto de los TPs."
    )

    with st.expander("🧭 Mapa Curricular: Laboratorios → Módulos Arandú v3.0", expanded=True):
        mapa = [
            ("Lab 1 y 2", "Algoritmos, Fibonacci, Series (e^x, π), Error, Máquina k dígitos, Cancelación y Horner", "views/view_fundamentos.py"),
            ("Lab 3", "Descartes, Tanteos, Bisección, Regula Falsi, Newton-Raphson (Fourier) y Punto Fijo + Aitken", "views/view_raices.py"),
            ("Lab 4", "Gauss, Gauss-Jordan, LU Doolittle, Gauss-Seidel, Potencias y Faddeev-Leverrier", "views/view_sistemas.py"),
            ("Lab 5", "Interpolación Lineal, Cuadrática Inversa, Lagrange y Newton-Gregory Asc/Desc", "views/view_interpolacion.py"),
            ("Lab 6", "Derivación con Gregory + Trapecio, Simpson 1/3, Simpson 3/8 y Combinaciones", "views/view_integracion.py"),
            ("Lab 7", "PVI: Euler, Heun (Euler Modificado), RK2, RK4 y Predictor-Corrector de Milne", "views/view_edo.py"),
        ]
        for lab, desc, ruta in mapa:
            c1, c2, c3 = st.columns([1.0, 3.3, 1.2])
            c1.markdown(f"**{lab}**")
            c2.write(desc)
            if c3.button("Abrir módulo", key=f"labmap_{lab}", use_container_width=True):
                st.switch_page(ruta)

    st.divider()
    st.subheader("🧪 Experimentos Clásicos del Laboratorio 1")

    t_sum, t_fib = st.tabs(["Sumatorias (N y Pares)", "Sucesión de Fibonacci y Razón Áurea φ"])
    with t_sum:
        n_lim = st.slider("Valor N:", 5, 500, 100, key="lab1_n_sum")
        enteros = list(range(1, n_lim + 1))
        res_n = sumar_n_positivos([float(v) for v in enteros])
        res_p = sumar_pares(enteros)
        c1, c2 = st.columns(2)
        c1.metric(f"Suma 1..{n_lim}", f"{res_n['total']:.0f}", f"Fórmula N(N+1)/2 = {n_lim*(n_lim+1)//2}")
        c2.metric(f"Suma pares ≤ {n_lim}", str(res_p["suma_pares"]), f"Pares sumados: {res_p['cant_pares']}")

    with t_fib:
        n_fib = st.slider("Cantidad de términos Fibonacci (TP1 pide N=200):", 5, 200, 30, key="lab1_n_fib")
        res_f = generar_fibonacci(n_fib)
        hist_f = res_f["tabla"]
        razones = [h for h in hist_f if h.get("cociente") is not None]
        ultima_razon = razones[-1]["cociente"] if razones else None

        c1, c2, c3 = st.columns(3)
        c1.metric(f"F_{n_fib}", f"{res_f['ultimo_termino']:.4e}" if n_fib > 25 else str(res_f["ultimo_termino"]))
        c2.metric("Razón F_n / F_{n-1}", f"{ultima_razon:.12f}" if ultima_razon else "-")
        c3.metric("φ = (1+√5)/2 exacto", f"{res_f['phi_exacto']:.12f}")

        df_fib = pd.DataFrame(hist_f[-20:])[["orden", "valor_str", "cifras", "cociente", "distancia_phi"]]
        st.dataframe(df_fib, use_container_width=True, hide_index=True)

        if razones:
            fig = crear_figura_base_plotly("Convergencia de F_k / F_{k-1} hacia φ", "Índice k", "Cociente", altura=340)
            fig.add_hline(y=res_f["phi_exacto"], line_dash="dash", line_color="#D4A359", annotation_text="φ")
            fig.add_trace(
                go.Scatter(
                    x=[r["orden"] for r in razones[:40]],
                    y=[r["cociente"] for r in razones[:40]],
                    mode="lines+markers",
                    name="F_k / F_{k-1}",
                    line=dict(color="#7A9A65", width=2.4),
                    marker=dict(size=7, color="#FFFDF9", line=dict(color="#B5673F", width=1.4)),
                )
            )
            st.plotly_chart(fig, use_container_width=True)


vista_laboratorios()
