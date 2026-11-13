"""
Vista de Fundamentos y Teoría de Errores - Arandú v3.0.
Métodos: Series de Taylor (e^x, pi), Aritmética de Máquina & Asociatividad,
Cancelación Catastrófica en Cuadráticas y Esquema Anidado de Horner.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.fundamentos_errores import (
    analizar_cancelacion_cuadratica,
    calcular_pi_leibniz_historial,
    calcular_serie_euler,
    comparar_asociatividad_personalizada,
    evaluar_polinomio_horner,
    operar_en_maquina,
)
from views import (
    crear_figura_base_plotly,
    inject_arandu_css,
    render_banner_edicion_profesora,
    render_bloque_1_teoria_y_ejemplo,
)


def _tab_series_taylor() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("series_taylor")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"x_euler": 1.0, "n_terminos_euler": 8, "cifras_scarborough": 3, "cifras_leibniz": 3},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="fund_taylor",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
        mensaje_edicion="✏️ Parámetros Interactivos: Ajusta x, cantidad de términos N o cifras significativas en vivo.",
    )

    solo_lectura = modo == "catedra"
    c1, c2, c3 = st.columns(3)
    with c1:
        x_val = st.number_input(
            "Valor x para e^x:",
            value=float(st.session_state["fund_taylor_x_euler"]),
            step=0.25,
            format="%.4f",
            disabled=solo_lectura,
            key="ui_taylor_x",
        )
    with c2:
        n_max = st.slider(
            "Términos máximos N (serie e^x):",
            min_value=2,
            max_value=25,
            value=int(st.session_state["fund_taylor_n_terminos_euler"]),
            disabled=solo_lectura,
            key="ui_taylor_n",
        )
    with c3:
        cifras_s = st.slider(
            "Cifras significativas (Scarborough):",
            min_value=1,
            max_value=8,
            value=int(st.session_state["fund_taylor_cifras_scarborough"]),
            disabled=solo_lectura,
            key="ui_taylor_scar",
        )

    if not solo_lectura:
        st.session_state["fund_taylor_x_euler"] = x_val
        st.session_state["fund_taylor_n_terminos_euler"] = n_max
        st.session_state["fund_taylor_cifras_scarborough"] = cifras_s

    res = calcular_serie_euler(n_max=n_max, x_val=x_val, cifras_scarborough=cifras_s)

    st.markdown("### ⚙️ Bloque 3 — Resultados en Vivo (Serie de Maclaurin vs Cota de Scarborough)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Aproximación e^x", f"{res['aproximacion_float']:.8f}")
    m2.metric("Valor exacto math.exp", f"{res['valor_exacto']:.8f}")
    m3.metric("Error absoluto Ea", f"{res['error_absoluto_final']:.3e}")
    m4.metric("Cota Scarborough Es (%)", f"{res['es_scarborough_pct']:.4g}%")

    df_euler = pd.DataFrame(res["historial"])
    st.dataframe(df_euler, use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Convergencia Geométrica Paso a Paso")
    k_vis = st.slider(
        "Ver hasta la iteración / término k:",
        min_value=1,
        max_value=len(df_euler),
        value=len(df_euler),
        key="vis_taylor_k",
    )
    df_vis = df_euler.iloc[:k_vis]

    fig = crear_figura_base_plotly(
        f"Convergencia de e^({x_val:g}) hasta k = {k_vis} términos",
        x_label="Índice del término (k)",
        y_label="Suma parcial",
    )
    fig.add_hline(y=res["valor_exacto"], line_dash="dash", line_color="#10b981", annotation_text="Exacto")
    fig.add_trace(
        go.Scatter(
            x=df_vis["k"],
            y=df_vis["suma_parcial"],
            mode="lines+markers",
            name="Suma parcial e^x",
            line=dict(color="#2563eb", width=2.5),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🥧 Ver además Serie de Gregory-Leibniz para π (Convergencia lenta)"):
        pasos_pi = st.slider("Cantidad de términos para π:", 10, 500, 100, step=10, key="ui_pi_pasos")
        res_pi = calcular_pi_leibniz_historial(n_terminos=pasos_pi)
        st.metric("π aproximado", f"{res_pi['pi_aproximado']:.6f}", f"Ea = {res_pi['error_absoluto']:.4e}")
        df_pi = pd.DataFrame(res_pi["historial"])
        fig_pi = crear_figura_base_plotly("Oscilación amortiguada de Gregory-Leibniz", "Términos", "Aprox π", altura=320)
        fig_pi.add_hline(y=res_pi["pi_exacto"], line_dash="dash", line_color="#10b981")
        fig_pi.add_trace(go.Scatter(x=df_pi["k"], y=df_pi["pi_aprox"], mode="lines+markers", name="π Leibniz"))
        st.plotly_chart(fig_pi, use_container_width=True)


def _tab_aritmetica_maquina() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("aritmetica_maquina")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"k_digitos": 3, "u_val": 121.0, "v_val": 119.0, "w_val": 0.327},
    )
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="fund_asoc",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
        mensaje_edicion="✏️ Tabla/Parámetros Interactivos: Modifica u, v, w o la mantisa de k dígitos en vivo.",
    )
    solo_lectura = modo == "catedra"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k_dig = st.slider("Mantisa k dígitos:", 2, 7, int(st.session_state["fund_asoc_k_digitos"]), disabled=solo_lectura, key="ui_asoc_k")
    with c2:
        u_val = st.number_input("u:", value=float(st.session_state["fund_asoc_u_val"]), format="%.4f", disabled=solo_lectura, key="ui_asoc_u")
    with c3:
        v_val = st.number_input("v:", value=float(st.session_state["fund_asoc_v_val"]), format="%.4f", disabled=solo_lectura, key="ui_asoc_v")
    with c4:
        w_val = st.number_input("w:", value=float(st.session_state["fund_asoc_w_val"]), format="%.4f", disabled=solo_lectura, key="ui_asoc_w")

    if not solo_lectura:
        st.session_state["fund_asoc_k_digitos"] = k_dig
        st.session_state["fund_asoc_u_val"] = u_val
        st.session_state["fund_asoc_v_val"] = v_val
        st.session_state["fund_asoc_w_val"] = w_val

    res_asoc = comparar_asociatividad_personalizada(u=u_val, v=v_val, w=w_val, k=k_dig)
    st.markdown("### ⚙️ Bloque 3 — Impacto de la Pérdida de Asociatividad")
    st.metric("Valor Exacto (u - v) - w", f"{res_asoc['exacto']:.8g}")
    df_asoc = pd.DataFrame(res_asoc["tabla"])
    st.dataframe(df_asoc, use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Comparación Visual del Error Relativo (%)")
    k_vis = st.slider("Ver hasta el caso k:", 1, len(df_asoc), len(df_asoc), key="vis_asoc_k")
    df_vis = df_asoc.iloc[:k_vis]
    fig = crear_figura_base_plotly("Error Relativo (%) según Orden de Asociación", "Esquema", "Error Relativo (%)", altura=360)
    fig.add_trace(
        go.Bar(
            x=df_vis["agrupacion"],
            y=df_vis["error_redondeo_pct"],
            name="Redondeo",
            marker_color=["#10b981" if e < 2 else "#ef4444" for e in df_vis["error_redondeo_pct"]],
            text=[f"{e:.2f}%" for e in df_vis["error_redondeo_pct"]],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_vis["agrupacion"],
            y=df_vis["error_trunc_pct"],
            name="Truncamiento",
            marker_color="#f59e0b",
            text=[f"{e:.2f}%" for e in df_vis["error_trunc_pct"]],
            textposition="auto",
        )
    )
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧮 Calculadora Elemental fl(x op y) a k dígitos"):
        cx, cop, cy = st.columns([2, 1, 2])
        ox = cx.number_input("x:", value=1.0 / 3.0, format="%.6f", key="op_x")
        oper = cop.selectbox("Operación:", ["+", "-", "*", "/"], key="op_sym")
        oy = cy.number_input("y:", value=5.0 / 7.0, format="%.6f", key="op_y")
        res_op = operar_en_maquina(ox, oy, oper, k=k_dig)
        st.json(res_op)


def _tab_cancelacion_cuadratica() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("cancelacion_cuadratica")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get("datos_iniciales", {"a": 1.0, "b": 62.10, "c": 1.0, "k_digitos": 4})
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="fund_canc",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
        mensaje_edicion="✏️ Coeficientes Interactivos: Modifica a, b, c o dígitos k para provocar o evitar cancelación catastrófica.",
    )
    solo_lectura = modo == "catedra"

    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("a:", value=float(st.session_state["fund_canc_a"]), format="%.4f", disabled=solo_lectura, key="ui_canc_a")
    b_val = c2.number_input("b:", value=float(st.session_state["fund_canc_b"]), format="%.4f", disabled=solo_lectura, key="ui_canc_b")
    c_val = c3.number_input("c:", value=float(st.session_state["fund_canc_c"]), format="%.4f", disabled=solo_lectura, key="ui_canc_c")
    k_dig = c4.slider("Dígitos k:", 2, 8, int(st.session_state["fund_canc_k_digitos"]), disabled=solo_lectura, key="ui_canc_k")

    if not solo_lectura:
        st.session_state["fund_canc_a"] = a_val
        st.session_state["fund_canc_b"] = b_val
        st.session_state["fund_canc_c"] = c_val
        st.session_state["fund_canc_k_digitos"] = k_dig

    try:
        res = analizar_cancelacion_cuadratica(a=a_val, b=b_val, c=c_val, k=k_dig)
    except Exception as exc:
        st.error(str(exc))
        return

    st.markdown("### ⚙️ Bloque 3 — Resolvente Clásica vs Racionalizada")
    df_c = pd.DataFrame(res["tabla_comparativa"])
    st.dataframe(df_c, use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Comparación de Pérdida de Precisión")
    paso_k = st.slider("Comparar hasta k variantes:", 1, len(df_c), len(df_c), key="vis_canc_k")
    df_vis = df_c.iloc[:paso_k]
    fig = crear_figura_base_plotly(f"Error Relativo (%) en Raíces (k = {k_dig})", "Variante", "Error Relativo (%)", altura=350)
    fig.add_trace(
        go.Bar(
            x=df_vis["metodo"],
            y=df_vis["error_relativo_pct"],
            marker_color=["#ef4444" if flag else "#10b981" for flag in df_vis["sufre_cancelacion"]],
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_horner() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("horner")
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    coefs_def = [float(c.strip()) for c in str(raw_def.get("coefs_str", "1, -6, 3, -0.149")).split(",") if c.strip()]
    defaults = {
        "x_eval": float(raw_def.get("x_eval", 4.71)),
        "k_digitos": int(raw_def.get("k_digitos", 3)),
        "coefs": coefs_def or [1.0, -6.0, 3.0, -0.149],
    }
    st.divider()

    modo = render_banner_edicion_profesora(
        clave_state="fund_horner",
        defaults_dict=defaults,
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
        mensaje_edicion="✏️ Tabla de Coeficientes Interactiva: Haz doble clic en la tabla de coeficientes [a_n ... a_0] para editar el polinomio.",
    )
    solo_lectura = modo == "catedra"

    c1, c2 = st.columns(2)
    x_val = c1.number_input("Punto x:", value=float(st.session_state["fund_horner_x_eval"]), format="%.4f", disabled=solo_lectura, key="ui_horner_x")
    k_dig = c2.slider("Mantisa k dígitos:", 2, 8, int(st.session_state["fund_horner_k_digitos"]), disabled=solo_lectura, key="ui_horner_k")

    coefs_curr = st.session_state["fund_horner_coefs"]
    df_in = pd.DataFrame(
        {"Potencia": [f"x^{len(coefs_curr)-1-i}" for i in range(len(coefs_curr))], "Coeficiente": coefs_curr}
    )
    df_edit = st.data_editor(
        df_in,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic" if not solo_lectura else "fixed",
        disabled=solo_lectura,
        key="fund_horner__editor_df",
    )

    if not solo_lectura:
        st.session_state["fund_horner_x_eval"] = x_val
        st.session_state["fund_horner_k_digitos"] = k_dig
        nuevos_coefs = [float(v) for v in df_edit["Coeficiente"].dropna().tolist()]
        if nuevos_coefs:
            st.session_state["fund_horner_coefs"] = nuevos_coefs
            coefs_curr = nuevos_coefs

    res = evaluar_polinomio_horner(x_val=x_val, k=k_dig, coefs=coefs_curr)
    tabla_err = res["tabla_errores"]
    st.markdown("### ⚙️ Bloque 3 — Evaluación Directa vs Anidada de Horner")
    m1, m2, m3 = st.columns(3)
    m1.metric("Valor Exacto", f"{res['f_exacto']:.6f}")
    m2.metric("Estándar (Redondeo)", f"{tabla_err[1]['aproximacion']:.6g}", f"Er {tabla_err[1]['error_relativo_pct']:.2f}%")
    m3.metric("Horner (Redondeo)", f"{tabla_err[3]['aproximacion']:.6g}", f"Er {tabla_err[3]['error_relativo_pct']:.2f}%")
    st.dataframe(pd.DataFrame(tabla_err), use_container_width=True, hide_index=True)

    st.markdown("### 📐 Bloque 4 — Comparación Paso a Paso de Errores por Esquema")
    df_h = pd.DataFrame(tabla_err)
    k_vis = st.slider("Ver hasta el esquema k:", 1, len(df_h), len(df_h), key="vis_horner_k")
    fig = crear_figura_base_plotly("Error Relativo (%): Estándar vs Horner", "Esquema", "Error Rel (%)", altura=350)
    df_vis = df_h.iloc[:k_vis]
    fig.add_trace(go.Bar(x=df_vis["metodo"], y=df_vis["error_relativo_pct"], marker_color=["#f59e0b" if "Estándar" in s else "#10b981" for s in df_vis["metodo"]]))
    st.plotly_chart(fig, use_container_width=True)


def vista_fundamentos() -> None:
    inject_arandu_css()
    st.title("🧮 Fundamentos, Series y Teoría de Errores")
    st.caption("Estabilidad numérica, representación de máquina y propagación del error.")

    tabs = st.tabs(
        [
            "1️⃣ Series de Taylor (e^x y π)",
            "2️⃣ Aritmética de Máquina & Asociatividad",
            "3️⃣ Cancelación Cuadrática",
            "4️⃣ Forma Anidada de Horner",
        ]
    )
    with tabs[0]:
        _tab_series_taylor()
    with tabs[1]:
        _tab_aritmetica_maquina()
    with tabs[2]:
        _tab_cancelacion_cuadratica()
    with tabs[3]:
        _tab_horner()


vista_fundamentos()
