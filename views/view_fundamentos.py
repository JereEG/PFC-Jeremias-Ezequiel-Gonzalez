"""
Vista de Fundamentos y Teoría de Errores - Arandú.
Métodos: Series iterativas (e^x, pi), Aritmética de Máquina & Asociatividad,
Cancelación Catastrófica en Cuadráticas y Esquema Anidado de Horner.
Aislamiento Static-First mediante @st.fragment en cada laboratorio interactivo.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.fundamentos_errores import (
    analizar_cancelacion_cuadratica,
    analizar_ffe_propagacion,
    calcular_epsilon_maquina,
    calcular_pi_leibniz_historial,
    calcular_serie_euler,
    comparar_asociatividad_personalizada,
    evaluar_polinomio_horner,
    obtener_catalogo_escenarios_ffe,
    operar_en_maquina,
)
from views.theme import PALETA_ARANDU
from views.ui_shared import (
    crear_figura_base_plotly,
    inject_arandu_css,
    muestrear_para_grafico,
    registrar_contexto_ia,
    render_bloque_1_teoria_y_ejemplo,
    render_controles_edicion_practica,
    render_data_editor_arandu,
    render_dataframe_acotado,
    render_deduccion_paso_a_paso_viva,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
)


@st.fragment
def _frag_series_taylor(defaults: dict) -> None:
    st.markdown(
        "**Por qué existen en computación:** Una CPU solo sabe ejecutar operaciones aritméticas básicas "
        "(`+`, `-`, `*`, `/`). Series infinitas como $e^x$, $\\ln(1+x)$ o $\\pi$ traducen funciones trascendentes "
        "a un bucle `for` que acumula potencias y factoriales hasta satisfacer la tolerancia pedida."
    )

    serie_sel = st.radio(
        "Selecciona la aproximación iterativa a simular:",
        options=[
            r"Exponencial: $e^x = \sum_{k=0}^{N} \frac{x^k}{k!}$ (Cota de parada de Scarborough)",
            r"Constante geométrica: $\pi = 4\sum_{k=0}^{N} \frac{(-1)^k}{2k+1}$ (Gregory-Leibniz alternante)",
        ],
        horizontal=True,
        key="ui_taylor_selector",
    )

    modo = render_controles_edicion_practica(
        clave_state="fund_taylor",
        defaults_dict=defaults,
        mensaje_edicion=r"Ajusta $x$, la cantidad de términos $N$ o las cifras significativas en vivo.",
    )
    solo_lectura = modo == "catedra"

    if serie_sel.startswith("Exponencial"):
        c1, c2, c3 = st.columns(3)
        with c1:
            x_val = st.number_input(
                "Valor $x$ para $e^x$:",
                value=float(st.session_state["fund_taylor_x_euler"]),
                step=0.25,
                format="%.4f",
                disabled=solo_lectura,
                key="ui_taylor_x",
            )
        with c2:
            n_max = st.slider(
                "Términos máximos $N$ (serie $e^x$):",
                min_value=2,
                max_value=25,
                value=int(st.session_state["fund_taylor_n_terminos_euler"]),
                disabled=solo_lectura,
                key="ui_taylor_n",
            )
        with c3:
            cifras_s = st.slider(
                "Cifras significativas $n$ (condición de corte de Scarborough):",
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

        st.markdown("### Bloque 3 — Resultados en Vivo (Bucle Maclaurin vs Cota de Corte)")
        st.caption(
            r"Criterio de parada ($E_{rp} < E_s = 0.5 \times 10^{2-n}\%$, Scarborough): detiene el bucle cuando "
            r"agregar un término ya no cambia las $n$ cifras significativas buscadas."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Aproximación eˣ", f"{res['aproximacion_float']:.8f}")
        m2.metric("Valor exacto exp(x)", f"{res['valor_exacto']:.8f}")
        m3.metric("Error absoluto Eₐ", f"{res['error_absoluto_final']:.3e}")
        m4.metric("Cota Scarborough Eₛ (%)", f"{res['es_scarborough_pct']:.4g}%")
        registrar_contexto_ia(
            "Resultados Serie de Maclaurin e^x",
            f"- **$x = {x_val:g}$ | Términos máx:** `{n_max}` | **Cifras:** `{cifras_s}`\n"
            f"- **Aproximación:** `{res['aproximacion_float']:.8f}` | **Exacto:** `{res['valor_exacto']:.8f}` | **Error abs.:** `{res['error_absoluto_final']:.3e}`",
            clave="fund_taylor_res",
        )

        hist = res["historial"]
        pasos_taylor_vivos = [
            (
                f"**Configuración y cota de parada (Scarborough):** Evaluamos $e^{{{x_val:g}}} = \\sum_{{k=0}}^{{{n_max}}} \\frac{{({x_val:g})^k}}{{k!}}$. "
                f"Para garantizar $n = {cifras_s}$ cifras significativas, exigimos $E_{{rp}} < E_s = 0.5 \\times 10^{{2-{cifras_s}}}\\% = {res['es_scarborough_pct']:.4g}\\%$."
            )
        ]
        for r_h in hist[:6]:
            erp_txt = f", $E_{{rp}} = {r_h['error_aprox_rp_pct']:.4g}\\%$" if r_h["error_aprox_rp_pct"] is not None else ""
            pasos_taylor_vivos.append(
                f"**Término $k = {r_h['k']}$:** $T_{{{r_h['k']}}} = {r_h['expresion']} = {r_h['termino']:.8g} \\implies "
                f"S_{{{r_h['k']}}} = {r_h['suma_parcial']:.8f}$ ($E_a = {r_h['error_absoluto']:.3e}${erp_txt})."
            )
        if len(hist) > 6:
            pasos_taylor_vivos.append(
                f"**Cierre en $k = {n_max}$:** La acumulación completa $N = {n_max}$ alcanza $S_{{{n_max}}} = {res['aproximacion_float']:.8f}$ "
                f"frente a $\\exp({x_val:g}) = {res['valor_exacto']:.8f}$ ($E_a = {res['error_absoluto_final']:.3e}$)."
            )
        if res.get("termino_scarborough") is not None:
            pasos_taylor_vivos.append(
                f"**Verificación de corte:** El criterio de Scarborough ($|E_{{rp}}| < {res['es_scarborough_pct']:.4g}\\%$) se cumple por primera vez en el término $k = {res['termino_scarborough']}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"Serie Maclaurin e^x (x={x_val:g}, N={n_max})",
            pasos_taylor_vivos,
            clave_ia="fund_taylor_ex",
        )
        render_dataframe_acotado(hist, max_filas=300, key_prefix="tbl_fund_taylor")

        st.markdown("### Bloque 4 — Convergencia Paso a Paso")
        k_vis = render_slider_paso_k("Ver hasta la iteración / término $k$:", len(hist), key="vis_taylor_k")
        sub_hist = hist[:k_vis]

        fig = crear_figura_base_plotly(
            f"Convergencia de e^({x_val:g}) hasta k = {k_vis} términos",
            x_label="Índice del término (k)",
            y_label="Suma parcial",
        )
        fig.add_hline(y=res["valor_exacto"], line_dash="dash", line_color=PALETA_ARANDU["oro"], annotation_text="Exacto")
        fig.add_trace(
            go.Scatter(
                x=[r["k"] for r in sub_hist],
                y=[r["suma_parcial"] for r in sub_hist],
                mode="lines+markers",
                name="Suma parcial eˣ",
                line=dict(color=PALETA_ARANDU["salvia"], width=2.5),
                marker=dict(size=7, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.5)),
            )
        )
        render_plotly_arandu(fig, key="plot_fund_taylor")
    else:
        st.latex(r"\pi \approx 4 \sum_{k=0}^{N-1} \frac{(-1)^k}{2k+1} = 4\left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \dots\right)")
        pasos_pi = st.slider(r"Cantidad de términos $N$ para $\pi$:", 10, 500, 100, step=10, key="ui_pi_pasos")
        res_pi = calcular_pi_leibniz_historial(n_terminos=pasos_pi)
        m1, m2, m3 = st.columns(3)
        m1.metric("π aproximado", f"{res_pi['pi_aproximado']:.6f}")
        m2.metric("π exacto", f"{res_pi['pi_exacto']:.6f}")
        m3.metric("Error absoluto Eₐ", f"{res_pi['error_absoluto']:.4e}")

        hist_pi = res_pi["historial"]
        cota_leib = 4.0 / (2 * pasos_pi + 1)
        render_deduccion_paso_a_paso_viva(
            f"Serie Gregory-Leibniz π (N={pasos_pi})",
            [
                f"**Planteo de la serie alternante:** Evaluamos $\\pi \\approx 4 \\sum_{{k=0}}^{{{pasos_pi - 1}}} \\frac{{(-1)^k}}{{2k+1}}$ acumulando signos alternados de amplitud decreciente.",
                "**Primeros 4 términos explícitos:** $k=0: +4$; $k=1: 4 - \\frac{4}{3} \\approx 2.666667$; $k=2: 2.666667 + \\frac{4}{5} \\approx 3.466667$; $k=3: 3.466667 - \\frac{4}{7} \\approx 2.895238$ (oscila acotando el límite exacto).",
                f"**Cota del primer término omitido:** Al detener en $N = {pasos_pi}$, el error teórico queda acotado por $\\frac{{4}}{{2({pasos_pi}) + 1}} = {cota_leib:.6f}$.",
                f"**Resultado acumulado:** $\\pi_{{{pasos_pi}}} = {res_pi['pi_aproximado']:.6f}$ frente a $\\pi = {res_pi['pi_exacto']:.6f}$ ($E_a = {res_pi['error_absoluto']:.4e}$).",
            ],
            clave_ia="fund_taylor_pi",
        )
        render_dataframe_acotado(hist_pi, max_filas=300, key_prefix="tbl_fund_pi")
        muestra_pi = muestrear_para_grafico(hist_pi, max_puntos=220)
        fig_pi = crear_figura_base_plotly(
            "Oscilación amortiguada de Gregory-Leibniz (convergencia lenta)",
            "Términos (k)",
            "Aprox π",
            altura=340,
        )
        fig_pi.add_hline(y=res_pi["pi_exacto"], line_dash="dash", line_color=PALETA_ARANDU["oro"], annotation_text="π exacto")
        fig_pi.add_trace(
            go.Scatter(
                x=[r["k"] for r in muestra_pi],
                y=[r["pi_aprox"] for r in muestra_pi],
                mode="lines+markers",
                name="π Leibniz",
                line=dict(color=PALETA_ARANDU["salvia"], width=2.2),
            )
        )
        render_plotly_arandu(fig_pi, key="plot_fund_pi")


def _tab_series_taylor() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("series_taylor")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"x_euler": 1.0, "n_terminos_euler": 8, "cifras_scarborough": 3, "cifras_leibniz": 3},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_series_taylor(defaults)


@st.fragment
def _frag_aritmetica_maquina(defaults: dict) -> None:
    st.markdown(
        r"En memoria finita de $k$ dígitos, las operaciones no son exactamente asociativas: "
        r"restar primero dos números cercanos (`u - v`) elimina bits significativos antes de operar con `w`."
    )
    modo = render_controles_edicion_practica(
        clave_state="fund_asoc",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica $u$, $v$, $w$ o la mantisa de $k$ dígitos en vivo.",
    )
    solo_lectura = modo == "catedra"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k_dig = st.slider(
            "Mantisa ($k$ dígitos):",
            2,
            7,
            int(st.session_state["fund_asoc_k_digitos"]),
            disabled=solo_lectura,
            key="ui_asoc_k",
        )
    with c2:
        u_val = st.number_input("$u$:", value=float(st.session_state["fund_asoc_u_val"]), format="%.4f", disabled=solo_lectura, key="ui_asoc_u")
    with c3:
        v_val = st.number_input("$v$:", value=float(st.session_state["fund_asoc_v_val"]), format="%.4f", disabled=solo_lectura, key="ui_asoc_v")
    with c4:
        w_val = st.number_input("$w$:", value=float(st.session_state["fund_asoc_w_val"]), format="%.4f", disabled=solo_lectura, key="ui_asoc_w")

    if not solo_lectura:
        st.session_state["fund_asoc_k_digitos"] = k_dig
        st.session_state["fund_asoc_u_val"] = u_val
        st.session_state["fund_asoc_v_val"] = v_val
        st.session_state["fund_asoc_w_val"] = w_val

    res_asoc = comparar_asociatividad_personalizada(u=u_val, v=v_val, w=w_val, k=k_dig)
    st.markdown("### Bloque 3 — Impacto de la Pérdida de Asociatividad")
    st.metric("Valor Exacto (u - v) - w", f"{res_asoc['exacto']:.8g}")
    tabla_asoc = res_asoc["tabla"]
    t_a0, t_a1 = tabla_asoc[0], tabla_asoc[1]
    render_deduccion_paso_a_paso_viva(
        f"Asociatividad en punto flotante (k={k_dig})",
        [
            f"**Referencia exacta:** Evaluamos en precisión exacta $(u - v) - w = ({u_val:g} - {v_val:g}) - {w_val:g} = {res_asoc['exacto']:.8g}$ con registros normalizados a $k = {k_dig}$ dígitos.",
            f"**Agrupación 1 — `{t_a0['agrupacion']}`:** Al restar primero $(u - v)$ el intermedio normalizado vale `{t_a0['paso_intermedio_redondeo']:g}`; al operar luego con $w$ se obtiene `{t_a0['redondeo_norm']}` en redondeo ($E_r = {t_a0['error_redondeo_pct']:.3g}\\%$) y `{t_a0['trunc_norm']}` en truncamiento ($E_r = {t_a0['error_trunc_pct']:.3g}\\%$).",
            f"**Agrupación 2 — `{t_a1['agrupacion']}`:** Al restar primero $(u - w)$ el intermedio en redondeo da `{t_a1['paso_intermedio_redondeo']:g}` (desplazando los dígitos menos significativos de $w$ si las magnitudes difieren); al restar $v$ resulta `{t_a1['redondeo_norm']}` en redondeo ($E_r = {t_a1['error_redondeo_pct']:.3g}\\%$) y `{t_a1['trunc_norm']}` en truncamiento ($E_r = {t_a1['error_trunc_pct']:.3g}\\%$).",
            "**Conclusión:** En aritmética finita $(u \\ominus v) \\ominus w \\neq (u \\ominus w) \\ominus v$; conviene agrupar primero operandos cercanos en magnitud para minimizar la absorción.",
        ],
        clave_ia="fund_asoc_viva",
    )
    render_dataframe_acotado(tabla_asoc, max_filas=300, key_prefix="tbl_fund_asoc")

    st.markdown("### Bloque 4 — Comparación Visual del Error Relativo (%)")
    k_vis = render_slider_paso_k("Ver hasta el caso $k$:", len(tabla_asoc), key="vis_asoc_k")
    sub_asoc = tabla_asoc[:k_vis]
    fig = crear_figura_base_plotly("Error Relativo (%) según Orden de Asociación", "Esquema", "Error Relativo (%)", altura=360)
    fig.add_trace(
        go.Bar(
            x=[r["agrupacion"] for r in sub_asoc],
            y=[r["error_redondeo_pct"] for r in sub_asoc],
            name="Redondeo",
            marker_color=[
                PALETA_ARANDU["salvia"] if r["error_redondeo_pct"] < 2 else PALETA_ARANDU["terracota"]
                for r in sub_asoc
            ],
            text=[f"{r['error_redondeo_pct']:.2f}%" for r in sub_asoc],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[r["agrupacion"] for r in sub_asoc],
            y=[r["error_trunc_pct"] for r in sub_asoc],
            name="Truncamiento",
            marker_color=PALETA_ARANDU["oro"],
            text=[f"{r['error_trunc_pct']:.2f}%" for r in sub_asoc],
            textposition="auto",
        )
    )
    fig.update_layout(barmode="group")
    render_plotly_arandu(fig, key="plot_fund_asoc")

    with st.expander(r"Calculadora Elemental fl(x op y) a k dígitos"):
        cx, cop, cy = st.columns([2, 1, 2])
        ox = cx.number_input("$x$:", value=1.0 / 3.0, format="%.6f", key="op_x")
        oper = cop.selectbox("Operación:", ["+", "-", "*", "/"], key="op_sym")
        oy = cy.number_input("$y$:", value=5.0 / 7.0, format="%.6f", key="op_y")
        res_op = operar_en_maquina(ox, oy, oper, k=k_dig)
        st.json(res_op)

    with st.expander("Simulador de Épsilon de la Computadora (fl(1 + ε) > 1)"):
        res_eps = calcular_epsilon_maquina(k_digitos=k_dig)
        st.markdown(
            rf"El **épsilon de máquina ($\varepsilon_M$)** es el menor número positivo tal que en la mantisa de $k = {k_dig}$ dígitos "
            r"se cumple $fl(1 + \varepsilon) > 1$. Por debajo de ese umbral se produce absorción en la suma con $1$."
        )
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"ε empírico (k={k_dig})", f"{res_eps['eps_empirico_biseccion']:.3e}")
        e2.metric("Cota redondeo (0.5·10^(1-k))", f"{res_eps['eps_teorico_redondeo']:.3e}")
        e3.metric("IEEE 754 float32 (2⁻²³)", f"{res_eps['eps_ieee754_float32']:.3e}")
        e4.metric("IEEE 754 float64 (2⁻⁵²)", f"{res_eps['eps_ieee754_float64']:.3e}")
        registrar_contexto_ia(
            "Épsilon de la Computadora",
            f"- **Mantisa:** `k = {k_dig}` dígitos | **Último ε distinguido (bisección):** `{res_eps['eps_empirico_biseccion']:.4e}` | "
            f"**Absorción en 1:** `{res_eps['eps_absorcion_biseccion']:.4e}` | **Cota teórica redondeo:** `{res_eps['eps_teorico_redondeo']:.4e}`",
            clave="fund_eps_res",
        )
        render_dataframe_acotado(res_eps["historial"], max_filas=100, key_prefix="tbl_fund_eps")


def _tab_aritmetica_maquina() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("aritmetica_maquina")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"k_digitos": 3, "u_val": 121.0, "v_val": 119.0, "w_val": 0.327},
    )
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_aritmetica_maquina(defaults)


@st.fragment
def _frag_cancelacion_cuadratica(defaults: dict) -> None:
    st.markdown(
        r"Cuando $b^2 \gg 4ac$, $\sqrt{b^2 - 4ac} \approx |b|$. Calcular $-b + \sqrt{\Delta}$ resta dos flotantes casi "
        r"idénticos, anulando dígitos significativos (**cancelación catastrófica**). Racionalizar convierte la resta en suma + división."
    )
    modo = render_controles_edicion_practica(
        clave_state="fund_canc",
        defaults_dict=defaults,
        mensaje_edicion=r"Modifica $a$, $b$, $c$ o los dígitos $k$ para provocar o evitar cancelación catastrófica.",
    )
    solo_lectura = modo == "catedra"

    c1, c2, c3, c4 = st.columns(4)
    a_val = c1.number_input("$a$:", value=float(st.session_state["fund_canc_a"]), format="%.4f", disabled=solo_lectura, key="ui_canc_a")
    b_val = c2.number_input("$b$:", value=float(st.session_state["fund_canc_b"]), format="%.4f", disabled=solo_lectura, key="ui_canc_b")
    c_val = c3.number_input("$c$:", value=float(st.session_state["fund_canc_c"]), format="%.4f", disabled=solo_lectura, key="ui_canc_c")
    k_dig = c4.slider("Dígitos significativos $k$:", 2, 8, int(st.session_state["fund_canc_k_digitos"]), disabled=solo_lectura, key="ui_canc_k")

    if not solo_lectura:
        st.session_state["fund_canc_a"] = a_val
        st.session_state["fund_canc_b"] = b_val
        st.session_state["fund_canc_c"] = c_val
        st.session_state["fund_canc_k_digitos"] = k_dig

    try:
        res = analizar_cancelacion_cuadratica(a=a_val, b=b_val, c=c_val, k=k_dig)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Coeficientes cuadráticos (a, b, c)",
            sugerencia_defecto="Usá un coeficiente principal a distinto de 0 y valores finitos reales (b² - 4ac >= 0 para raíces reales).",
            ejemplo_defecto="a = 1.0, b = 62.10, c = 1.0",
        )
        return

    st.markdown("### Bloque 3 — Resolvente Clásica vs Racionalizada")
    tabla_comp = res["tabla_comparativa"]
    pm = res["pasos_maquina"]
    pasos_canc_vivos = [
        f"**Representación a $k = {k_dig}$ dígitos:** $fl(a) = {pm['fl_a']:g}$, $fl(b) = {pm['fl_b']:g}$, $fl(c) = {pm['fl_c']:g}$. Raíces exactas: $x_1 = {res['x1_exacto']:.8g}$, $x_2 = {res['x2_exacto']:.8g}$.",
        f"**Cálculo finito del discriminante:** $fl(b^2) = {pm['fl_b_sq']:g}$, $fl(4ac) = {pm['fl_4ac']:g} \\implies \\Delta = fl(b^2 - 4ac) = {pm['fl_disc']:g}$, cuya raíz normalizada es $\\sqrt{{\\Delta}} = {pm['fl_sqrt_disc']:g}$.",
    ]
    for fila_c in tabla_comp:
        pasos_canc_vivos.append(
            f"**{fila_c['metodo']}:** Resultado en máquina `{fila_c['aprox_norm']}` ($E_r = {fila_c['error_relativo_pct']:.3g}\\%$) — {fila_c['diagnostico']}"
        )
    render_deduccion_paso_a_paso_viva(
        f"Cancelación Cuadrática (k={k_dig})",
        pasos_canc_vivos,
        clave_ia="fund_canc_viva",
    )
    render_dataframe_acotado(tabla_comp, max_filas=300, key_prefix="tbl_fund_canc")

    st.markdown("### Bloque 4 — Comparación de Pérdida de Precisión")
    paso_k = render_slider_paso_k("Comparar hasta $k$ variantes:", len(tabla_comp), key="vis_canc_k")
    sub_comp = tabla_comp[:paso_k]
    fig = crear_figura_base_plotly(f"Error Relativo (%) en Raíces (k = {k_dig})", "Variante", "Error Relativo (%)", altura=350)
    fig.add_trace(
        go.Bar(
            x=[r["metodo"] for r in sub_comp],
            y=[r["error_relativo_pct"] for r in sub_comp],
            name="Error relativo (%)",
            text=["CANCELACIÓN" if r["sufre_cancelacion"] else "ESTABLE" for r in sub_comp],
            textposition="auto",
            marker=dict(
                color=[
                    PALETA_ARANDU["terracota"] if r["sufre_cancelacion"] else PALETA_ARANDU["salvia"]
                    for r in sub_comp
                ],
                pattern_shape=["/" if r["sufre_cancelacion"] else "" for r in sub_comp],
            ),
        )
    )
    render_plotly_arandu(fig, key="plot_fund_canc")


def _tab_cancelacion_cuadratica() -> None:
    meta = render_bloque_1_teoria_y_ejemplo("cancelacion_cuadratica")
    practica = meta.get("practica_profesora", {})
    defaults = practica.get("datos_iniciales", {"a": 1.0, "b": 62.10, "c": 1.0, "k_digitos": 4})
    st.divider()
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_cancelacion_cuadratica(defaults)


@st.fragment
def _frag_horner(defaults: dict) -> None:
    st.markdown(
        r"Factorizar en forma anidada reduce el costo de $O(N^2)$ a $O(N)$ multiplicaciones (sin calcular potencias $x^k$ repetidas), "
        r"acumulando menor error de redondeo en cada ciclo del algoritmo."
    )
    modo = render_controles_edicion_practica(
        clave_state="fund_horner",
        defaults_dict=defaults,
        mensaje_edicion=r"Haz doble clic en la tabla de coeficientes $[a_n \dots a_0]$ para editar el polinomio.",
    )
    solo_lectura = modo == "catedra"

    c1, c2 = st.columns(2)
    x_val = c1.number_input("Punto $x$:", value=float(st.session_state["fund_horner_x_eval"]), format="%.4f", disabled=solo_lectura, key="ui_horner_x")
    k_dig = c2.slider("Mantisa ($k$ dígitos):", 2, 8, int(st.session_state["fund_horner_k_digitos"]), disabled=solo_lectura, key="ui_horner_k")

    coefs_curr = st.session_state["fund_horner_coefs"]
    df_in = pd.DataFrame(
        {"Potencia": [f"x^{len(coefs_curr) - 1 - i}" for i in range(len(coefs_curr))], "Coeficiente": coefs_curr}
    )
    df_edit = render_data_editor_arandu(
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
    st.markdown("### Bloque 3 — Evaluación Directa vs Anidada de Horner")
    m1, m2, m3 = st.columns(3)
    m1.metric("Valor Exacto", f"{res['f_exacto']:.6f}")
    m2.metric("Estándar (Redondeo)", f"{tabla_err[1]['aproximacion']:.6g}", f"Er {tabla_err[1]['error_relativo_pct']:.2f}%")
    m3.metric("Horner (Redondeo)", f"{tabla_err[3]['aproximacion']:.6g}", f"Er {tabla_err[3]['error_relativo_pct']:.2f}%")
    registrar_contexto_ia(
        "Resultados Esquema de Horner vs Estándar",
        f"- **Punto $x = {x_val:g}$ | Dígitos $k = {k_dig}$ | Coeficientes:** `{coefs_curr}`\n"
        f"- **Valor exacto:** `{res['f_exacto']:.6f}` | **Estándar redondeo:** `{tabla_err[1]['aproximacion']:.6g}` | **Horner redondeo:** `{tabla_err[3]['aproximacion']:.6g}`",
        clave="fund_horner_res",
    )
    pasos_horner_vivos = [
        f"**Factorización anidada:** Para grado $n = {len(coefs_curr) - 1}$ con coeficientes `{coefs_curr}` evaluado en $x = {x_val:g}$ ($k = {k_dig}$ dígitos, valor exacto ${res['f_exacto']:.6f}$), agrupamos productos sucesivos por $x$ evitando calcular potencias $x^2, x^3, \\dots$ por separado."
    ]
    for ph in res.get("pasos_horner_round", []):
        pasos_horner_vivos.append(
            f"**Paso $k = {ph['paso']}$ (Redondeo):** `{ph['operacion']}` $\\implies b = {ph['valor']:g}$."
        )
    pasos_horner_vivos.append(
        f"**Comparación de precisión frente al exacto (${res['f_exacto']:.6f}$):** "
        f"Estándar (Redondeo) = `{tabla_err[1]['aproximacion_norm']}` ($E_r = {tabla_err[1]['error_relativo_pct']:.3g}\\%$) frente a "
        f"Horner (Redondeo) = `{tabla_err[3]['aproximacion_norm']}` ($E_r = {tabla_err[3]['error_relativo_pct']:.3g}\\%$) y "
        f"Horner (Truncado) = `{tabla_err[2]['aproximacion_norm']}` ($E_r = {tabla_err[2]['error_relativo_pct']:.3g}\\%)$."
    )
    render_deduccion_paso_a_paso_viva(
        f"Esquema Anidado de Horner (x={x_val:g}, k={k_dig})",
        pasos_horner_vivos,
        clave_ia="fund_horner_viva",
    )
    render_dataframe_acotado(tabla_err, max_filas=300, key_prefix="tbl_fund_horner")

    st.markdown("### Bloque 4 — Comparación Paso a Paso de Errores por Esquema")
    k_vis = render_slider_paso_k("Ver hasta el esquema $k$:", len(tabla_err), key="vis_horner_k")
    sub_h = tabla_err[:k_vis]
    fig = crear_figura_base_plotly("Error Relativo (%): Estándar vs Horner", "Esquema", "Error Rel (%)", altura=350)
    fig.add_trace(
        go.Bar(
            x=[r["metodo"] for r in sub_h],
            y=[r["error_relativo_pct"] for r in sub_h],
            marker_color=[
                PALETA_ARANDU["terracota"] if "Estándar" in r["metodo"] else PALETA_ARANDU["salvia"]
                for r in sub_h
            ],
        )
    )
    render_plotly_arandu(fig, key="plot_fund_horner")


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
    render_practica_catedra_info(
        titulo_practica=practica.get("titulo"),
        enunciado_practica=practica.get("enunciado"),
        solucion_referencia=practica.get("solucion_referencia"),
    )
    _frag_horner(defaults)


@st.fragment
def _frag_laboratorio_ffe_errores() -> None:
    st.markdown("## Laboratorio FFE: Propagación de Errores y Problema Inverso")
    st.markdown(
        r"Evalúa la **Fórmula Fundamental del Cálculo de Errores (FFE)** tanto en su forma directa "
        r"($\Delta u \le \sum \left|\frac{\partial f}{\partial x_i}\right| \Delta x_i$) como en el "
        r"**Problema Inverso por Igual Efecto** ($\Delta x_{i,\text{máx}} \le \frac{\Delta u_{\text{adm}}}{m \cdot |\partial f / \partial x_i|}$)."
    )
    catalogo_ffe = obtener_catalogo_escenarios_ffe()
    ids_ffe = list(catalogo_ffe.keys())
    if "ui_ffe_esc" not in st.session_state or st.session_state["ui_ffe_esc"] not in ids_ffe:
        st.session_state["ui_ffe_esc"] = ids_ffe[0]

    c_sel, c_mod = st.columns([1.4, 1.3])
    esc_id = c_sel.selectbox(
        "Modelo físico / matemático FFE:",
        options=ids_ffe,
        format_func=lambda k_e: catalogo_ffe[k_e]["nombre"],
        key="ui_ffe_esc",
    )
    modo_ffe = c_mod.radio(
        "Modo de cálculo:",
        ["Problema Directo (Δxᵢ → Δu)", "Problema Inverso por Igual Efecto (Δu_adm → Δxᵢ máx)"],
        horizontal=True,
        key="ui_ffe_modo",
    )
    meta = catalogo_ffe[esc_id]
    st.latex(meta["formula_tex"])

    es_directo = modo_ffe.startswith("Problema Directo")
    cols_v = st.columns(len(meta["variables"]) + (0 if es_directo else 1))
    vals_in: dict[str, float] = {}
    deltas_in: dict[str, float] = {}
    for idx_v, var in enumerate(meta["variables"]):
        with cols_v[idx_v]:
            vals_in[var] = st.number_input(
                f"Valor {var}:",
                value=float(meta["defaults"][var]),
                format="%.5f",
                help=f"Valor nominal de la variable {var}.",
                key=f"ui_ffe_v_{esc_id}_{var}",
            )
            if es_directo:
                deltas_in[var] = st.number_input(
                    f"Cota Δ{var}:",
                    value=float(meta["deltas"][var]),
                    min_value=0.0,
                    format="%.5f",
                    help=f"Error absoluto de entrada para {var}.",
                    key=f"ui_ffe_d_{esc_id}_{var}",
                )
    if es_directo:
        du_adm = float(meta["delta_u_adm"])
    else:
        with cols_v[-1]:
            du_adm = st.number_input(
                f"Δu admisible ({meta['unidad']}):",
                value=float(meta["delta_u_adm"]),
                min_value=1e-7,
                format="%.5f",
                help="Tolerancia máxima admitida en el resultado final u.",
                key=f"ui_ffe_adm_{esc_id}",
            )

    try:
        res_ffe = analizar_ffe_propagacion(
            escenario_id=esc_id,
            valores=vals_in,
            deltas=deltas_in if es_directo else meta["deltas"],
            delta_u_adm=du_adm,
        )
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Parámetros de entrada FFE",
            sugerencia_defecto="Ingresá valores finitos positivos donde la función y sus derivadas parciales existan (por ejemplo z != 0, L > 0, g > 0).",
            ejemplo_defecto="r = 2.5, h = 10.0, Δr = 0.01",
        )
        return

    tabla_v = res_ffe["tabla_variables"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Valor nominal u ({res_ffe['unidad']})", f"{res_ffe['u_nominal']:.5f}")
    if es_directo:
        m2.metric(f"Cota Δu ({res_ffe['unidad']})", f"{res_ffe['delta_u_directo']:.5f}")
        m3.metric("Error relativo Er(u)", f"{res_ffe['error_porcentual_directo']:.4f}%")
        var_crit = max(tabla_v, key=lambda r: r["peso_porcentual"])
        m4.metric("Mayor sensibilidad", f"{var_crit['variable']} ({var_crit['peso_porcentual']:.1f}%)")
    else:
        m2.metric(f"Cuota por variable (Δu/m)", f"{res_ffe['cuota_igual_efecto']:.5f}")
        min_cif = max(r["cifras_req_inverso"] for r in tabla_v)
        var_estr = min(tabla_v, key=lambda r: r["delta_x_max_inverso"])
        m3.metric("Variable más exigente", f"Δ{var_estr['variable']} <= {var_estr['delta_x_max_inverso']:.3e}")
        m4.metric("Cifras sig. mín. sugeridas", f"{min_cif}")

    pasos_ffe = [
        f"**Función nominal:** `{res_ffe['nombre']}` evaluada en `{res_ffe['valores']}` $\\implies u = {res_ffe['u_nominal']:.6f}\\,\\text{{{res_ffe['unidad']}}}$."
    ]
    if es_directo:
        for row_v in tabla_v:
            pasos_ffe.append(
                f"**Término `{row_v['variable']}`:** `{row_v['expresion_derivada']}` $\\implies \\left|\\frac{{\\partial u}}{{\\partial {row_v['variable']}}}\\right| = {row_v['sensibilidad_abs']:.5g}$; "
                f"aporte $= {row_v['sensibilidad_abs']:.5g} \\times {row_v['delta_x']:.5g} = {row_v['aporte_delta_u']:.6f}$ ({row_v['peso_porcentual']:.1f}% del error)."
            )
        pasos_ffe.append(
            f"**Suma FFE directa:** $\\Delta u \\le {res_ffe['delta_u_directo']:.6f}$ ($E_r = {res_ffe['error_porcentual_directo']:.4f}\\%)$."
        )
        y_key_bar = "aporte_delta_u"
        tit_bar = f"Aporte de cada variable al error Δu ({res_ffe['unidad']})"
        y_lbl_bar = f"Aporte |∂u/∂xᵢ|·Δxᵢ ({res_ffe['unidad']})"
    else:
        pasos_ffe.append(
            f"**Criterio de igual efecto ($m = {res_ffe['m_variables']}$ variables):** Cada variable recibe una cuota máxima de error $\\frac{{\\Delta u_{{\\text{{adm}}}}}}{{m}} = \\frac{{{res_ffe['delta_u_adm']:.5g}}}{{{res_ffe['m_variables']}}} = {res_ffe['cuota_igual_efecto']:.6f}$."
        )
        for row_v in tabla_v:
            pasos_ffe.append(
                f"**Cota máxima para `{row_v['variable']}`:** $\\Delta {row_v['variable']}_{{\\text{{máx}}}} = \\frac{{{res_ffe['cuota_igual_efecto']:.5g}}}{{{row_v['sensibilidad_abs']:.5g}}} = {row_v['delta_x_max_inverso']:.5e}$ "
                f"($E_r \\le {row_v['error_rel_inverso']:.3e}$, exige al menos `{row_v['cifras_req_inverso']}` cifras significativas)."
            )
        y_key_bar = "delta_x_max_inverso"
        tit_bar = "Cota máxima admisible Δxᵢ por Igual Efecto"
        y_lbl_bar = "Δxᵢ máxima admisible"

    registrar_contexto_ia(
        f"Laboratorio FFE ({esc_id})",
        f"- **Modo:** {modo_ffe} | **$u$:** `{res_ffe['u_nominal']:.6f}` | **Δu directo:** `{res_ffe['delta_u_directo']:.6f}` | **Δu_adm:** `{res_ffe['delta_u_adm']:.6f}`",
        clave="fund_ffe_res",
    )
    render_deduccion_paso_a_paso_viva(
        f"FFE — {res_ffe['nombre']}",
        pasos_ffe,
        clave_ia="fund_ffe_viva",
    )
    render_dataframe_acotado(tabla_v, max_filas=50, key_prefix=f"tbl_fund_ffe_{esc_id}")

    fig_ffe = crear_figura_base_plotly(tit_bar, "Variable", y_lbl_bar, altura=330)
    fig_ffe.add_trace(
        go.Bar(
            x=[r["variable"] for r in tabla_v],
            y=[r[y_key_bar] for r in tabla_v],
            name=y_lbl_bar,
            text=[f"{r[y_key_bar]:.4g}" for r in tabla_v],
            textposition="auto",
            marker=dict(
                color=[PALETA_ARANDU["terracota"] if i == 0 else (PALETA_ARANDU["oro"] if i == 1 else PALETA_ARANDU["salvia"]) for i in range(len(tabla_v))],
                pattern_shape=["/", "\\", "x"][:len(tabla_v)],
            ),
        )
    )
    render_plotly_arandu(fig_ffe, key=f"plot_fund_ffe")


def vista_fundamentos() -> None:
    inject_arandu_css()
    st.title("Fundamentos, Series y Teoría de Errores")
    st.markdown(
        r"Cómo traduce la CPU funciones ($e^x$, $\pi$) a bucles, por qué se pierde precisión en punto flotante ($k$ dígitos) "
        r"y cómo reordenar expresiones para preservar exactitud."
    )
    activo = render_selector_fichas_metodos(
        "fundamentos",
        (
            {
                "numero": "1",
                "clave_svg": "taylor",
                "titulo": "Series de Taylor (e^x y π)",
                "subtitulo": "Maclaurin · Truncamiento",
                "resumen": "Aproxima funciones como e^x, ln(1+x) y π sumando términos hasta alcanzar la tolerancia.",
                "pista": "Bucle for CPU · Error de resto",
            },
            {
                "numero": "2",
                "clave_svg": "aritmetica_maquina",
                "titulo": "Aritmética de Máquina & Asociatividad",
                "subtitulo": "Punto flotante · Mantisa k",
                "resumen": "Comprueba cuándo falla la propiedad asociativa (u+v)+w ≠ u+(v+w) al limitar la mantisa.",
                "pista": "Redondeo frente a truncamiento",
            },
            {
                "numero": "3",
                "clave_svg": "cancelacion",
                "titulo": "Cancelación Cuadrática",
                "subtitulo": "Racionalización · Estabilidad",
                "resumen": "Evita la pérdida de dígitos al restar números casi iguales mediante racionalización algebraica.",
                "pista": "Pérdida de dígitos · Condición",
            },
            {
                "numero": "4",
                "clave_svg": "horner",
                "titulo": "Forma Anidada de Horner",
                "subtitulo": "Anidamiento · Costo O(n)",
                "resumen": "Evalúa polinomios anidando productos sucesivos para reducir las multiplicaciones y los redondeos acumulados.",
                "pista": "Polinomio anidado · Costo O(n)",
            },
        ),
    )
    (
        _tab_series_taylor,
        _tab_aritmetica_maquina,
        _tab_cancelacion_cuadratica,
        _tab_horner,
    )[activo]()
    st.divider()
    _frag_laboratorio_ffe_errores()


vista_fundamentos()
