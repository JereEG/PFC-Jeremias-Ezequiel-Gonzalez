"""
Vista de Fundamentos y Teoría de Errores - Arandú.
Métodos: Series iterativas (e^x, pi), Aritmética de Máquina & Asociatividad,
Cancelación Catastrófica en Cuadráticas y Esquema Anidado de Horner.
Aislamiento Static-First mediante @st.fragment en cada laboratorio interactivo.
"""

from __future__ import annotations

import math

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
    render_dataframe_acotado,
    render_deduccion_paso_a_paso_viva,
    render_excepcion_validacion,
    render_plotly_arandu,
    render_practica_catedra_info,
    render_selector_fichas_metodos,
    render_slider_paso_k,
    render_toggle_modo_detallado,
    rerun_fragmento_seguro,
)


def _aplicar_preset_fundamentos(valores: dict[str, float | int], claves_ui: tuple[str, ...]) -> None:
    """Actualiza parámetros en session_state y limpia las claves UI asociadas."""
    for clave, valor in valores.items():
        st.session_state[clave] = valor
    for clave_ui in claves_ui:
        st.session_state.pop(clave_ui, None)


def _construir_filas_taylor_euler(historial: list[dict]) -> list[dict]:
    """Formatea la tabla de Maclaurin para e^x con encabezados limpios en español."""
    return [
        {
            "Término k": r["k"],
            "Expresión xᵏ/k!": r["expresion"],
            "Aporte Tₖ": round(float(r["termino"]), 8),
            "Suma Parcial Sₖ": round(float(r["suma_parcial"]), 8),
            "Valor Exacto": round(float(r["valor_exacto"]), 8),
            "Error Absoluto |Eₐ|": round(float(r["error_absoluto"]), 8),
            "Error Relativo (%)": round(float(r["error_relativo_pct"]), 6),
            "Error Aprox. Eᵣₚ (%)": round(float(r["error_aprox_rp_pct"]), 6)
            if r["error_aprox_rp_pct"] is not None
            else None,
            "¿Cumple Scarborough?": "Sí" if r["cumple_scarborough"] else "No",
        }
        for r in historial
    ]


def _construir_filas_taylor_pi(historial: list[dict]) -> list[dict]:
    """Formatea la tabla de Gregory-Leibniz para pi con encabezados limpios en español."""
    return [
        {
            "Paso k": r["k"],
            "Término (±4/(2k+1))": r["termino_expr"],
            "Aporte Numérico": round(float(r["termino_num"]), 8),
            "Aproximación πₖ": round(float(r["pi_aprox"]), 8),
            "π Exacto": round(float(r["pi_exacto"]), 8),
            "Error Absoluto |Eₐ|": round(float(r["error_absoluto"]), 8),
            "Error Relativo (%)": round(float(r["error_relativo_pct"]), 6),
        }
        for r in historial
    ]


def _simular_taylor_euler(defaults: dict, modo_detallado: bool) -> None:
    if "fund_taylor_x_euler" not in st.session_state:
        st.session_state["fund_taylor_x_euler"] = float(defaults.get("x_euler", 1.0))
    if "fund_taylor_n_terminos_euler" not in st.session_state:
        st.session_state["fund_taylor_n_terminos_euler"] = int(defaults.get("n_terminos_euler", 8))
    if "fund_taylor_cifras_scarborough" not in st.session_state:
        st.session_state["fund_taylor_cifras_scarborough"] = int(defaults.get("cifras_scarborough", 3))

    st.caption(
        r"Simulá el bucle acumulador $e^x \approx \sum_{k=0}^{N} \frac{x^k}{k!}$ evaluando cuándo "
        r"el error relativo aproximado $E_{rp}$ cae por debajo de la cota de Scarborough $E_s = 0.5 \times 10^{2-n}\%$."
    )
    claves_ui = ("ui_taylor_x", "ui_taylor_n", "ui_taylor_scar")
    b1, b2, b3 = st.columns(3)
    if b1.button(
        "Crecimiento base / Lab 1 (x = 1.0, N = 8, 3 cifras → Corta en k=7)",
        key="btn_tay_preset_base",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_taylor_x_euler": 1.0, "fund_taylor_n_terminos_euler": 8, "fund_taylor_cifras_scarborough": 3},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b2.button(
        "Atenuación RF negativa (x = -2.5, N = 15, 4 cifras → Signos alternos)",
        key="btn_tay_preset_neg",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_taylor_x_euler": -2.5, "fund_taylor_n_terminos_euler": 15, "fund_taylor_cifras_scarborough": 4},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b3.button(
        "Crecimiento rápido (x = 2.5, N = 16, 5 cifras → Alta precisión)",
        key="btn_tay_preset_rapido",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_taylor_x_euler": 2.5, "fund_taylor_n_terminos_euler": 16, "fund_taylor_cifras_scarborough": 5},
            claves_ui,
        )
        rerun_fragmento_seguro()

    c1, c2, c3 = st.columns(3)
    with c1:
        x_val = float(
            st.number_input(
                "Exponente x para eˣ:",
                min_value=-10.0,
                max_value=10.0,
                value=float(st.session_state["fund_taylor_x_euler"]),
                step=0.25,
                format="%.2f",
                key="ui_taylor_x",
            )
        )
    with c2:
        n_max = int(
            st.slider(
                "Orden máximo N (términos 0..N):",
                min_value=2,
                max_value=30,
                value=int(st.session_state["fund_taylor_n_terminos_euler"]),
                key="ui_taylor_n",
            )
        )
    with c3:
        cifras_s = int(
            st.slider(
                "Cifras significativas n (Scarborough):",
                min_value=1,
                max_value=8,
                value=int(st.session_state["fund_taylor_cifras_scarborough"]),
                key="ui_taylor_scar",
            )
        )

    st.session_state["fund_taylor_x_euler"] = x_val
    st.session_state["fund_taylor_n_terminos_euler"] = n_max
    st.session_state["fund_taylor_cifras_scarborough"] = cifras_s

    try:
        res = calcular_serie_euler(n_max=n_max, x_val=x_val, cifras_scarborough=cifras_s)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Parámetros de Serie de Taylor eˣ",
            sugerencia_defecto="Usá x en [-10, 10], orden N >= 2 y cifras significativas >= 1.",
            ejemplo_defecto="x = 1.0, N = 8, cifras = 3",
        )
        return

    k_corte = res.get("termino_scarborough")
    val_corte = f"k = {k_corte}" if k_corte is not None else f"> {n_max} (aumentar N)"
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Aproximación eˣ",
        f"{res['aproximacion_float']:.8f}",
        f"Err: {res['error_absoluto_final']:.2e} ({res['error_relativo_final_pct']:.4f}%)",
        delta_color="off",
    )
    m2.metric(
        "Valor Exacto exp(x)",
        f"{res['valor_exacto']:.8f}",
        f"Orden N = {n_max} en x = {x_val:g}",
        delta_color="off",
    )
    m3.metric(
        "Parada Scarborough",
        val_corte,
        f"Cota Eₛ = {res['es_scarborough_pct']:.4g}% ({cifras_s} cifras)",
        delta_color="off",
    )

    hist = res["historial"]
    k_vis = (
        render_slider_paso_k("Ver convergencia hasta el término $k$:", len(hist), key="vis_taylor_k")
        if modo_detallado
        else len(hist)
    )
    sub_hist = hist[:k_vis]
    fig = crear_figura_base_plotly(
        f"Convergencia de Maclaurin para e^({x_val:g}) hasta orden N = {n_max}",
        x_label="Índice del término (k)",
        y_label="Suma parcial Sₖ",
    )
    fig.add_hline(y=res["valor_exacto"], line_dash="dash", line_color=PALETA_ARANDU["oro"], annotation_text="Exacto")
    fig.add_trace(
        go.Scatter(
            x=[r["k"] for r in sub_hist],
            y=[r["suma_parcial"] for r in sub_hist],
            mode="lines+markers",
            name="Suma parcial eˣ",
            line=dict(color=PALETA_ARANDU["salvia"], width=2.5),
            marker=dict(
                size=7,
                color=PALETA_ARANDU["marfil_nodo"],
                line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.5),
            ),
        )
    )
    render_plotly_arandu(fig, key="plot_fund_taylor")

    registrar_contexto_ia(
        "Resultados Serie de Maclaurin e^x",
        f"- **$x = {x_val:g}$ | Orden máx:** `{n_max}` | **Cifras:** `{cifras_s}`\n"
        f"- **Aproximación:** `{res['aproximacion_float']:.8f}` | **Exacto:** `{res['valor_exacto']:.8f}` | "
        f"**Error abs.:** `{res['error_absoluto_final']:.3e}` | **Corte Scarborough:** `{val_corte}`",
        clave="fund_taylor_res",
    )

    if modo_detallado:
        st.caption(
            f"Cota teórica del Resto de Lagrange $|R_{{{n_max}}}({x_val:g})| \\le {res['cota_resto_lagrange']:.3e}$ "
            f"(error real observado: `{res['error_absoluto_final']:.3e}`)."
        )
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
                f"**Cierre en $k = {n_max}$:** $S_{{{n_max}}} = {res['aproximacion_float']:.8f}$ frente a "
                f"$\\exp({x_val:g}) = {res['valor_exacto']:.8f}$ ($E_a = {res['error_absoluto_final']:.3e}$)."
            )
        if k_corte is not None:
            pasos_taylor_vivos.append(
                f"**Verificación de corte:** El criterio de Scarborough ($|E_{{rp}}| < {res['es_scarborough_pct']:.4g}\\%$) se alcanza en $k = {k_corte}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"Serie Maclaurin e^x (x={x_val:g}, N={n_max})",
            pasos_taylor_vivos,
            clave_ia="fund_taylor_ex",
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_taylor_euler(hist), max_filas=300, key_prefix="tbl_fund_taylor")


def _simular_taylor_pi(modo_detallado: bool) -> None:
    if "fund_taylor_pasos_pi" not in st.session_state:
        st.session_state["fund_taylor_pasos_pi"] = 100

    st.caption(
        r"Serie alternante de Gregory-Leibniz $\pi \approx 4 \sum_{k=0}^{N-1} \frac{(-1)^k}{2k+1}$: "
        r"convergencia lenta $\mathcal{O}(1/N)$ acotada por el primer término omitido $\frac{4}{2N+1}$."
    )
    b1, b2, b3 = st.columns(3)
    if b1.button("100 términos (~2 decimales)", key="btn_pi_100", use_container_width=True):
        _aplicar_preset_fundamentos({"fund_taylor_pasos_pi": 100}, ("ui_pi_pasos",))
        rerun_fragmento_seguro()
    if b2.button("500 términos (Error ≤ 0.004)", key="btn_pi_500", use_container_width=True):
        _aplicar_preset_fundamentos({"fund_taylor_pasos_pi": 500}, ("ui_pi_pasos",))
        rerun_fragmento_seguro()
    if b3.button("2000 términos — Lab 1 Ej. 6 (3 decimales)", key="btn_pi_2000", use_container_width=True):
        _aplicar_preset_fundamentos({"fund_taylor_pasos_pi": 2000}, ("ui_pi_pasos",))
        rerun_fragmento_seguro()

    pasos_pi = int(
        st.slider(
            "Cantidad de términos N para aproximar π:",
            min_value=10,
            max_value=2500,
            value=int(st.session_state["fund_taylor_pasos_pi"]),
            step=10,
            key="ui_pi_pasos",
        )
    )
    st.session_state["fund_taylor_pasos_pi"] = pasos_pi

    res_pi = calcular_pi_leibniz_historial(n_terminos=pasos_pi, max_muestras=220)
    cota_leib = res_pi["cota_leibniz"]

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Aproximación π",
        f"{res_pi['pi_aproximado']:.6f}",
        f"Err: {res_pi['error_absoluto']:.2e} ({res_pi['error_relativo_pct']:.4f}%)",
        delta_color="off",
    )
    m2.metric("π Exacto", f"{res_pi['pi_exacto']:.6f}", f"N = {pasos_pi} términos", delta_color="off")
    m3.metric("Cota Teórica Alternante", f"≤ {cota_leib:.6f}", "Primer término omitido 4/(2N+1)", delta_color="off")

    hist_pi = res_pi["historial"]
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

    registrar_contexto_ia(
        "Resultados Serie Gregory-Leibniz pi",
        f"- **Términos N:** `{pasos_pi}` | **π aprox:** `{res_pi['pi_aproximado']:.6f}` | "
        f"**Error abs.:** `{res_pi['error_absoluto']:.3e}` | **Cota 4/(2N+1):** `{cota_leib:.6f}`",
        clave="fund_taylor_res_pi",
    )

    if modo_detallado:
        render_deduccion_paso_a_paso_viva(
            f"Serie Gregory-Leibniz π (N={pasos_pi})",
            [
                f"**Planteo alternante:** Evaluamos $\\pi \\approx 4 \\sum_{{k=0}}^{{{pasos_pi - 1}}} \\frac{{(-1)^k}}{{2k+1}}$.",
                "**Primeros 4 pasos:** $k=1: +4$; $k=2: 4 - \\frac{4}{3} \\approx 2.666667$; $k=3: + \\frac{4}{5} \\approx 3.466667$; $k=4: - \\frac{4}{7} \\approx 2.895238$.",
                f"**Cota del resto alternante:** Al detener en $N = {pasos_pi}$, $|E_a| \\le \\frac{{4}}{{2({pasos_pi}) + 1}} = {cota_leib:.6f}$.",
                f"**Resultado acumulado:** $\\pi_{{{pasos_pi}}} = {res_pi['pi_aproximado']:.6f}$ frente a $\\pi = {res_pi['pi_exacto']:.6f}$ ($E_a = {res_pi['error_absoluto']:.4e}$).",
            ],
            clave_ia="fund_taylor_pi",
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_taylor_pi(hist_pi), max_filas=300, key_prefix="tbl_fund_pi")


@st.fragment
def _frag_series_taylor(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    serie_sel = st.radio(
        "Escenario / serie iterativa:",
        options=["euler_ex", "pi_leibniz"],
        format_func=lambda op: (
            "Caso Real: Kernel eˣ sin FPU (Atenuación / Crecimiento con Scarborough)"
            if op == "euler_ex"
            else "Serie Alternante π: Gregory-Leibniz (Convergencia Lenta O(1/N))"
        ),
        horizontal=True,
        key="ui_taylor_selector",
    )
    if serie_sel == "euler_ex":
        _simular_taylor_euler(defaults, modo_detallado)
    else:
        _simular_taylor_pi(modo_detallado)


def _tab_series_taylor() -> None:
    modo_detallado = render_toggle_modo_detallado("series_taylor")
    meta = render_bloque_1_teoria_y_ejemplo("series_taylor", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"x_euler": 1.0, "n_terminos_euler": 8, "cifras_scarborough": 3, "cifras_leibniz": 3},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_series_taylor(defaults, modo_detallado=modo_detallado)


def _construir_filas_asociatividad(tabla: list[dict]) -> list[dict]:
    """Formatea la comparativa de asociatividad con encabezados limpios en español."""
    return [
        {
            "Agrupación Evaluada": r["agrupacion"],
            "Exacto": round(float(r["exacto"]), 8),
            "Intermedio (Redondeo)": round(float(r["paso_intermedio_redondeo"]), 6),
            "Redondeo Normalizado": r["redondeo_norm"],
            "Valor Redondeado": round(float(r["redondeo_val"]), 8),
            "Error Redondeo (%)": round(float(r["error_redondeo_pct"]), 4),
            "Truncamiento Normalizado": r["trunc_norm"],
            "Valor Truncado": round(float(r["trunc_val"]), 8),
            "Error Truncamiento (%)": round(float(r["error_trunc_pct"]), 4),
        }
        for r in tabla
    ]


def _construir_filas_epsilon(historial: list[dict]) -> list[dict]:
    """Formatea la tabla del épsilon de máquina con encabezados limpios en español."""
    return [
        {
            "Paso": r["paso"],
            "Épsilon ε": r["epsilon"],
            "ε Normalizado": r["epsilon_norm"],
            "fl(1 + ε) Redondeo": r["fl_1_mas_eps_redondeo"],
            "fl(1 + ε) Truncamiento": r["fl_1_mas_eps_trunc"],
            "Estado en Máquina": r["estado"],
        }
        for r in historial
    ]


def _render_sublabs_aritmetica_detallada(k_dig: int) -> None:
    """Despliega calculadora elemental y análisis de épsilon solo en Modo Detallado."""
    with st.expander(r"Calculadora Elemental fl(x op y) a k dígitos"):
        cx, cop, cy = st.columns([2, 1, 2])
        ox = cx.number_input("Operando x:", value=1.0 / 3.0, format="%.6f", key="op_x")
        oper = cop.selectbox("Operación:", ["+", "-", "*", "/"], key="op_sym")
        oy = cy.number_input("Operando y:", value=5.0 / 7.0, format="%.6f", key="op_y")
        try:
            res_op = operar_en_maquina(ox, oy, oper, k=k_dig)
            o1, o2, o3 = st.columns(3)
            o1.metric("Valor Exacto", f"{res_op['exacto']:.8g}", f"Operación {res_op['operacion']}", delta_color="off")
            o2.metric(
                f"Redondeo (k={k_dig})",
                res_op["redondeo"]["norm_str"],
                f"Err rel.: {res_op['redondeo']['error_relativo_pct']:.4f}%",
                delta_color="off",
            )
            o3.metric(
                f"Truncamiento (k={k_dig})",
                res_op["truncamiento"]["norm_str"],
                f"Err rel.: {res_op['truncamiento']['error_relativo_pct']:.4f}%",
                delta_color="off",
            )
        except Exception as exc:
            render_excepcion_validacion(
                exc,
                campo="Operandos de máquina fl(x op y)",
                sugerencia_defecto="Ingresá valores finitos y evitá dividir por y = 0.",
                ejemplo_defecto="x = 0.333333, y = 0.714286, op = '+'",
            )

    with st.expander("Simulador de Épsilon de la Computadora (fl(1 + ε) > 1)"):
        res_eps = calcular_epsilon_maquina(k_digitos=k_dig)
        st.caption(
            rf"El **épsilon de máquina ($\varepsilon_M$)** es el menor positivo con $fl(1 + \varepsilon) > 1$ para mantisa $k = {k_dig}$; "
            r"incrementos menores se absorben en $1$."
        )
        e1, e2, e3 = st.columns(3)
        e1.metric(
            f"ε Empírico (k={k_dig})",
            f"{res_eps['eps_empirico_biseccion']:.3e}",
            f"Absorbe en {res_eps['eps_absorcion_biseccion']:.3e}",
            delta_color="off",
        )
        e2.metric(
            "Cota Teórica Redondeo",
            f"{res_eps['eps_teorico_redondeo']:.3e}",
            f"Truncamiento: {res_eps['eps_teorico_truncamiento']:.3e}",
            delta_color="off",
        )
        e3.metric(
            "IEEE 754 float32 / float64",
            f"{res_eps['eps_ieee754_float32']:.3e}",
            f"float64: {res_eps['eps_ieee754_float64']:.3e}",
            delta_color="off",
        )
        registrar_contexto_ia(
            "Épsilon de la Computadora",
            f"- **Mantisa:** `k = {k_dig}` dígitos | **Último ε distinguido:** `{res_eps['eps_empirico_biseccion']:.4e}` | "
            f"**Absorción en 1:** `{res_eps['eps_absorcion_biseccion']:.4e}`",
            clave="fund_eps_res",
        )
        render_dataframe_acotado(
            _construir_filas_epsilon(res_eps["historial"]),
            max_filas=100,
            key_prefix="tbl_fund_eps",
        )


@st.fragment
def _frag_aritmetica_maquina(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    if "fund_asoc_k_digitos" not in st.session_state:
        st.session_state["fund_asoc_k_digitos"] = int(defaults.get("k_digitos", 3))
    if "fund_asoc_u_val" not in st.session_state:
        st.session_state["fund_asoc_u_val"] = float(defaults.get("u_val", 121.0))
    if "fund_asoc_v_val" not in st.session_state:
        st.session_state["fund_asoc_v_val"] = float(defaults.get("v_val", 119.0))
    if "fund_asoc_w_val" not in st.session_state:
        st.session_state["fund_asoc_w_val"] = float(defaults.get("w_val", 0.327))

    st.caption(
        r"Compará el orden seguro `(u - v) - w` frente al orden riesgoso `(u - w) - v` al operar con registros finitos de $k$ dígitos."
    )
    claves_ui = ("ui_asoc_k", "ui_asoc_u", "ui_asoc_v", "ui_asoc_w")
    b1, b2, b3 = st.columns(3)
    if b1.button(
        "TP2 Cátedra (k=3: 121 − 119 − 0.327 → 19.6% vs 0.18%)",
        key="btn_asoc_tp2",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_asoc_k_digitos": 3, "fund_asoc_u_val": 121.0, "fund_asoc_v_val": 119.0, "fund_asoc_w_val": 0.327},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b2.button(
        "Absorción contable (k=4: 10000 − 9998 − 0.45 → 29% vs 0%)",
        key="btn_asoc_contable",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_asoc_k_digitos": 4, "fund_asoc_u_val": 10000.0, "fund_asoc_v_val": 9998.0, "fund_asoc_w_val": 0.45},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b3.button(
        "Alta mantisa (k=6: 121 − 119 − 0.327 → Recupera exactitud)",
        key="btn_asoc_alta",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_asoc_k_digitos": 6, "fund_asoc_u_val": 121.0, "fund_asoc_v_val": 119.0, "fund_asoc_w_val": 0.327},
            claves_ui,
        )
        rerun_fragmento_seguro()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k_dig = int(
            st.slider(
                "Mantisa (k dígitos):",
                2,
                7,
                int(st.session_state["fund_asoc_k_digitos"]),
                key="ui_asoc_k",
            )
        )
    with c2:
        u_val = float(
            st.number_input("Operando u:", value=float(st.session_state["fund_asoc_u_val"]), format="%.4f", key="ui_asoc_u")
        )
    with c3:
        v_val = float(
            st.number_input("Operando v:", value=float(st.session_state["fund_asoc_v_val"]), format="%.4f", key="ui_asoc_v")
        )
    with c4:
        w_val = float(
            st.number_input("Operando w:", value=float(st.session_state["fund_asoc_w_val"]), format="%.4f", key="ui_asoc_w")
        )

    st.session_state["fund_asoc_k_digitos"] = k_dig
    st.session_state["fund_asoc_u_val"] = u_val
    st.session_state["fund_asoc_v_val"] = v_val
    st.session_state["fund_asoc_w_val"] = w_val

    try:
        res_asoc = comparar_asociatividad_personalizada(u=u_val, v=v_val, w=w_val, k=k_dig)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Parámetros u, v, w o dígitos k",
            sugerencia_defecto="Ingresá valores finitos y una mantisa k entre 1 y 15 dígitos.",
            ejemplo_defecto="k = 3, u = 121.0, v = 119.0, w = 0.327",
        )
        return

    tabla_asoc = res_asoc["tabla"]
    t_a0, t_a1 = tabla_asoc[0], tabla_asoc[1]
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Orden Seguro (u − v) − w",
        t_a0["redondeo_norm"],
        f"Err red.: {t_a0['error_redondeo_pct']:.2f}% | Trunc.: {t_a0['error_trunc_pct']:.2f}%",
        delta_color="off",
    )
    m2.metric(
        "Orden Riesgoso (u − w) − v",
        t_a1["redondeo_norm"],
        f"Err red.: {t_a1['error_redondeo_pct']:.2f}% | Trunc.: {t_a1['error_trunc_pct']:.2f}%",
        delta_color="off",
    )
    m3.metric(
        f"Exacto y Mantisa k={k_dig}",
        f"{res_asoc['exacto']:.6g}",
        f"Truncado riesgoso: {t_a1['trunc_norm']}",
        delta_color="off",
    )

    k_vis = (
        render_slider_paso_k("Ver hasta el caso $k$:", len(tabla_asoc), key="vis_asoc_k")
        if modo_detallado
        else len(tabla_asoc)
    )
    sub_asoc = tabla_asoc[:k_vis]
    fig = crear_figura_base_plotly(
        "Error Relativo (%) según Orden de Asociación",
        "Esquema evaluado",
        "Error Relativo (%)",
        altura=360,
    )
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

    registrar_contexto_ia(
        "Resultados Asociatividad en Máquina",
        f"- **Mantisa:** `k = {k_dig}` | **Exacto:** `{res_asoc['exacto']:.6g}`\n"
        f"- **Seguro:** `{t_a0['agrupacion']} = {t_a0['redondeo_norm']}` (`{t_a0['error_redondeo_pct']:.2f}%`)\n"
        f"- **Riesgoso:** `{t_a1['agrupacion']} = {t_a1['redondeo_norm']}` (`{t_a1['error_redondeo_pct']:.2f}%`)",
        clave="fund_asoc_res",
    )

    if modo_detallado:
        st.caption(
            f"Cotas por operación ($k={k_dig}$): redondeo $\\le {res_asoc['cota_rel_redondeo_pct']:.3g}\\%$, "
            f"truncamiento $< {res_asoc['cota_rel_trunc_pct']:.3g}\\%$."
        )
        render_deduccion_paso_a_paso_viva(
            f"Asociatividad en punto flotante (k={k_dig})",
            [
                f"**Referencia exacta:** Evaluamos en precisión exacta $(u - v) - w = ({u_val:g} - {v_val:g}) - {w_val:g} = {res_asoc['exacto']:.8g}$ con registros de $k = {k_dig}$ dígitos.",
                f"**Agrupación 1 — `{t_a0['agrupacion']}`:** Al restar primero $(u - v)$ el intermedio redondeado vale `{t_a0['paso_intermedio_redondeo']:g}`; al operar con $w$ se obtiene `{t_a0['redondeo_norm']}` en redondeo ($E_r = {t_a0['error_redondeo_pct']:.3g}\\%$) y `{t_a0['trunc_norm']}` en truncamiento ($E_r = {t_a0['error_trunc_pct']:.3g}\\%$).",
                f"**Agrupación 2 — `{t_a1['agrupacion']}`:** Al operar primero $(u - w)$ el intermedio en redondeo da `{t_a1['paso_intermedio_redondeo']:g}` (absorbiendo cifras de $w$); al restar $v$ resulta `{t_a1['redondeo_norm']}` en redondeo ($E_r = {t_a1['error_redondeo_pct']:.3g}\\%$) y `{t_a1['trunc_norm']}` en truncamiento ($E_r = {t_a1['error_trunc_pct']:.3g}\\%$).",
                "**Conclusión:** En aritmética finita $(u \\ominus v) \\ominus w \\neq (u \\ominus w) \\ominus v$; conviene agrupar primero operandos cercanos en magnitud.",
            ],
            clave_ia="fund_asoc_viva",
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_asociatividad(tabla_asoc), max_filas=300, key_prefix="tbl_fund_asoc")
        _render_sublabs_aritmetica_detallada(k_dig)


def _tab_aritmetica_maquina() -> None:
    modo_detallado = render_toggle_modo_detallado("aritmetica_maquina")
    meta = render_bloque_1_teoria_y_ejemplo("aritmetica_maquina", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"k_digitos": 3, "u_val": 121.0, "v_val": 119.0, "w_val": 0.327},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_aritmetica_maquina(defaults, modo_detallado=modo_detallado)


def _fmt_err_canc(er_pct: float) -> str:
    """Formatea un error relativo porcentual manejando NaN por división entre cero."""
    return f"Err rel.: {er_pct:.3f}%" if math.isfinite(er_pct) else "DIV/0 (indet.)"


def _construir_filas_cancelacion(tabla: list[dict]) -> list[dict]:
    """Formatea la comparativa de cancelación cuadrática con encabezados limpios en español."""
    return [
        {
            "Variante Evaluada": r["metodo"],
            "Valor Exacto": round(float(r["exacto"]), 8),
            "Máquina Normalizada": r["aprox_norm"],
            "Valor Máquina": round(float(r["aprox_num"]), 8) if math.isfinite(r["aprox_num"]) else None,
            "Error Absoluto": round(float(r["error_absoluto"]), 8) if math.isfinite(r["error_absoluto"]) else None,
            "Error Relativo (%)": round(float(r["error_relativo_pct"]), 4)
            if math.isfinite(r["error_relativo_pct"])
            else None,
            "Estado Numérico": "Cancelación" if r["sufre_cancelacion"] else "Estable",
            "Diagnóstico": r["diagnostico"],
        }
        for r in tabla
    ]


@st.fragment
def _frag_cancelacion_cuadratica(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    if "fund_canc_a" not in st.session_state:
        st.session_state["fund_canc_a"] = float(defaults.get("a", 1.0))
    if "fund_canc_b" not in st.session_state:
        st.session_state["fund_canc_b"] = float(defaults.get("b", 62.10))
    if "fund_canc_c" not in st.session_state:
        st.session_state["fund_canc_c"] = float(defaults.get("c", 1.0))
    if "fund_canc_k_digitos" not in st.session_state:
        st.session_state["fund_canc_k_digitos"] = int(defaults.get("k_digitos", 4))

    st.caption(
        r"Contrastá la resolvente tradicional frente a su forma racionalizada cuando $b^2 \gg 4ac$ en registros de $k$ dígitos."
    )
    claves_ui = ("ui_canc_a", "ui_canc_b", "ui_canc_c", "ui_canc_k")
    b1, b2, b3 = st.columns(3)
    if b1.button(
        "TP2 Cátedra (x² + 62.10x + 1 = 0, k=4 → 24.2% vs 0.04%)",
        key="btn_canc_tp2",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_canc_a": 1.0, "fund_canc_b": 62.10, "fund_canc_c": 1.0, "fund_canc_k_digitos": 4},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b2.button(
        "Raytracing 3D (x² + 3000.001x + 3 = 0, k=6 → Colapso 100% vs 0%)",
        key="btn_canc_ray3d",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_canc_a": 1.0, "fund_canc_b": 3000.001, "fund_canc_c": 3.0, "fund_canc_k_digitos": 6},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b3.button(
        "Signo opuesto b < 0 (x² − 62.10x + 1 = 0, k=4 → Simetría en x₂)",
        key="btn_canc_bneg",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_canc_a": 1.0, "fund_canc_b": -62.10, "fund_canc_c": 1.0, "fund_canc_k_digitos": 4},
            claves_ui,
        )
        rerun_fragmento_seguro()

    c1, c2, c3, c4 = st.columns(4)
    a_val = float(c1.number_input("Coeficiente a:", value=float(st.session_state["fund_canc_a"]), format="%.4f", key="ui_canc_a"))
    b_val = float(c2.number_input("Coeficiente b:", value=float(st.session_state["fund_canc_b"]), format="%.4f", key="ui_canc_b"))
    c_val = float(c3.number_input("Coeficiente c:", value=float(st.session_state["fund_canc_c"]), format="%.4f", key="ui_canc_c"))
    k_dig = int(c4.slider("Mantisa (k dígitos):", 2, 8, int(st.session_state["fund_canc_k_digitos"]), key="ui_canc_k"))

    st.session_state["fund_canc_a"] = a_val
    st.session_state["fund_canc_b"] = b_val
    st.session_state["fund_canc_c"] = c_val
    st.session_state["fund_canc_k_digitos"] = k_dig

    try:
        res = analizar_cancelacion_cuadratica(a=a_val, b=b_val, c=c_val, k=k_dig)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Coeficientes cuadráticos (a, b, c) o dígitos k",
            sugerencia_defecto="Usá a != 0, discriminante b² - 4ac > 0 y mantisa k entre 1 y 15 dígitos.",
            ejemplo_defecto="a = 1.0, b = 62.10, c = 1.0, k = 4",
        )
        return

    par = res["pareja_estable"]
    r_vuln = par["trad_vulnerable"]
    r_salv = par["rat_rescatada"]
    r_dir = par["trad_directa"]

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"Clásica en {par['raiz_vulnerable']} (Riesgo)",
        r_vuln["aprox_norm"],
        _fmt_err_canc(r_vuln["error_relativo_pct"]),
        delta_color="off",
    )
    m2.metric(
        f"Racionalizada en {par['raiz_vulnerable']} (Segura)",
        r_salv["aprox_norm"],
        _fmt_err_canc(r_salv["error_relativo_pct"]),
        delta_color="off",
    )
    m3.metric(
        "Raíces Exactas (x₁ | x₂)",
        f"{res['x1_exacto']:.5g} | {res['x2_exacto']:.5g}",
        f"Directa {par['raiz_directa']}: {_fmt_err_canc(r_dir['error_relativo_pct'])}",
        delta_color="off",
    )

    tabla_comp = res["tabla_comparativa"]
    paso_k = (
        render_slider_paso_k("Comparar hasta $k$ variantes:", len(tabla_comp), key="vis_canc_k")
        if modo_detallado
        else len(tabla_comp)
    )
    sub_comp = tabla_comp[:paso_k]
    fig = crear_figura_base_plotly(
        f"Error Relativo (%) en Raíces (k = {k_dig})",
        "Variante evaluada",
        "Error Relativo (%)",
        altura=350,
    )
    fig.add_trace(
        go.Bar(
            x=[r["metodo"] for r in sub_comp],
            y=[r["error_relativo_pct"] if math.isfinite(r["error_relativo_pct"]) else 100.0 for r in sub_comp],
            name="Error relativo (%)",
            text=[
                "DIV/0 (NaN)"
                if not math.isfinite(r["error_relativo_pct"])
                else ("CANCELACIÓN" if r["sufre_cancelacion"] else "ESTABLE")
                for r in sub_comp
            ],
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

    registrar_contexto_ia(
        "Resultados Cancelación Cuadrática",
        f"- **Ecuación:** `{a_val:g}x^2 + ({b_val:g})x + ({c_val:g}) = 0` (`k = {k_dig}`)\n"
        f"- **Clásica vulnerable ({par['raiz_vulnerable']}):** `{r_vuln['aprox_norm']}` (`{_fmt_err_canc(r_vuln['error_relativo_pct'])}`)\n"
        f"- **Racionalizada ({par['raiz_vulnerable']}):** `{r_salv['aprox_norm']}` (`{_fmt_err_canc(r_salv['error_relativo_pct'])}`)",
        clave="fund_canc_res",
    )

    if modo_detallado:
        pm = res["pasos_maquina"]
        pasos_canc_vivos = [
            f"**Representación a $k = {k_dig}$ dígitos:** $fl(a) = {pm['fl_a']:g}$, $fl(b) = {pm['fl_b']:g}$, $fl(c) = {pm['fl_c']:g}$. Raíces exactas: $x_1 = {res['x1_exacto']:.8g}$, $x_2 = {res['x2_exacto']:.8g}$.",
            f"**Cálculo finito del discriminante:** $fl(b^2) = {pm['fl_b_sq']:g}$, $fl(4ac) = {pm['fl_4ac']:g} \\implies \\Delta = fl(b^2 - 4ac) = {pm['fl_disc']:g}$, cuya raíz normalizada es $\\sqrt{{\\Delta}} = {pm['fl_sqrt_disc']:g}$.",
        ]
        for fila_c in tabla_comp:
            err_txt = (
                f"$E_r = {fila_c['error_relativo_pct']:.3g}\\%$"
                if math.isfinite(fila_c["error_relativo_pct"])
                else "división por cero ($\\text{NaN}$)"
            )
            pasos_canc_vivos.append(
                f"**{fila_c['metodo']}:** Resultado en máquina `{fila_c['aprox_norm']}` ({err_txt}) — {fila_c['diagnostico']}"
            )
        render_deduccion_paso_a_paso_viva(
            f"Cancelación Cuadrática (k={k_dig})",
            pasos_canc_vivos,
            clave_ia="fund_canc_viva",
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_cancelacion(tabla_comp), max_filas=300, key_prefix="tbl_fund_canc")


def _tab_cancelacion_cuadratica() -> None:
    modo_detallado = render_toggle_modo_detallado("cancelacion_cuadratica")
    meta = render_bloque_1_teoria_y_ejemplo("cancelacion_cuadratica", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get("datos_iniciales", {"a": 1.0, "b": 62.10, "c": 1.0, "k_digitos": 4})
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_cancelacion_cuadratica(defaults, modo_detallado=modo_detallado)


def _parsear_coefs_horner(coefs_txt: str) -> list[float]:
    """Convierte coeficientes separados por coma [aₙ, ..., a₀] en una lista de flotantes."""
    partes = [p.strip() for p in (coefs_txt or "").split(",") if p.strip()]
    if len(partes) < 2:
        raise ValueError("Ingresá al menos 2 coeficientes separados por coma (ej. 1, -6, 3, -0.149).")
    try:
        return [float(p) for p in partes]
    except ValueError as exc:
        raise ValueError("Todos los coeficientes deben ser números reales válidos (ej. 1, -6, 3, -0.149).") from exc


def _construir_filas_horner(tabla_errores: list[dict]) -> list[dict]:
    """Formatea la comparativa Estándar vs Horner con encabezados limpios en español."""
    return [
        {
            "Esquema Evaluado": r["metodo"],
            "Aproximación Numérica": round(float(r["aproximacion"]), 8),
            "Máquina Normalizada": r["aproximacion_norm"],
            "Error Absoluto": round(float(r["error_absoluto"]), 8),
            "Error Relativo (%)": round(float(r["error_relativo_pct"]), 4),
        }
        for r in tabla_errores
    ]


@st.fragment
def _frag_horner(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    if "fund_horner_coefs_str" not in st.session_state:
        st.session_state["fund_horner_coefs_str"] = ", ".join(f"{c:g}" for c in defaults.get("coefs", [1.0, -6.0, 3.0, -0.149]))
    if "fund_horner_x_eval" not in st.session_state:
        st.session_state["fund_horner_x_eval"] = float(defaults.get("x_eval", 4.71))
    if "fund_horner_k_digitos" not in st.session_state:
        st.session_state["fund_horner_k_digitos"] = int(defaults.get("k_digitos", 3))

    st.caption(
        r"Compará la evaluación directa $\sum a_j x^j$ ($\mathcal{O}(n^2)$) frente a la forma anidada de **Horner** ($\mathcal{O}(n)$) en mantisa de $k$ dígitos."
    )
    claves_ui = ("ui_horner_coefs_txt", "ui_horner_x", "ui_horner_k")
    b1, b2, b3 = st.columns(3)
    if b1.button(
        "TP2 Cátedra (x³ − 6x² + 3x − 0.149 en x=4.71, k=3 → 4.35% vs 0.25%)",
        key="btn_horner_tp2",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_horner_coefs_str": "1, -6, 3, -0.149", "fund_horner_x_eval": 4.71, "fund_horner_k_digitos": 3},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b2.button(
        "Sensor Grado 5 (x=2.65, k=3 → 248% vs 11.3% y 5 vs 15 mult.)",
        key="btn_horner_g5",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_horner_coefs_str": "0.5, -4.5, 14, -18, 9, -1.2", "fund_horner_x_eval": 2.65, "fund_horner_k_digitos": 3},
            claves_ui,
        )
        rerun_fragmento_seguro()
    if b3.button(
        "Mal condicionado ((x − 2)⁴ en x=2.25, k=4 → Falla por condición)",
        key="btn_horner_cond",
        use_container_width=True,
    ):
        _aplicar_preset_fundamentos(
            {"fund_horner_coefs_str": "1, -8, 24, -32, 16", "fund_horner_x_eval": 2.25, "fund_horner_k_digitos": 4},
            claves_ui,
        )
        rerun_fragmento_seguro()

    c_coef, c_x, c_k = st.columns([2.0, 1.0, 1.0])
    coefs_txt = c_coef.text_input(
        "Coeficientes [aₙ, …, a₀] (orden descendente):",
        value=str(st.session_state["fund_horner_coefs_str"]),
        key="ui_horner_coefs_txt",
    )
    x_val = float(
        c_x.number_input("Punto x:", value=float(st.session_state["fund_horner_x_eval"]), format="%.4f", key="ui_horner_x")
    )
    k_dig = int(
        c_k.slider("Mantisa (k dígitos):", 2, 8, int(st.session_state["fund_horner_k_digitos"]), key="ui_horner_k")
    )

    st.session_state["fund_horner_coefs_str"] = coefs_txt
    st.session_state["fund_horner_x_eval"] = x_val
    st.session_state["fund_horner_k_digitos"] = k_dig

    try:
        coefs_curr = _parsear_coefs_horner(coefs_txt)
        res = evaluar_polinomio_horner(x_val=x_val, k=k_dig, coefs=coefs_curr)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Coeficientes del polinomio, punto x o dígitos k",
            sugerencia_defecto="Ingresá al menos 2 coeficientes reales separados por coma y valores finitos.",
            ejemplo_defecto="1, -6, 3, -0.149 (con x = 4.71, k = 3)",
        )
        return

    tabla_err = res["tabla_errores"]
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Estándar (Redondeo)",
        tabla_err[1]["aproximacion_norm"],
        f"Err rel.: {tabla_err[1]['error_relativo_pct']:.3f}% ({res['mults_ingenua']} mult.)",
        delta_color="off",
    )
    m2.metric(
        "Horner Anidado (Redondeo)",
        tabla_err[3]["aproximacion_norm"],
        f"Err rel.: {tabla_err[3]['error_relativo_pct']:.3f}% ({res['mults_horner']} mult.)",
        delta_color="off",
    )
    m3.metric(
        f"Exacto y Costo (Grado {res['grado']})",
        f"{res['f_exacto']:.6g}",
        f"Ahorro {res['ahorro_mult_pct']:.0f}% multiplicaciones (O(n) vs O(n²))",
        delta_color="off",
    )

    k_vis = (
        render_slider_paso_k("Ver hasta el esquema $k$:", len(tabla_err), key="vis_horner_k")
        if modo_detallado
        else len(tabla_err)
    )
    sub_h = tabla_err[:k_vis]
    fig = crear_figura_base_plotly(
        f"Error Relativo (%): Estándar vs Horner (k = {k_dig})",
        "Esquema evaluado",
        "Error Relativo (%)",
        altura=350,
    )
    fig.add_trace(
        go.Bar(
            x=[r["metodo"] for r in sub_h],
            y=[r["error_relativo_pct"] for r in sub_h],
            text=[f"{r['error_relativo_pct']:.2f}%" for r in sub_h],
            textposition="auto",
            marker_color=[
                PALETA_ARANDU["terracota"] if "Estándar" in r["metodo"] else PALETA_ARANDU["salvia"]
                for r in sub_h
            ],
        )
    )
    render_plotly_arandu(fig, key="plot_fund_horner")

    registrar_contexto_ia(
        "Resultados Esquema de Horner vs Estándar",
        f"- **Punto $x = {x_val:g}$ | Dígitos $k = {k_dig}$ | Coeficientes:** `{coefs_curr}`\n"
        f"- **Valor exacto:** `{res['f_exacto']:.6f}` | **Estándar redondeo:** `{tabla_err[1]['aproximacion_norm']}` "
        f"(`{tabla_err[1]['error_relativo_pct']:.3f}%`) | **Horner redondeo:** `{tabla_err[3]['aproximacion_norm']}` "
        f"(`{tabla_err[3]['error_relativo_pct']:.3f}%`)",
        clave="fund_horner_res",
    )

    if modo_detallado:
        pasos_horner_vivos = [
            f"**Factorización anidada:** Grado $n = {res['grado']}$ con `{coefs_curr}` en $x = {x_val:g}$ ($k = {k_dig}$ dígitos, exacto ${res['f_exacto']:.6f}$). Horner baja el costo de ${res['mults_ingenua']}$ a ${res['mults_horner']}$ multiplicaciones."
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
            expanded=True,
        )
        render_dataframe_acotado(_construir_filas_horner(tabla_err), max_filas=300, key_prefix="tbl_fund_horner")


def _tab_horner() -> None:
    modo_detallado = render_toggle_modo_detallado("horner")
    meta = render_bloque_1_teoria_y_ejemplo("horner", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    coefs_def = [float(c.strip()) for c in str(raw_def.get("coefs_str", "1, -6, 3, -0.149")).split(",") if c.strip()]
    defaults = {
        "x_eval": float(raw_def.get("x_eval", 4.71)),
        "k_digitos": int(raw_def.get("k_digitos", 3)),
        "coefs": coefs_def or [1.0, -6.0, 3.0, -0.149],
    }
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_horner(defaults, modo_detallado=modo_detallado)


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
