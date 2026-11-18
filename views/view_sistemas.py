"""
Vista de Sistemas de Ecuaciones Lineales y Autovalores - Arandú.
Gauss, Gauss-Jordan, LU (Doolittle), Gauss-Seidel, Potencias (Max/Min) y Faddeev-Leverrier.
Separación Static-First con aislamiento mediante @st.fragment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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
    registrar_contexto_ia,
    render_bloque_1_teoria_y_ejemplo,
    render_data_editor_arandu,
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


_A_CLUSTER_GAUSS = [
    [1.0, -0.20, -0.10],
    [-0.35, 1.0, -0.15],
    [-0.25, -0.30, 1.0],
]


def _aplicar_preset_sistemas(valores: dict[str, object], claves_ui: tuple[str, ...]) -> None:
    """Actualiza parámetros en session_state y limpia claves de widgets UI asociados."""
    for clave, valor in valores.items():
        st.session_state[clave] = valor
    for clave_ui in claves_ui:
        st.session_state.pop(clave_ui, None)


def _parsear_fila_aumentada(texto: str, n_cols: int) -> tuple[list[float], float]:
    """Convierte una fila 'a1, a2, a3 | b' o 'a1, a2, a3, b' en coeficientes y término b."""
    limpio = (texto or "").replace("|", ",").replace(";", ",")
    partes = [float(p.strip()) for p in limpio.split(",") if p.strip()]
    if len(partes) != n_cols + 1:
        raise ValueError(f"Se esperan {n_cols} coeficientes + 1 término independiente ({n_cols + 1} números); se recibieron {len(partes)}.")
    return partes[:n_cols], partes[-1]


def _matriz_sensor_imu(k: float) -> list[list[float]]:
    """Matriz de acoplamiento cruzado (cross-talk) 3x3 de sensor IMU."""
    kf = float(k)
    return [
        [1.0, kf, 0.5 * kf],
        [0.8 * kf, 1.0, -kf],
        [-0.5 * kf, 0.6 * kf, 1.0],
    ]


_A_RACK_LU = [
    [8.0, -2.0, -1.0],
    [-2.0, 9.0, -2.5],
    [-1.0, -2.5, 10.0],
]
_A_BD_GS = [
    [5.0, 1.0, 1.0],
    [1.0, 6.0, 1.0],
    [1.0, 1.0, 7.0],
]


def _matriz_cascada_potencias(d: float, c: float) -> list[list[float]]:
    """Matriz tridiagonal 3x3 de cascada de reintentos entre Gateway, API y BD."""
    df = float(d)
    cf = float(c)
    return [
        [df, cf, 0.0],
        [cf, df, cf],
        [0.0, cf, df],
    ]


def _parsear_vector_libre(texto: str, n_esperado: int) -> list[float]:
    """Convierte 'x0_1, x0_2, x0_3' en un vector finito de longitud exacta n_esperado."""
    limpio = (texto or "").replace(";", ",").replace("[", "").replace("]", "")
    partes = [float(p.strip()) for p in limpio.split(",") if p.strip()]
    if len(partes) != n_esperado:
        raise ValueError(f"Se esperan exactamente {n_esperado} valores separados por coma; se recibieron {len(partes)}.")
    if not all(np.isfinite(p) for p in partes):
        raise ValueError("Todos los valores ingresados deben ser números reales finitos.")
    return partes


def _construir_tabla_aumentada(matriz: list[list[float]], cols_vars: list[str]) -> list[dict]:
    """Formatea una matriz aumentada [A | b] con encabezados 100% en español."""
    filas = []
    for idx, fila in enumerate(matriz, start=1):
        registro: dict[str, object] = {"Ecuación": f"Fila {idx}"}
        for j, nombre in enumerate(cols_vars):
            registro[f"Col. {nombre}"] = float(fila[j])
        registro["Término b"] = float(fila[-1])
        filas.append(registro)
    return filas


def _construir_tabla_matriz_es(matriz: list[list[float]], cols_vars: list[str]) -> list[dict]:
    """Formatea cualquier matriz cuadrada n×n (A⁻¹, L o U) con encabezados 100% en español."""
    filas = []
    for idx, fila in enumerate(matriz, start=1):
        registro: dict[str, object] = {"Fila": f"Fila {idx}"}
        for j, nombre in enumerate(cols_vars):
            registro[f"Col. {nombre}"] = round(float(fila[j]), 6)
        filas.append(registro)
    return filas


def _crear_heatmap_matriz_arandu(
    titulo: str,
    matriz: list[list[float]],
    cols_vars: list[str],
    altura: int = 310,
) -> go.Figure:
    """Crea un mapa térmico anotado y accesible para matrices A⁻¹, L o U."""
    fig = crear_figura_base_plotly(titulo, "Columna", "Fila", altura=altura)
    n_m = len(matriz)
    fig.add_trace(
        go.Heatmap(
            z=matriz,
            x=[f"Col. {cols_vars[j]}" for j in range(n_m)],
            y=[f"Fila {i+1}" for i in range(n_m)],
            texttemplate="%{z:.4f}",
            colorscale=[
                [0.0, PALETA_ARANDU["terracota"]],
                [0.5, PALETA_ARANDU["bg_elevado"]],
                [1.0, PALETA_ARANDU["oro"]],
            ],
            zmid=0.0,
            hovertemplate="%{y} · %{x}: %{z:.6f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _construir_tabla_ampliada_gj(matriz: list[list[float]], cols_vars: list[str], modo: str) -> list[dict]:
    """Formatea la matriz ampliada [A | b] o [A | I] de Gauss-Jordan en español."""
    if modo == "sistema":
        return _construir_tabla_aumentada(matriz, cols_vars)
    n = len(cols_vars)
    filas = []
    for idx, fila in enumerate(matriz, start=1):
        registro: dict[str, object] = {"Ecuación": f"Fila {idx}"}
        for j, nombre in enumerate(cols_vars):
            registro[f"Izq. {nombre}"] = float(fila[j])
        for j in range(n):
            registro[f"Der. e{j+1}"] = float(fila[n + j])
        filas.append(registro)
    return filas


def _render_editor_sistema_dual(
    clave_state: str,
    ui_prefix: str,
    defaults: dict,
    modo_detallado: bool,
) -> tuple[list[list[float]], list[float], Exception | None]:
    """Renderiza editor matricial dual: filas de texto en Presentación y data_editor en Detallado."""
    if f"{clave_state}_A" not in st.session_state:
        st.session_state[f"{clave_state}_A"] = [
            list(r) for r in defaults.get("A", [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]])
        ]
    if f"{clave_state}_b" not in st.session_state:
        st.session_state[f"{clave_state}_b"] = list(defaults.get("b", [121.28, 126.321, 118.522]))

    if modo_detallado:
        n_m = len(st.session_state[f"{clave_state}_A"])
        A, b, _ = _editor_sistema(
            clave_state,
            solo_lectura=False,
            cols_vars=[f"Coef. x{i+1}" for i in range(n_m)],
            col_b="Término b",
        )
        return A, b, None

    A_cur = st.session_state[f"{clave_state}_A"]
    b_cur = st.session_state[f"{clave_state}_b"]
    n_m = len(A_cur)
    cols_r = st.columns(n_m)
    A_tmp, b_tmp = [], []
    error_parseo: Exception | None = None
    for idx_r in range(n_m):
        def_txt = ", ".join(f"{v:g}" for v in A_cur[idx_r]) + f" | {b_cur[idx_r]:g}"
        txt_r = cols_r[idx_r].text_input(
            f"Ecuación {idx_r + 1} [aᵢ₁, aᵢ₂, aᵢ₃ | bᵢ]:",
            value=def_txt,
            key=f"{ui_prefix}_row_{idx_r}",
        )
        try:
            fila_a, val_b = _parsear_fila_aumentada(txt_r, n_m)
            A_tmp.append(fila_a)
            b_tmp.append(val_b)
        except Exception as exc:
            error_parseo = exc
            break
    if error_parseo is None:
        st.session_state[f"{clave_state}_A"] = A_tmp
        st.session_state[f"{clave_state}_b"] = b_tmp
        return A_tmp, b_tmp, None
    return A_cur, b_cur, error_parseo


def _editor_sistema(
    clave_state: str,
    solo_lectura: bool,
    incluir_x0: bool = False,
    cols_vars: list[str] | None = None,
    col_b: str = "b",
    col_x0: str = "x0",
) -> tuple[list[list[float]], list[float], list[float] | None]:
    A = st.session_state[f"{clave_state}_A"]
    b = st.session_state[f"{clave_state}_b"]
    n = len(A)
    cols_A = cols_vars if (cols_vars is not None and len(cols_vars) == n) else [f"a_{i+1}" for i in range(n)]
    df_mat = pd.DataFrame(A, columns=cols_A)
    df_mat[col_b] = b
    if incluir_x0:
        x0 = st.session_state[f"{clave_state}_x0"]
        df_mat[col_x0] = x0

    df_edit = render_data_editor_arandu(
        df_mat,
        use_container_width=True,
        hide_index=True,
        disabled=solo_lectura,
        key=f"{clave_state}__editor_df",
    )
    if not solo_lectura:
        A = df_edit[cols_A].astype(float).values.tolist()
        b = df_edit[col_b].astype(float).tolist()
        st.session_state[f"{clave_state}_A"] = A
        st.session_state[f"{clave_state}_b"] = b
        if incluir_x0:
            x0 = df_edit[col_x0].astype(float).tolist()
            st.session_state[f"{clave_state}_x0"] = x0
            return A, b, x0
    return A, b, (st.session_state[f"{clave_state}_x0"] if incluir_x0 else None)


@st.fragment
def _frag_gauss(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_cluster", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Balanceo de Tráfico en Clúster Cloud (req/s)"
            if op == "real_cluster"
            else "Sistema Libre 3×3 y Cátedra TP4 Ej. 5"
        ),
        horizontal=True,
        key="ui_gauss_contexto",
    )
    es_caso_real = contexto == "real_cluster"
    if "sis_gauss_piv" not in st.session_state:
        st.session_state["sis_gauss_piv"] = True

    if es_caso_real:
        st.caption(
            r"Malla de 3 nodos (`Gateway`, `Auth`, `Billing/DB`): modificá el tráfico externo entrante $d_i$ (req/s) "
            r"para calcular la carga total real de cada servicio tras los reenvíos internos."
        )
        if "sis_gauss_d1" not in st.session_state:
            st.session_state["sis_gauss_d1"] = 500.0
        if "sis_gauss_d2" not in st.session_state:
            st.session_state["sis_gauss_d2"] = 120.0
        if "sis_gauss_d3" not in st.session_state:
            st.session_state["sis_gauss_d3"] = 80.0

        claves_ui_real = ("ui_gauss_d1", "ui_gauss_d2", "ui_gauss_d3", "ui_gauss_piv")
        b1, b2, b3 = st.columns(3)
        if b1.button("Carga normal (500 / 120 / 80 req/s)", key="btn_gauss_normal", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gauss_d1": 500.0, "sis_gauss_d2": 120.0, "sis_gauss_d3": 80.0, "sis_gauss_piv": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button("Pico Black Friday (1250 / 300 / 150)", key="btn_gauss_pico", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gauss_d1": 1250.0, "sis_gauss_d2": 300.0, "sis_gauss_d3": 150.0, "sis_gauss_piv": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button("Presión fuerte DB (200 / 180 / 650)", key="btn_gauss_db", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gauss_d1": 200.0, "sis_gauss_d2": 180.0, "sis_gauss_d3": 650.0, "sis_gauss_piv": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns(3)
        d1_val = float(c1.number_input("d₁ API Gateway (req/s):", min_value=0.0, value=float(st.session_state["sis_gauss_d1"]), step=20.0, key="ui_gauss_d1"))
        d2_val = float(c2.number_input("d₂ Auth (req/s):", min_value=0.0, value=float(st.session_state["sis_gauss_d2"]), step=20.0, key="ui_gauss_d2"))
        d3_val = float(c3.number_input("d₃ Billing/DB (req/s):", min_value=0.0, value=float(st.session_state["sis_gauss_d3"]), step=20.0, key="ui_gauss_d3"))
        usar_pivoteo = bool(st.checkbox("Activar pivoteo parcial", value=bool(st.session_state["sis_gauss_piv"]), key="ui_gauss_piv"))
        st.session_state["sis_gauss_d1"] = d1_val
        st.session_state["sis_gauss_d2"] = d2_val
        st.session_state["sis_gauss_d3"] = d3_val
        st.session_state["sis_gauss_piv"] = usar_pivoteo
        A = _A_CLUSTER_GAUSS
        b = [d1_val, d2_val, d3_val]
        etiquetas_vars = ["Gateway", "Auth", "Billing/DB"]
        error_parseo = None
    else:
        if "sis_gauss_A" not in st.session_state:
            st.session_state["sis_gauss_A"] = [list(r) for r in defaults.get("A", [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]])]
        if "sis_gauss_b" not in st.session_state:
            st.session_state["sis_gauss_b"] = list(defaults.get("b", [121.28, 126.321, 118.522]))

        claves_ui_lib = ("ui_gauss_row_0", "ui_gauss_row_1", "ui_gauss_row_2", "sis_gauss__editor_df", "ui_gauss_piv")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP4 Ej. 5 (Cátedra 3×3)", key="btn_gauss_tp4", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_gauss_A": [list(r) for r in defaults.get("A", [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]])],
                    "sis_gauss_b": list(defaults.get("b", [121.28, 126.321, 118.522])),
                    "sis_gauss_piv": True,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Trampa pivote 10⁻¹³ (probá Pivoteo ON vs OFF)", key="btn_gauss_piv_trap", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_gauss_A": [[1e-13, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 1.0, 1.0]],
                    "sis_gauss_b": [2.0 + 1e-13, 6.0, 4.0],
                    "sis_gauss_piv": False,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        usar_pivoteo = bool(st.checkbox("Activar pivoteo parcial", value=bool(st.session_state["sis_gauss_piv"]), key="ui_gauss_piv"))
        st.session_state["sis_gauss_piv"] = usar_pivoteo
        A, b, error_parseo = _render_editor_sistema_dual("sis_gauss", "ui_gauss", defaults, modo_detallado)
        etiquetas_vars = [f"x_{i+1}" for i in range(len(A))]

    try:
        if error_parseo is not None:
            raise error_parseo
        res = eliminacion_gauss(A, b, pivoteo=usar_pivoteo)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A y vector b (Eliminación de Gauss)",
            sugerencia_defecto="Verificá formato numérico válido (N coefs | b), que A sea cuadrada no singular y sin pivotes nulos insalvables.",
            ejemplo_defecto="18.72, 8.2, 8.76 | 121.28",
        )
        return

    sol = res["solucion"]
    m1, m2, m3 = st.columns(3)
    if es_caso_real:
        m1.metric(
            "Carga Total Gateway (x₁)",
            f"{sol[0]:.2f} req/s",
            f"Externo d₁ = {b[0]:g} req/s",
            delta_color="off",
        )
        m2.metric(
            "Auth (x₂) / Billing (x₃)",
            f"{sol[1]:.1f} / {sol[2]:.1f}",
            f"req/s ({res['intercambios']} pivoteos)",
            delta_color="off",
        )
    else:
        m1.metric(
            "Incógnitas x₁ / x₂",
            f"{sol[0]:.4f} / {sol[1]:.4f}" if len(sol) >= 2 else f"{sol[0]:.4f}",
            f"x₃ = {sol[2]:.4f}" if len(sol) == 3 else f"{len(sol)} variables",
            delta_color="off",
        )
        m2.metric(
            "Pasos de Escalonamiento",
            f"{len(res['pasos_escalonamiento']) - 1} columnas",
            f"Pivoteo {'ON' if usar_pivoteo else 'OFF'} ({res['intercambios']} swaps)",
            delta_color="off",
        )
    m3.metric(
        "Residuo ‖Ax − b‖₂",
        f"{res['norma_error']:.2e}",
        f"{'Precisión máquina' if res['norma_error'] < 1e-10 else 'Error numérico elevado (sin pivoteo)'}",
        delta_color="off",
    )

    sol_txt = ", ".join(f"{etiquetas_vars[i]}={v:.6f}" for i, v in enumerate(sol))
    registrar_contexto_ia(
        "Solución Eliminación de Gauss",
        f"- **Escenario:** {'Clúster Cloud' if es_caso_real else 'Libre / TP4 Ej. 5'} | **Pivoteo parcial:** `{usar_pivoteo}`\n"
        f"- **Vector solución:** `[{sol_txt}]`\n- **Norma residual $\\|Ax - b\\|_2$:** `{res['norma_error']:.3e}`",
        clave="sis_gauss_res",
    )

    fig = crear_figura_base_plotly(
        "Carga Externa dᵢ vs Carga Total Solucionada xᵢ (req/s)" if es_caso_real else "Solución xᵢ del Sistema Lineal",
        "Servicio del Clúster" if es_caso_real else "Componente",
        "Tráfico (req/s)" if es_caso_real else "Valor xᵢ",
        altura=330,
    )
    if es_caso_real:
        fig.add_trace(
            go.Bar(
                x=etiquetas_vars,
                y=b,
                name="Tráfico externo dᵢ",
                marker_color=PALETA_ARANDU["salvia"],
            )
        )
        fig.add_trace(
            go.Bar(
                x=etiquetas_vars,
                y=sol,
                name="Tráfico total xᵢ",
                marker_color=PALETA_ARANDU["oro"],
                marker_line_color=PALETA_ARANDU["terracota_oscuro"],
                marker_line_width=1.2,
            )
        )
        fig.update_layout(barmode="group")
    else:
        fig.add_trace(
            go.Bar(
                x=etiquetas_vars,
                y=sol,
                name="Solución x",
                marker_color=PALETA_ARANDU["oro"],
                marker_line_color=PALETA_ARANDU["terracota_oscuro"],
                marker_line_width=1.2,
            )
        )
    render_plotly_arandu(fig, key="plot_sis_gauss")

    if modo_detallado:
        pasos = res["pasos_escalonamiento"]
        pasos_gauss_vivos = [
            "**Armado del sistema aumentado $[A \\mid b]$:** Aplicamos eliminación progresiva con pivoteo parcial columna por columna para triangular la matriz sin cambiar la solución."
        ]
        for p_e in pasos[1:]:
            ops = "; ".join(f"`{op}`" for op in p_e.get("operaciones", [])) or "Sin intercambio ni eliminación requerida."
            pasos_gauss_vivos.append(f"**{p_e['descripcion']}:** {ops}")
        for p_s in reversed(res.get("pasos_sustitucion", [])):
            pasos_gauss_vivos.append(
                f"**Sustitución regresiva (`{p_s['variable']}`):** `{p_s['ecuacion']}` $= {p_s['valor']:.6f}$."
            )
        render_deduccion_paso_a_paso_viva(
            "Eliminación Gaussiana y Sustitución Regresiva",
            pasos_gauss_vivos,
            clave_ia="sis_gauss_viva",
            expanded=True,
        )
        if pasos:
            k_vis = render_slider_paso_k("Ver paso de escalonamiento $k$:", len(pasos), key="vis_gauss_k")
            paso = pasos[k_vis - 1]
            st.info(paso["descripcion"])
            render_dataframe_acotado(
                _construir_tabla_aumentada(paso["matriz"], etiquetas_vars),
                max_filas=300,
                key_prefix="tbl_sis_gauss",
                hide_index=True,
            )


def _tab_gauss() -> None:
    modo_detallado = render_toggle_modo_detallado("eliminacion_gauss")
    meta = render_bloque_1_teoria_y_ejemplo("eliminacion_gauss", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
        },
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_gauss(defaults, modo_detallado=modo_detallado)


@st.fragment
def _frag_gauss_jordan(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_sensor", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Calibración e Inversa de Sensor IMU 3-Ejes (A⁻¹)"
            if op == "real_sensor"
            else "Sistema / Inversa Libre 3×3 y Cátedra TP4 Ej. 6"
        ),
        horizontal=True,
        key="ui_gj_contexto",
    )
    es_caso_real = contexto == "real_sensor"
    if "sis_gj_piv" not in st.session_state:
        st.session_state["sis_gj_piv"] = True

    c_m, c_p = st.columns([1.8, 1.2])
    modo_gj = str(
        c_m.radio(
            "Objetivo de Gauss-Jordan:",
            ["Calcular Inversa A⁻¹", "Resolver Ax = b"],
            horizontal=True,
            key="ui_gj_modo",
        )
    )
    usar_pivoteo = bool(c_p.checkbox("Activar pivoteo parcial", value=bool(st.session_state["sis_gj_piv"]), key="ui_gj_piv"))
    st.session_state["sis_gj_piv"] = usar_pivoteo

    if es_caso_real:
        st.caption(
            r"Desacoplamiento de sensor IMU 3-ejes ($\vec{y} = A\,\vec{x}$): regulá el factor de acoplamiento cruzado $k$ "
            r"para calcular la matriz de calibración inversa $A^{-1}$ o reconstruir la lectura real $\vec{x}$."
        )
        if "sis_gj_k" not in st.session_state:
            st.session_state["sis_gj_k"] = 0.12
        if "sis_gj_y1" not in st.session_state:
            st.session_state["sis_gj_y1"] = 1.0
        if "sis_gj_y2" not in st.session_state:
            st.session_state["sis_gj_y2"] = 2.0
        if "sis_gj_y3" not in st.session_state:
            st.session_state["sis_gj_y3"] = -0.5

        claves_ui_real = ("ui_gj_k", "ui_gj_y1", "ui_gj_y2", "ui_gj_y3", "ui_gj_piv")
        b1, b2, b3 = st.columns(3)
        if b1.button("Desalineación típica (k = 0.12)", key="btn_gj_imu_tipico", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gj_k": 0.12, "sis_gj_y1": 1.0, "sis_gj_y2": 2.0, "sis_gj_y3": -0.5, "sis_gj_piv": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button("Fuerte cross-talk (k = 0.45)", key="btn_gj_imu_alto", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gj_k": 0.45, "sis_gj_y1": 1.0, "sis_gj_y2": 2.0, "sis_gj_y3": -0.5, "sis_gj_piv": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button("Casi ortogonal (k = 0.02)", key="btn_gj_imu_orto", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gj_k": 0.02, "sis_gj_y1": 1.0, "sis_gj_y2": 2.0, "sis_gj_y3": -0.5, "sis_gj_piv": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3, c4 = st.columns(4)
        k_val = float(c1.slider("Cross-talk k:", min_value=0.0, max_value=0.60, value=float(st.session_state["sis_gj_k"]), step=0.01, key="ui_gj_k"))
        y1_val = float(c2.number_input("Lectura cruda y₁ (g):", value=float(st.session_state["sis_gj_y1"]), step=0.1, key="ui_gj_y1"))
        y2_val = float(c3.number_input("Lectura cruda y₂ (g):", value=float(st.session_state["sis_gj_y2"]), step=0.1, key="ui_gj_y2"))
        y3_val = float(c4.number_input("Lectura cruda y₃ (g):", value=float(st.session_state["sis_gj_y3"]), step=0.1, key="ui_gj_y3"))
        st.session_state["sis_gj_k"] = k_val
        st.session_state["sis_gj_y1"] = y1_val
        st.session_state["sis_gj_y2"] = y2_val
        st.session_state["sis_gj_y3"] = y3_val
        A = _matriz_sensor_imu(k_val)
        b = [y1_val, y2_val, y3_val]
        etiquetas_vars = ["Eje X", "Eje Y", "Eje Z"]
        error_parseo = None
    else:
        claves_ui_lib = ("ui_gj_row_0", "ui_gj_row_1", "ui_gj_row_2", "sis_gj__editor_df", "ui_gj_piv")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP4 Ej. 6 (Cátedra 3×3)", key="btn_gj_tp4", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_gj_A": [list(r) for r in defaults.get("A", [[18.72, 8.2, 8.76], [6.4, 15.9, 7.18], [5.22, 6.4, 14.31]])],
                    "sis_gj_b": list(defaults.get("b", [121.28, 126.321, 118.522])),
                    "sis_gj_piv": True,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Ejemplo teórico 3×3 (x = [1, 2, 3]ᵀ)", key="btn_gj_ej_teorico", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_gj_A": [[2.0, -2.0, 5.0], [2.0, 3.0, 4.0], [3.0, -1.0, 3.0]],
                    "sis_gj_b": [13.0, 20.0, 10.0],
                    "sis_gj_piv": True,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        A, b, error_parseo = _render_editor_sistema_dual("sis_gj", "ui_gj", defaults, modo_detallado)
        etiquetas_vars = [f"x_{i+1}" for i in range(len(A))]

    try:
        if error_parseo is not None:
            raise error_parseo
        res = gauss_jordan(A, b if modo_gj.startswith("Resolver") else None, pivoteo=usar_pivoteo)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A (Gauss-Jordan)",
            sugerencia_defecto="La matriz A debe ser cuadrada, finita e invertible (sin pivotes nulos tras pivoteo).",
            ejemplo_defecto="A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]",
        )
        return

    m1, m2, m3 = st.columns(3)
    if res["modo"] == "sistema":
        sol = res["solucion"]
        if es_caso_real:
            m1.metric("Eje X Calibrado (x₁)", f"{sol[0]:.4f} g", f"Crudo y₁ = {b[0]:g} g", delta_color="off")
            m2.metric("Ejes Y / Z Calibrados", f"{sol[1]:.3f} / {sol[2]:.3f}", f"{res['intercambios']} pivoteos", delta_color="off")
        else:
            m1.metric(
                "Incógnitas x₁ / x₂",
                f"{sol[0]:.4f} / {sol[1]:.4f}" if len(sol) >= 2 else f"{sol[0]:.4f}",
                f"x₃ = {sol[2]:.4f}" if len(sol) == 3 else f"{len(sol)} variables",
                delta_color="off",
            )
            m2.metric(
                "Columnas Reducidas [I | x]",
                f"{len(res['pasos']) - 1} columnas",
                f"Pivoteo {'ON' if usar_pivoteo else 'OFF'} ({res['intercambios']} swaps)",
                delta_color="off",
            )
        m3.metric(
            "Residuo ‖Ax − b‖₂",
            f"{res['norma_error']:.2e}",
            f"{'Precisión máquina' if res['norma_error'] < 1e-10 else 'Error elevado'}",
            delta_color="off",
        )
        sol_txt = ", ".join(f"{etiquetas_vars[i]}={v:.6f}" for i, v in enumerate(sol))
        registrar_contexto_ia(
            "Solución Gauss-Jordan",
            f"- **Escenario:** {'Sensor IMU' if es_caso_real else 'Libre / TP4 Ej. 6'} | **Vector solución:** `[{sol_txt}]`\n"
            f"- **Norma residual:** `{res['norma_error']:.3e}`",
            clave="sis_gj_res",
        )
        fig = crear_figura_base_plotly(
            "Lectura Cruda yᵢ vs Calibrada xᵢ (g)" if es_caso_real else "Solución Directa [I | x] (Gauss-Jordan)",
            "Eje del Sensor" if es_caso_real else "Componente",
            "Aceleración (g)" if es_caso_real else "Valor xᵢ",
            altura=330,
        )
        if es_caso_real:
            fig.add_trace(go.Bar(x=etiquetas_vars, y=b, name="Lectura cruda yᵢ", marker_color=PALETA_ARANDU["salvia"]))
            fig.add_trace(
                go.Bar(
                    x=etiquetas_vars,
                    y=sol,
                    name="Lectura calibrada xᵢ",
                    marker_color=PALETA_ARANDU["oro"],
                    marker_line_color=PALETA_ARANDU["terracota_oscuro"],
                    marker_line_width=1.2,
                )
            )
            fig.update_layout(barmode="group")
        else:
            fig.add_trace(
                go.Bar(
                    x=etiquetas_vars,
                    y=sol,
                    name="Solución x",
                    marker_color=PALETA_ARANDU["oro"],
                    marker_line_color=PALETA_ARANDU["terracota_oscuro"],
                    marker_line_width=1.2,
                )
            )
        render_plotly_arandu(fig, key="plot_sis_gj")
    else:
        inv_mat = res["inversa"]
        traza = sum(inv_mat[idx][idx] for idx in range(len(inv_mat)))
        max_inv = max(abs(val) for fila in inv_mat for val in fila)
        m1.metric(
            "Error Inversión ‖A·A⁻¹ − I‖₂",
            f"{res['norma_error']:.2e}",
            "Precisión máquina" if res["norma_error"] < 1e-10 else "Error elevado",
            delta_color="off",
        )
        m2.metric(
            "Traza Tr(A⁻¹)",
            f"{traza:.4f}",
            f"{len(res['pasos']) - 1} columnas · {res['intercambios']} swaps",
            delta_color="off",
        )
        m3.metric(
            "Elemento Máx |A⁻¹|",
            f"{max_inv:.4f}",
            f"Matriz inversa {len(inv_mat)}×{len(inv_mat)}",
            delta_color="off",
        )
        registrar_contexto_ia(
            "Error Inversión Gauss-Jordan",
            f"- **Escenario:** {'Sensor IMU' if es_caso_real else 'Libre / TP4 Ej. 6'} | **Error $\\|A \\cdot A^{{-1}} - I\\|_2$:** `{res['norma_error']:.3e}`\n"
            f"- **Traza Tr(A⁻¹):** `{traza:.4f}` | **Máximo |A⁻¹|:** `{max_inv:.4f}`",
            clave="sis_gj_res",
        )
        fig = _crear_heatmap_matriz_arandu(
            "Mapa Térmico de Matriz Inversa A⁻¹ (Desacoplamiento)" if es_caso_real else "Mapa Térmico de Matriz Inversa A⁻¹",
            inv_mat,
            etiquetas_vars,
            altura=340,
        )
        render_plotly_arandu(fig, key="plot_sis_gj")

    if modo_detallado:
        if res["modo"] == "inversa":
            st.markdown("#### Matriz Inversa Explícita $A^{-1}$")
            render_dataframe_acotado(
                _construir_tabla_matriz_es(res["inversa"], etiquetas_vars),
                max_filas=300,
                key_prefix="tbl_sis_gj_inv",
                hide_index=True,
            )
        pasos = res["pasos"]
        pasos_gj_vivos = [
            f"**Objetivo activo (`{modo_gj}`):** Reducimos el bloque ampliado hasta llevar la submatriz izquierda a la matriz identidad $I$ mediante pivoteo, normalización y eliminación superior e inferior."
        ]
        for p_j in pasos[1:]:
            ops = "; ".join(f"`{op}`" for op in p_j.get("operaciones", [])) or "Columna ya diagonalizada."
            pasos_gj_vivos.append(f"**{p_j['descripcion']}:** {ops}")
        if res["modo"] == "sistema":
            pasos_gj_vivos.append(
                f"**Lectura directa de $x$:** `[{', '.join(f'{v:.6f}' for v in res['solucion'])}]` con residuo $\\|Ax - b\\|_2 = {res['norma_error']:.3e}$."
            )
        else:
            pasos_gj_vivos.append(
                f"**Verificación de $A^{{-1}}$:** $\\|A\\cdot A^{{-1}} - I\\|_2 = {res['norma_error']:.3e}$."
            )
        render_deduccion_paso_a_paso_viva(
            f"Gauss-Jordan ({modo_gj})",
            pasos_gj_vivos,
            clave_ia="sis_gj_viva",
            expanded=True,
        )
        if pasos:
            k_vis = render_slider_paso_k("Ver paso de pivoteo $k$:", len(pasos), key="vis_gj_k")
            paso = pasos[k_vis - 1]
            st.info(paso["descripcion"])
            render_dataframe_acotado(
                _construir_tabla_ampliada_gj(paso["matriz"], etiquetas_vars, res["modo"]),
                max_filas=300,
                key_prefix="tbl_sis_gj_paso",
                hide_index=True,
            )


def _tab_gauss_jordan() -> None:
    modo_detallado = render_toggle_modo_detallado("gauss_jordan")
    meta = render_bloque_1_teoria_y_ejemplo("gauss_jordan", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
        },
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_gauss_jordan(defaults, modo_detallado=modo_detallado)


@st.fragment
def _frag_lu(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_termica", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Malla Térmica 3-Racks (L y U fijos, cambia carga q)"
            if op == "real_termica"
            else "Sistema Libre 3×3 y Cátedra TP4 Ej. 7"
        ),
        horizontal=True,
        key="ui_lu_contexto",
    )
    es_caso_real = contexto == "real_termica"

    if es_caso_real:
        st.caption(
            r"Matriz de conductancias térmicas fija $A = L \cdot U$: cambiá solo el perfil de potencia $\vec{q}$ de los racks "
            r"(`Web`, `DB`, `GPU-IA`) para recalcular sobretemperaturas $\Delta T_i$ (°C) con los mismos factores $L$ y $U$ en $O(n^2)$."
        )
        if "sis_lu_q1" not in st.session_state:
            st.session_state["sis_lu_q1"] = 180.0
        if "sis_lu_q2" not in st.session_state:
            st.session_state["sis_lu_q2"] = 220.0
        if "sis_lu_q3" not in st.session_state:
            st.session_state["sis_lu_q3"] = 150.0

        claves_ui_real = ("ui_lu_q1", "ui_lu_q2", "ui_lu_q3")
        b1, b2, b3 = st.columns(3)
        if b1.button("Carga base (180 / 220 / 150)", key="btn_lu_base", use_container_width=True):
            _aplicar_preset_sistemas({"sis_lu_q1": 180.0, "sis_lu_q2": 220.0, "sis_lu_q3": 150.0}, claves_ui_real)
            rerun_fragmento_seguro()
        if b2.button("Batch nocturno DB (100 / 380 / 160)", key="btn_lu_db", use_container_width=True):
            _aplicar_preset_sistemas({"sis_lu_q1": 100.0, "sis_lu_q2": 380.0, "sis_lu_q3": 160.0}, claves_ui_real)
            rerun_fragmento_seguro()
        if b3.button("Pico IA GPU (200 / 250 / 520)", key="btn_lu_gpu", use_container_width=True):
            _aplicar_preset_sistemas({"sis_lu_q1": 200.0, "sis_lu_q2": 250.0, "sis_lu_q3": 520.0}, claves_ui_real)
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns(3)
        q1 = float(c1.number_input("Carga térmica q₁ Web:", min_value=0.0, value=float(st.session_state["sis_lu_q1"]), step=10.0, key="ui_lu_q1"))
        q2 = float(c2.number_input("Carga térmica q₂ DB:", min_value=0.0, value=float(st.session_state["sis_lu_q2"]), step=10.0, key="ui_lu_q2"))
        q3 = float(c3.number_input("Carga térmica q₃ GPU-IA:", min_value=0.0, value=float(st.session_state["sis_lu_q3"]), step=10.0, key="ui_lu_q3"))
        st.session_state["sis_lu_q1"] = q1
        st.session_state["sis_lu_q2"] = q2
        st.session_state["sis_lu_q3"] = q3
        A = _A_RACK_LU
        b = [q1, q2, q3]
        etiquetas_vars = ["Rack Web", "Rack DB", "Rack GPU-IA"]
        error_parseo = None
    else:
        claves_ui_lib = ("ui_lu_row_0", "ui_lu_row_1", "ui_lu_row_2", "sis_lu__editor_df")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP4 Ej. 7 (Cátedra 3×3)", key="btn_lu_tp4", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_lu_A": [list(r) for r in defaults.get("A", [[4.0, -2.0, 1.0], [20.0, -7.0, 12.0], [-8.0, 13.0, 17.0]])],
                    "sis_lu_b": list(defaults.get("b", [11.0, 70.0, 17.0])),
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Falla Doolittle (U₁₁ = 0): pivote nulo sin permutación", key="btn_lu_cero_piv", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_lu_A": [[0.0, 2.0, 1.0], [2.0, 1.0, 3.0], [1.0, -1.0, 2.0]],
                    "sis_lu_b": [5.0, 11.0, 4.0],
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        A, b, error_parseo = _render_editor_sistema_dual("sis_lu", "ui_lu", defaults, modo_detallado)
        etiquetas_vars = [f"x_{i+1}" for i in range(len(A))]

    try:
        if error_parseo is not None:
            raise error_parseo
        res = factorizacion_lu(A, b)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A y vector b (Factorización LU)",
            sugerencia_defecto="Doolittle directo requiere menores principales no nulos (u_kk ≠ 0); reordená filas si u₁₁ = 0.",
            ejemplo_defecto="A = [[4, -2, 1], [20, -7, 12], [-8, 13, 17]]",
        )
        return

    sol = res["solucion"]
    y_vec = res["y"]
    m1, m2, m3 = st.columns(3)
    if es_caso_real:
        m1.metric(
            "Sobretemp. Pico ΔT",
            f"{max(sol):.2f} °C",
            f"Web {sol[0]:.1f} · DB {sol[1]:.1f} · GPU {sol[2]:.1f} °C",
            delta_color="off",
        )
        m2.metric(
            "Vector Progresivo y (Ly = q)",
            f"{y_vec[0]:.1f} / {y_vec[1]:.1f}",
            f"y₃ = {y_vec[2]:.1f} · Barrido O(n²)",
            delta_color="off",
        )
    else:
        m1.metric(
            "Solución x (Ux = y)",
            f"{sol[0]:.4f} / {sol[1]:.4f}" if len(sol) >= 2 else f"{sol[0]:.4f}",
            f"x₃ = {sol[2]:.4f}" if len(sol) == 3 else f"{len(sol)} incógnitas",
            delta_color="off",
        )
        m2.metric(
            "Vector Auxiliar y (Ly = b)",
            f"{y_vec[0]:.3f} / {y_vec[1]:.3f}" if len(y_vec) >= 2 else f"{y_vec[0]:.3f}",
            f"y₃ = {y_vec[2]:.3f}" if len(y_vec) == 3 else "Sustitución progresiva",
            delta_color="off",
        )
    m3.metric(
        "Error ‖L·U − A‖₂",
        f"{res['error_LU']:.2e}",
        f"Residuo ‖Ax − b‖ = {res['norma_error']:.2e}",
        delta_color="off",
    )

    sol_lu = ", ".join(f"{etiquetas_vars[i]}={v:.6f}" for i, v in enumerate(sol))
    registrar_contexto_ia(
        "Solución Factorización LU (Doolittle)",
        f"- **Escenario:** {'Malla Térmica 3-Racks' if es_caso_real else 'Libre / TP4 Ej. 7'}\n"
        f"- **Vector solución:** `[{sol_lu}]` | **Error $\\|LU - A\\|$:** `{res['error_LU']:.2e}`",
        clave="sis_lu_res",
    )

    c_h1, c_h2 = st.columns(2)
    with c_h1:
        render_plotly_arandu(
            _crear_heatmap_matriz_arandu("Factor L (Triangular Inferior)", res["L"], etiquetas_vars, altura=270),
            key="plot_sis_lu_l",
        )
    with c_h2:
        render_plotly_arandu(
            _crear_heatmap_matriz_arandu("Factor U (Triangular Superior)", res["U"], etiquetas_vars, altura=270),
            key="plot_sis_lu_u",
        )

    fig_sol = crear_figura_base_plotly(
        "Perfil Térmico ΔTᵢ (°C) por Rack" if es_caso_real else "Barrido Progresivo y (Ly=b) vs Solución x (Ux=y)",
        "Rack del Data Center" if es_caso_real else "Componente",
        "Sobretemperatura ΔT (°C)" if es_caso_real else "Valor",
        altura=290,
    )
    if es_caso_real:
        fig_sol.add_trace(
            go.Bar(
                x=etiquetas_vars,
                y=sol,
                name="ΔT sobre ambiente (°C)",
                marker_color=PALETA_ARANDU["oro"],
                marker_line_color=PALETA_ARANDU["terracota_oscuro"],
                marker_line_width=1.2,
            )
        )
    else:
        fig_sol.add_trace(go.Bar(x=etiquetas_vars, y=y_vec, name="Vector intermedio y", marker_color=PALETA_ARANDU["salvia"]))
        fig_sol.add_trace(
            go.Bar(
                x=etiquetas_vars,
                y=sol,
                name="Solución final x",
                marker_color=PALETA_ARANDU["oro"],
                marker_line_color=PALETA_ARANDU["terracota_oscuro"],
                marker_line_width=1.2,
            )
        )
        fig_sol.update_layout(barmode="group")
    render_plotly_arandu(fig_sol, key="plot_sis_lu")

    if modo_detallado:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matriz Triangular Inferior $L$:**")
            render_dataframe_acotado(
                _construir_tabla_matriz_es(res["L"], etiquetas_vars),
                max_filas=300,
                key_prefix="tbl_sis_lu_l",
                hide_index=True,
            )
        with c2:
            st.markdown("**Matriz Triangular Superior $U$:**")
            render_dataframe_acotado(
                _construir_tabla_matriz_es(res["U"], etiquetas_vars),
                max_filas=300,
                key_prefix="tbl_sis_lu_u",
                hide_index=True,
            )

        pasos = res["pasos_eliminacion"]
        pasos_lu_vivos = [
            "**Factorización $A = L \\cdot U$ (Doolittle):** Almacenamos los multiplicadores $L_{ik}$ debajo de la diagonal de $L$ (con $1$ en su diagonal) y el escalonamiento resultante en $U$."
        ]
        for p_el in pasos:
            facts = "; ".join(f"`{f['formula']}`" for f in p_el.get("factores", [])) or "Sin eliminaciones en esta columna."
            pasos_lu_vivos.append(f"**Columna $k = {p_el['columna']}$:** {facts}.")
        for py in res.get("pasos_y", []):
            pasos_lu_vivos.append(
                f"**Sustitución progresiva $Ly = b$ (`{py['variable']}`):** `{py['ecuacion']}` $= {py['valor']:.6f}$."
            )
        for px in reversed(res.get("pasos_x", [])):
            pasos_lu_vivos.append(
                f"**Sustitución regresiva $Ux = y$ (`{px['variable']}`):** `{px['ecuacion']}` $= {px['valor']:.6f}$."
            )
        render_deduccion_paso_a_paso_viva(
            "Factorización LU (Doolittle)",
            pasos_lu_vivos,
            clave_ia="sis_lu_viva",
            expanded=True,
        )
        if pasos:
            k_vis = render_slider_paso_k("Ver columna eliminada $k$:", len(pasos), key="vis_lu_k")
            paso = pasos[k_vis - 1]
            c_l, c_u = st.columns(2)
            with c_l:
                st.caption(f"$L$ tras paso ${k_vis}$")
                render_dataframe_acotado(
                    _construir_tabla_matriz_es(paso["L_actual"], etiquetas_vars),
                    max_filas=300,
                    key_prefix="tbl_sis_lu_kl",
                    hide_index=True,
                )
            with c_u:
                st.caption(f"$U$ tras paso ${k_vis}$")
                render_dataframe_acotado(
                    _construir_tabla_matriz_es(paso["U_actual"], etiquetas_vars),
                    max_filas=300,
                    key_prefix="tbl_sis_lu_ku",
                    hide_index=True,
                )


def _tab_lu() -> None:
    modo_detallado = render_toggle_modo_detallado("factorizacion_lu")
    meta = render_bloque_1_teoria_y_ejemplo("factorizacion_lu", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"A": [[4.0, -2.0, 1.0], [20.0, -7.0, 12.0], [-8.0, 13.0, 17.0]], "b": [11.0, 70.0, 17.0]},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_lu(defaults, modo_detallado=modo_detallado)


def _construir_tabla_gauss_seidel(historial: list[dict], etiquetas_vars: list[str]) -> list[dict]:
    """Formatea la tabla iterativa de Gauss-Seidel con encabezados 100% en español."""
    filas = []
    for h in historial:
        reg: dict[str, object] = {"Iteración k": h["iteracion"]}
        for idx, val in enumerate(h["x"]):
            col_nom = etiquetas_vars[idx] if idx < len(etiquetas_vars) else f"x_{idx+1}"
            reg[col_nom] = round(float(val), 6)
        err_m = h["error_max"]
        res_m = h["norma_residuo"]
        reg["Error máx |Δx|"] = round(float(err_m), 6) if np.isfinite(err_m) else "Inf (diverge)"
        reg["Norma Residuo"] = round(float(res_m), 6) if np.isfinite(res_m) else "Inf (diverge)"
        reg["¿Cumple Tolerancia?"] = "Sí" if h["cumple_tol"] else "No"
        filas.append(reg)
    return filas


@st.fragment
def _frag_gauss_seidel(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_bd", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Latencias en Servidor BD (Consultas c, Updates a, Reportes r)"
            if op == "real_bd"
            else "Sistema Libre 3×3 y Cátedra TP4 Ej. 8"
        ),
        horizontal=True,
        key="ui_gs_contexto",
    )
    es_caso_real = contexto == "real_bd"

    if es_caso_real:
        st.caption(
            r"Tiempos medios de BD acoplados ($5c+a+r=b_1,\; c+6a+r=b_2,\; c+a+7r=b_3$): probá como **intercambiar dos filas** "
            r"mantiene la misma solución $(2, 3, 4)\;\text{ms}$ pero rompe la dominancia diagonal y dispara el error."
        )
        if "sis_gs_real_b1" not in st.session_state:
            st.session_state["sis_gs_real_b1"] = 17.0
        if "sis_gs_real_b2" not in st.session_state:
            st.session_state["sis_gs_real_b2"] = 24.0
        if "sis_gs_real_b3" not in st.session_state:
            st.session_state["sis_gs_real_b3"] = 33.0
        if "sis_gs_real_tol" not in st.session_state:
            st.session_state["sis_gs_real_tol"] = 1e-3
        if "sis_gs_real_swap" not in st.session_state:
            st.session_state["sis_gs_real_swap"] = False

        claves_ui_real = ("ui_gs_real_b1", "ui_gs_real_b2", "ui_gs_real_b3", "ui_gs_real_tol", "ui_gs_real_swap")
        b1, b2, b3 = st.columns(3)
        if b1.button("Ordenado dominante (5 pasos → 2, 3, 4 ms)", key="btn_gs_bd_dom", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gs_real_b1": 17.0, "sis_gs_real_b2": 24.0, "sis_gs_real_b3": 33.0, "sis_gs_real_tol": 1e-3, "sis_gs_real_swap": False},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button("Desordenar filas 1↔2 (sin dominancia → diverge)", key="btn_gs_bd_div", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gs_real_b1": 17.0, "sis_gs_real_b2": 24.0, "sis_gs_real_b3": 33.0, "sis_gs_real_tol": 1e-3, "sis_gs_real_swap": True},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button("Carga BD ×2 (34 / 48 / 66 → 4, 6, 8 ms)", key="btn_gs_bd_pico", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_gs_real_b1": 34.0, "sis_gs_real_b2": 48.0, "sis_gs_real_b3": 66.0, "sis_gs_real_tol": 1e-3, "sis_gs_real_swap": False},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3, c4 = st.columns(4)
        b1_val = float(c1.number_input("Carga eq. Consultas b₁:", min_value=1.0, value=float(st.session_state["sis_gs_real_b1"]), step=1.0, key="ui_gs_real_b1"))
        b2_val = float(c2.number_input("Carga eq. Updates b₂:", min_value=1.0, value=float(st.session_state["sis_gs_real_b2"]), step=1.0, key="ui_gs_real_b2"))
        b3_val = float(c3.number_input("Carga eq. Reportes b₃:", min_value=1.0, value=float(st.session_state["sis_gs_real_b3"]), step=1.0, key="ui_gs_real_b3"))
        tol_val = float(c4.number_input("Tolerancia E (ms):", min_value=1e-9, max_value=1.0, value=float(st.session_state["sis_gs_real_tol"]), format="%.5f", key="ui_gs_real_tol"))
        swap_rows = bool(st.checkbox("Intercambiar ecuaciones 1 ↔ 2 (rompe dominancia diagonal)", value=bool(st.session_state["sis_gs_real_swap"]), key="ui_gs_real_swap"))

        st.session_state["sis_gs_real_b1"] = b1_val
        st.session_state["sis_gs_real_b2"] = b2_val
        st.session_state["sis_gs_real_b3"] = b3_val
        st.session_state["sis_gs_real_tol"] = tol_val
        st.session_state["sis_gs_real_swap"] = swap_rows

        A = [_A_BD_GS[1], _A_BD_GS[0], _A_BD_GS[2]] if swap_rows else [list(r) for r in _A_BD_GS]
        b = [b2_val, b1_val, b3_val] if swap_rows else [b1_val, b2_val, b3_val]
        x0 = [0.0, 0.0, 0.0]
        max_it = 25
        etiquetas_vars = ["Consultas c", "Updates a", "Reportes r"]
        error_parseo = None
    else:
        if "sis_gs_A" not in st.session_state:
            st.session_state["sis_gs_A"] = [list(r) for r in defaults["A"]]
        if "sis_gs_b" not in st.session_state:
            st.session_state["sis_gs_b"] = list(defaults["b"])
        if "sis_gs_x0" not in st.session_state:
            st.session_state["sis_gs_x0"] = list(defaults.get("x0", [1.0, 1.0, 1.0]))
        if "sis_gs_tol" not in st.session_state:
            st.session_state["sis_gs_tol"] = float(defaults.get("tol", 0.01))
        if "sis_gs_max_iter" not in st.session_state:
            st.session_state["sis_gs_max_iter"] = int(defaults.get("max_iter", 50))

        claves_ui_lib = ("ui_gs_row_0", "ui_gs_row_1", "ui_gs_row_2", "ui_gs_x0_txt", "sis_gs__editor_df", "ui_gs_tol", "ui_gs_max")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP4 Ej. 8 (Cátedra 3×3, X₀ = [1, 1, 1])", key="btn_gs_tp4", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_gs_A": [list(r) for r in defaults["A"]],
                    "sis_gs_b": list(defaults["b"]),
                    "sis_gs_x0": list(defaults.get("x0", [1.0, 1.0, 1.0])),
                    "sis_gs_tol": float(defaults.get("tol", 0.01)),
                    "sis_gs_max_iter": int(defaults.get("max_iter", 50)),
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("1er Parcial 2026: Tiempos BD (5 pasos, X₀ = [0, 0, 0])", key="btn_gs_parcial", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_gs_A": [list(r) for r in _A_BD_GS],
                    "sis_gs_b": [17.0, 24.0, 33.0],
                    "sis_gs_x0": [0.0, 0.0, 0.0],
                    "sis_gs_tol": 1e-3,
                    "sis_gs_max_iter": 25,
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        error_parseo = None
        if modo_detallado:
            n_m = len(st.session_state["sis_gs_A"])
            A, b, x0 = _editor_sistema(
                "sis_gs",
                solo_lectura=False,
                incluir_x0=True,
                cols_vars=[f"Coef. x{i+1}" for i in range(n_m)],
                col_b="Término b",
                col_x0="Semilla x⁽⁰⁾",
            )
        else:
            A, b, error_parseo = _render_editor_sistema_dual("sis_gs", "ui_gs", defaults, False)
            x0_cur = st.session_state["sis_gs_x0"]
            x0_txt = st.text_input(
                "Semilla inicial x⁽⁰⁾:",
                value=", ".join(f"{v:g}" for v in x0_cur),
                key="ui_gs_x0_txt",
            )
            try:
                x0 = _parsear_vector_libre(x0_txt, len(A))
                st.session_state["sis_gs_x0"] = x0
            except Exception as exc:
                x0 = x0_cur
                if error_parseo is None:
                    error_parseo = exc

        c1, c2 = st.columns(2)
        tol_val = float(c1.number_input("Tolerancia E:", value=float(st.session_state["sis_gs_tol"]), min_value=1e-12, format="%.6f", key="ui_gs_tol"))
        max_it = int(c2.slider("Iteraciones máx:", 5, 100, int(st.session_state["sis_gs_max_iter"]), key="ui_gs_max"))
        st.session_state["sis_gs_tol"] = tol_val
        st.session_state["sis_gs_max_iter"] = max_it
        etiquetas_vars = [f"Variable x{i+1}" for i in range(len(A))]

    try:
        if error_parseo is not None:
            raise error_parseo
        dom = verificar_dominancia_diagonal(A)
        res = gauss_seidel(A, b, x0=x0, tol=tol_val, max_iter=max_it)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A, vector b o semilla x⁽⁰⁾ (Gauss-Seidel)",
            sugerencia_defecto="Asegurate de que ningún elemento diagonal a_ii sea 0 y todos los valores sean numéricos finitos.",
            ejemplo_defecto="5, 1, 1 | 17",
        )
        return

    if not dom["es_dominante"]:
        st.warning(
            f"**{dom['diagnostico']}** Reordená ecuaciones para ubicar el mayor coeficiente en la diagonal."
        )
    elif not res["convergente"]:
        st.warning("No alcanzó la tolerancia dentro del tope de iteraciones; aumentá el límite de iteraciones.")

    sol = res["solucion"]
    err_fin = res["error_final"]
    err_txt = f"{err_fin:.2e}" if np.isfinite(err_fin) else "Divergió (∞)"
    m1, m2, m3 = st.columns(3)
    if es_caso_real:
        m1.metric(
            "Consultas / Updates",
            f"{sol[0]:.2f} / {sol[1]:.2f} ms" if res["convergente"] else "Inestable (diverge)",
            f"Reportes r = {sol[2]:.2f} ms" if res["convergente"] else "Crecimiento explosivo sin dominancia",
            delta_color="off",
        )
    else:
        m1.metric(
            "Incógnitas x₁ / x₂",
            f"{sol[0]:.4f} / {sol[1]:.4f}" if len(sol) >= 2 and np.isfinite(err_fin) else "Inestable (diverge)",
            f"x₃ = {sol[2]:.4f}" if len(sol) == 3 and np.isfinite(err_fin) else "Crecimiento divergente",
            delta_color="off",
        )
    m2.metric(
        "Iteraciones",
        f"{res['iteraciones']} pasos",
        "Convergió" if res["convergente"] else "No convergió",
        delta_color="off",
    )
    m3.metric(
        "Error máx |Δx|",
        err_txt,
        "Dominancia diagonal ✓" if dom["es_dominante"] else "Sin dominancia diagonal ⚠",
        delta_color="off",
    )

    sol_gs = ", ".join(f"{etiquetas_vars[i]}={v:.6f}" for i, v in enumerate(sol))
    registrar_contexto_ia(
        "Solución y Dominancia Gauss-Seidel",
        f"- **Escenario:** {'Servidor BD' if es_caso_real else 'Libre / TP4 Ej. 8'} | **Dominancia diagonal:** `{dom['es_dominante']}`\n"
        f"- **Vector solución:** `[{sol_gs}]` | **Iteraciones:** `{res['iteraciones']}` | **Error:** `{err_txt}`",
        clave="sis_gs_res",
    )

    hist = res["historial"]
    if hist:
        k_vis = (
            render_slider_paso_k("Ver hasta la iteración $k$:", len(hist), key="vis_gs_k")
            if modo_detallado
            else len(hist)
        )
        sub_hist = muestrear_para_grafico(hist[:k_vis], max_puntos=220)
        ys_raw = [r["error_max"] for r in sub_hist if np.isfinite(r["error_max"]) and r["error_max"] > 0]
        xs_plot = [r["iteracion"] for r in sub_hist if np.isfinite(r["error_max"]) and r["error_max"] > 0]
        fig = crear_figura_base_plotly(
            (
                f"Convergencia de Error |Δx| ({res['iteraciones']} pasos)"
                if es_caso_real and not modo_detallado
                else f"Convergencia de Gauss-Seidel hasta k = {k_vis}"
            ),
            "Iteración k",
            "Error máx |Δx| (escala log)",
            altura=320,
        )
        if ys_raw:
            fig.add_trace(
                go.Scatter(
                    x=xs_plot,
                    y=limpiar_valores_plotly(ys_raw),
                    mode="lines+markers",
                    name="max |Δx|",
                    line=dict(color=PALETA_ARANDU["terracota"] if not dom["es_dominante"] else PALETA_ARANDU["oro"], width=2.5),
                    marker=dict(size=8, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.6)),
                )
            )
            fig.update_yaxes(type="log")
        render_plotly_arandu(fig, key="plot_sis_gs")

    if modo_detallado:
        if dom["es_dominante"]:
            st.success(
                f"**{dom['diagnostico']}** En todas las filas se verifica $|a_{{ii}}| > \\sum_{{j \\ne i}} |a_{{ij}}|$."
            )
        diag_filas = "; ".join(
            f"Fila {f['fila']}: $|{f['diagonal']:g}|$ {'>' if f['cumple'] else '≤'} ${f['suma_otros']:g}$"
            for f in dom.get("filas", [])
        )
        pasos_gs_vivos = [
            f"**Chequeo de dominancia diagonal:** {dom['diagnostico']} ({diag_filas}).",
            f"**Semilla y actualización in-place:** Partimos de $x^{{(0)}} = {[round(v, 4) for v in x0]}$ despejando cada $x_i^{{(k+1)}}$ con los componentes recién calculados.",
        ]
        for r_gs in hist[:5]:
            vec_txt = ", ".join(f"{etiquetas_vars[idx]}={val:.5f}" for idx, val in enumerate(r_gs["x"]))
            err_step_txt = f"{r_gs['error_max']:.3e}" if np.isfinite(r_gs["error_max"]) else "Inf (diverge)"
            pasos_gs_vivos.append(
                f"**Iteración $k = {r_gs['iteracion']}$:** `[{vec_txt}]` | $\\max|\\Delta x| = {err_step_txt}$."
            )
        if len(hist) > 5:
            pasos_gs_vivos.append(
                f"**Cierre en iteración $k = {res['iteraciones']}$:** Vector final `[{sol_gs}]` con error `{err_txt}`."
            )
        render_deduccion_paso_a_paso_viva(
            "Método Iterativo de Gauss-Seidel",
            pasos_gs_vivos,
            clave_ia="sis_gs_viva",
            expanded=True,
        )
        render_dataframe_acotado(
            _construir_tabla_gauss_seidel(hist, etiquetas_vars),
            max_filas=300,
            key_prefix="tbl_sis_gs",
            hide_index=True,
        )


def _tab_gauss_seidel() -> None:
    modo_detallado = render_toggle_modo_detallado("gauss_seidel")
    meta = render_bloque_1_teoria_y_ejemplo("gauss_seidel", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {
            "A": [[18.72, 8.20, 8.76], [6.40, 15.90, 7.18], [5.22, 6.40, 14.31]],
            "b": [121.280, 126.321, 118.522],
            "x0": [1.0, 1.0, 1.0],
            "tol": 0.01,
            "max_iter": 50,
        },
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_gauss_seidel(defaults, modo_detallado=modo_detallado)


def _construir_tabla_potencias(
    historial: list[dict],
    es_dir: bool,
    etiquetas_nodos: list[str],
) -> list[dict]:
    """Formatea la tabla iterativa del Método de las Potencias con encabezados 100% en español."""
    filas = []
    for h in historial:
        reg: dict[str, object] = {"Iteración k": h["iteracion"]}
        if not es_dir and "lambda_inversa" in h:
            reg["Autovalor A⁻¹ (μ_k)"] = round(float(h["lambda_inversa"]), 6)
            reg["Autovalor λ_k = 1/μ_k"] = round(float(h["lambda"]), 6) if np.isfinite(h["lambda"]) else None
        else:
            reg["Autovalor λ_k"] = round(float(h["lambda"]), 6)
        for idx, val in enumerate(h["x_norm"]):
            nom = etiquetas_nodos[idx] if idx < len(etiquetas_nodos) else f"x{idx+1}"
            reg[f"Modo {nom}"] = round(float(val), 6)
        err_p = h["error_pct"]
        reg["Error relativo (%)"] = round(float(err_p), 4) if err_p is not None else None
        reg["¿Cumple Tolerancia?"] = "Sí" if h["cumple_tol"] else "No"
        filas.append(reg)
    return filas


@st.fragment
def _frag_potencias(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_cascada", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Cascada de Reintentos en Microservicios (Gateway, API, BD)"
            if op == "real_cascada"
            else "Matriz Libre 3×3 y Cátedra TP4 Ej. 10"
        ),
        horizontal=True,
        key="ui_pot_contexto",
    )
    es_caso_real = contexto == "real_cascada"

    if es_caso_real:
        st.caption(
            r"Matriz de propagación $A = [[d, c, 0], [c, d, c], [0, c, d]]$ desde carga inicial unitaria $X_0 = [1, 1, 1]^T$: "
            r"si $|\lambda_{\max}| < 1$ la cola se apaga; si $\lambda_{\max} > 1$ la red colapsa sobre el servicio pico."
        )
        if "sis_pot_real_d" not in st.session_state:
            st.session_state["sis_pot_real_d"] = 0.50
        if "sis_pot_real_c" not in st.session_state:
            st.session_state["sis_pot_real_c"] = 0.25
        if "sis_pot_real_tipo" not in st.session_state:
            st.session_state["sis_pot_real_tipo"] = "Mayor |λ_max| (Potencia Directa)"

        claves_ui_real = ("ui_pot_real_d", "ui_pot_real_c", "ui_pot_real_tipo")
        b1, b2, b3 = st.columns(3)
        if b1.button("Estable (c=0.25 → λ_max ≈ 0.85)", key="btn_pot_real_est", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_pot_real_d": 0.50, "sis_pot_real_c": 0.25, "sis_pot_real_tipo": "Mayor |λ_max| (Potencia Directa)"},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b2.button("Cascada inestable (c=0.45 → λ_max ≈ 1.14)", key="btn_pot_real_casc", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_pot_real_d": 0.50, "sis_pot_real_c": 0.45, "sis_pot_real_tipo": "Mayor |λ_max| (Potencia Directa)"},
                claves_ui_real,
            )
            rerun_fragmento_seguro()
        if b3.button("Empate |λ₁|=|λ₃| (d=0 → oscila 50%)", key="btn_pot_real_osc", use_container_width=True):
            _aplicar_preset_sistemas(
                {"sis_pot_real_d": 0.00, "sis_pot_real_c": 0.60, "sis_pot_real_tipo": "Mayor |λ_max| (Potencia Directa)"},
                claves_ui_real,
            )
            rerun_fragmento_seguro()

        c1, c2, c3 = st.columns([1.0, 1.0, 1.35])
        d_val = float(c1.slider("Retención local d:", 0.00, 1.00, float(st.session_state["sis_pot_real_d"]), step=0.05, key="ui_pot_real_d"))
        c_val = float(c2.slider("Tasa de reintentos c:", 0.05, 0.70, float(st.session_state["sis_pot_real_c"]), step=0.05, key="ui_pot_real_c"))
        tipo = str(
            c3.radio(
                "Modo espectral a extraer:",
                ["Mayor |λ_max| (Potencia Directa)", "Menor |λ_min| (Potencia Inversa)"],
                index=0 if str(st.session_state["sis_pot_real_tipo"]).startswith("Mayor") else 1,
                horizontal=True,
                key="ui_pot_real_tipo",
            )
        )
        st.session_state["sis_pot_real_d"] = d_val
        st.session_state["sis_pot_real_c"] = c_val
        st.session_state["sis_pot_real_tipo"] = tipo

        A = _matriz_cascada_potencias(d_val, c_val)
        x0 = [1.0, 1.0, 1.0]
        tol_pct = 0.05
        max_it = 25
        etiquetas_nodos = ["Gateway", "API", "BD"]
        error_parseo = None
    else:
        if "sis_pot_A" not in st.session_state:
            st.session_state["sis_pot_A"] = [list(r) for r in defaults["A"]]
        if "sis_pot_x0" not in st.session_state:
            st.session_state["sis_pot_x0"] = list(defaults["x0"])
        if "sis_pot_tol_pct" not in st.session_state:
            st.session_state["sis_pot_tol_pct"] = float(defaults["tol_pct"])
        if "sis_pot_max_iter" not in st.session_state:
            st.session_state["sis_pot_max_iter"] = int(defaults["max_iter"])
        if "sis_pot_tipo" not in st.session_state:
            st.session_state["sis_pot_tipo"] = "Mayor |λ_max| (Potencia Directa)"

        claves_ui_lib = ("ui_pot_row_0", "ui_pot_row_1", "ui_pot_row_2", "ui_pot_x0_txt", "sis_pot__editor_df", "ui_pot_tol", "ui_pot_max", "ui_pot_tipo")
        b_l1, b_l2 = st.columns(2)
        if b_l1.button("Restaurar TP4 Ej. 10 (Cátedra 3×3, X₀ = [1, 1, 1])", key="btn_pot_tp4", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_pot_A": [list(r) for r in defaults["A"]],
                    "sis_pot_x0": list(defaults["x0"]),
                    "sis_pot_tol_pct": float(defaults["tol_pct"]),
                    "sis_pot_max_iter": int(defaults["max_iter"]),
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()
        if b_l2.button("Cascada crítica 3 Microservicios (c=0.45 → λ_max ≈ 1.136)", key="btn_pot_cascada", use_container_width=True):
            _aplicar_preset_sistemas(
                {
                    "sis_pot_A": _matriz_cascada_potencias(0.50, 0.45),
                    "sis_pot_x0": [1.0, 1.0, 1.0],
                    "sis_pot_tol_pct": 0.05,
                    "sis_pot_max_iter": 30,
                    "sis_pot_tipo": "Mayor |λ_max| (Potencia Directa)",
                },
                claves_ui_lib,
            )
            rerun_fragmento_seguro()

        error_parseo = None
        A_cur = st.session_state["sis_pot_A"]
        x0_cur = st.session_state["sis_pot_x0"]
        n_m = len(A_cur)
        if modo_detallado:
            cols_A_es = [f"Coef. col. {i+1}" for i in range(n_m)]
            df_in = pd.DataFrame(A_cur, columns=cols_A_es)
            df_in["Semilla x⁽⁰⁾"] = x0_cur
            df_edit = render_data_editor_arandu(
                df_in,
                use_container_width=True,
                hide_index=True,
                disabled=False,
                key="sis_pot__editor_df",
            )
            A = df_edit[cols_A_es].astype(float).values.tolist()
            x0 = df_edit["Semilla x⁽⁰⁾"].astype(float).tolist()
            st.session_state["sis_pot_A"] = A
            st.session_state["sis_pot_x0"] = x0
        else:
            cols_r = st.columns(n_m)
            A_tmp = []
            for idx_r in range(n_m):
                def_txt = ", ".join(f"{v:g}" for v in A_cur[idx_r])
                txt_r = cols_r[idx_r].text_input(
                    f"Fila {idx_r + 1} [aᵢ₁, aᵢ₂, aᵢ₃]:",
                    value=def_txt,
                    key=f"ui_pot_row_{idx_r}",
                )
                try:
                    A_tmp.append(_parsear_vector_libre(txt_r, n_m))
                except Exception as exc:
                    if error_parseo is None:
                        error_parseo = exc
            if error_parseo is None:
                A = A_tmp
                st.session_state["sis_pot_A"] = A
            else:
                A = A_cur

            x0_txt = st.text_input(
                "Semilla inicial x⁽⁰⁾:",
                value=", ".join(f"{v:g}" for v in x0_cur),
                key="ui_pot_x0_txt",
            )
            try:
                x0 = _parsear_vector_libre(x0_txt, len(A))
                st.session_state["sis_pot_x0"] = x0
            except Exception as exc:
                x0 = x0_cur
                if error_parseo is None:
                    error_parseo = exc

        c1, c2, c3 = st.columns([0.9, 0.9, 1.4])
        tol_pct = float(c1.number_input("Tolerancia E (%):", value=float(st.session_state["sis_pot_tol_pct"]), min_value=1e-6, format="%.5f", key="ui_pot_tol"))
        max_it = int(c2.slider("Iteraciones máx:", 5, 100, int(st.session_state["sis_pot_max_iter"]), key="ui_pot_max"))
        tipo = str(
            c3.radio(
                "Autovalor buscado:",
                ["Mayor |λ_max| (Potencia Directa)", "Menor |λ_min| (Potencia Inversa)"],
                index=0 if str(st.session_state["sis_pot_tipo"]).startswith("Mayor") else 1,
                horizontal=True,
                key="ui_pot_tipo",
            )
        )
        st.session_state["sis_pot_tol_pct"] = tol_pct
        st.session_state["sis_pot_max_iter"] = max_it
        st.session_state["sis_pot_tipo"] = tipo
        etiquetas_nodos = [f"x{i+1}" for i in range(len(A))]

    es_dir = tipo.startswith("Mayor")
    try:
        if error_parseo is not None:
            raise error_parseo
        if es_dir:
            res = metodo_potencias_max(A, x0=x0, tol=tol_pct, max_iter=max_it)
            lam_final = res["autovalor_max"]
        else:
            res = metodo_potencias_min(A, x0=x0, tol=tol_pct, max_iter=max_it)
            lam_final = res["autovalor_min"]
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A o vector inicial x⁽⁰⁾ (Potencias)",
            sugerencia_defecto="Ingresá una matriz cuadrada con valores finitos y semilla no nula; en Potencia Inversa, det(A) no debe ser 0.",
            ejemplo_defecto="x⁽⁰⁾ = 1, 1, 1",
        )
        return

    if not res["convergente"]:
        st.warning(
            "No convergió dentro del tope de iteraciones (frecuente cuando dos autovalores dominantes empatan en módulo |λ₁| ≈ |λ₂|, provocando oscilación)."
        )

    vec_aut = res["autovector"]
    idx_pico = int(np.argmax(np.abs(vec_aut)))
    nodo_pico = etiquetas_nodos[idx_pico] if idx_pico < len(etiquetas_nodos) else f"x{idx_pico+1}"

    m1, m2, m3 = st.columns(3)
    if es_caso_real:
        sub_m1 = (
            "⚠ Oscila sin converger"
            if not res["convergente"]
            else (f"⚠ Cascada (+{(abs(lam_final) - 1.0) * 100:.1f}%/ciclo)" if es_dir and abs(lam_final) > 1.0 else "✓ Amortiguado (|λ| < 1)")
        )
        m1.metric("Factor dominante λ_max" if es_dir else "Autovalor menor λ_min", f"{lam_final:.4f}", sub_m1, delta_color="off")
        m2.metric("Iteraciones", f"{res['iteraciones']} pasos", "Convergió" if res["convergente"] else "Oscilando / tope", delta_color="off")
        m3.metric("Error relativo final", f"{res['error_final_pct']:.4f}%", f"Nodo pico: {nodo_pico} ({vec_aut[idx_pico]:.2f})", delta_color="off")
    else:
        m1.metric("Autovalor λ_max" if es_dir else "Autovalor λ_min", f"{lam_final:.5f}", f"‖Ax − λx‖₂ = {res['error_autovector']:.2e}", delta_color="off")
        m2.metric("Iteraciones", f"{res['iteraciones']} pasos", "Convergió" if res["convergente"] else "No convergió", delta_color="off")
        m3.metric("Error relativo final", f"{res['error_final_pct']:.4f}%", f"Pico: {nodo_pico} ({vec_aut[idx_pico]:.3f})", delta_color="off")

    vec_str = ", ".join(f"{v:.5f}" for v in vec_aut)
    registrar_contexto_ia(
        "Resultado Método de las Potencias",
        f"- **Escenario:** {'Cascada Microservicios' if es_caso_real else 'Libre / TP4 Ej. 10'} | **Modo:** {tipo}\n"
        f"- **Autovalor estimado λ:** `{lam_final:.6f}` | **Autovector:** `[{vec_str}]` | "
        f"**Iteraciones:** `{res['iteraciones']}` | **Error %:** `{res['error_final_pct']:.4f}%`",
        clave="sis_pot_res",
    )

    hist = res["historial"]
    if hist:
        k_vis = (
            render_slider_paso_k("Ver hasta la iteración k:", len(hist), key="vis_pot_k")
            if modo_detallado
            else len(hist)
        )
        sub_h = muestrear_para_grafico(hist[:k_vis], max_puntos=220)
        vec_vis = hist[k_vis - 1]["x_norm"]

        c_g1, c_g2 = st.columns([1.35, 1.0])
        with c_g1:
            fig = crear_figura_base_plotly(
                f"Estabilización de {'λ_max' if es_dir else 'λ_min'} ({k_vis} pasos)",
                "Iteración k",
                "λ_k estimado",
                altura=315,
            )
            color_lam = PALETA_ARANDU["terracota"] if (es_caso_real and es_dir and (abs(lam_final) > 1.0 or not res["convergente"])) else PALETA_ARANDU["salvia"]
            fig.add_trace(
                go.Scatter(
                    x=[r["iteracion"] for r in sub_h],
                    y=limpiar_valores_plotly([r["lambda"] for r in sub_h]),
                    mode="lines+markers",
                    name="λ_k",
                    line=dict(color=color_lam, width=2.5),
                    marker=dict(size=8, color=PALETA_ARANDU["marfil_nodo"], line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.6)),
                )
            )
            if es_caso_real and es_dir:
                fig.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color=PALETA_ARANDU["terracota_oscuro"],
                    annotation_text="Umbral cascada (λ = 1)",
                )
            render_plotly_arandu(fig, key="plot_sis_pot")

        with c_g2:
            idx_pico_vis = int(np.argmax(np.abs(vec_vis)))
            colores_bar = [
                PALETA_ARANDU["terracota"] if idx == idx_pico_vis else PALETA_ARANDU["oro"]
                for idx in range(len(vec_vis))
            ]
            fig_vec = crear_figura_base_plotly(
                "Carga relativa (autovector)" if es_caso_real else f"Autovector normalizado X_{k_vis}",
                "Componente",
                "Peso normalizado",
                altura=315,
            )
            fig_vec.add_trace(
                go.Bar(
                    x=etiquetas_nodos[: len(vec_vis)],
                    y=limpiar_valores_plotly(vec_vis),
                    marker_color=colores_bar,
                    text=[f"{v:.3f}" for v in vec_vis],
                    textposition="outside",
                    name="X_k",
                )
            )
            fig_vec.add_hline(y=0.0, line_color=PALETA_ARANDU["texto_atenuado"], line_width=1.0)
            if vec_vis and all(np.isfinite(v) for v in vec_vis):
                fig_vec.update_yaxes(range=[min(-0.18, min(vec_vis) * 1.24), max(1.22, max(vec_vis) * 1.24)])
            render_plotly_arandu(fig_vec, key="plot_sis_pot_vec")

    if modo_detallado:
        if not es_dir and "A_inversa" in res:
            with st.expander("Ver Matriz Inversa A⁻¹ empleada previamente (Potencia Inversa)"):
                render_dataframe_acotado(
                    _construir_tabla_matriz_es(res["A_inversa"], etiquetas_nodos),
                    max_filas=100,
                    key_prefix="tbl_sis_pot_inv",
                    hide_index=True,
                )

        pasos_pot_vivos = [
            f"**Planteo ({tipo}):** Partiendo de $x^{{(0)}} = {x0}$, en cada paso multiplicamos "
            f"$Y_k = {'A' if es_dir else 'A^{-1}'} X_{{k-1}}$ y normalizamos por el componente de mayor módulo ($\\ell_\\infty$)."
        ]
        for r_pt in hist[:5]:
            err_p = f", error $\\varepsilon = {r_pt['error_pct']:.4f}\\%$" if r_pt["error_pct"] is not None else ""
            vec_n = ", ".join(f"{v:.4f}" for v in r_pt["x_norm"])
            if es_dir:
                pasos_pot_vivos.append(
                    f"**Iteración $k = {r_pt['iteracion']}$:** $\\lambda_{{{r_pt['iteracion']}}} = {r_pt['lambda']:.6f}$, vector normalizado `[{vec_n}]`{err_p}."
                )
            else:
                pasos_pot_vivos.append(
                    f"**Iteración $k = {r_pt['iteracion']}$:** $\\mu_k = {r_pt['lambda_inversa']:.6f} \\implies \\lambda_k = 1/\\mu_k = {r_pt['lambda']:.6f}$, `X_k = [{vec_n}]`{err_p}."
                )
        if len(hist) > 5:
            pasos_pot_vivos.append(
                f"**Cierre en iteración $k = {res['iteraciones']}$:** $\\lambda \\approx {lam_final:.6f}$ (residual $\\|Ax - \\lambda x\\|_2 = {res['error_autovector']:.3e}$)."
            )
        render_deduccion_paso_a_paso_viva(
            f"Método de las Potencias ({'Directo' if es_dir else 'Inverso'})",
            pasos_pot_vivos,
            clave_ia="sis_pot_viva",
            expanded=True,
        )
        render_dataframe_acotado(
            _construir_tabla_potencias(hist, es_dir, etiquetas_nodos),
            max_filas=300,
            key_prefix="tbl_sis_pot",
            hide_index=True,
        )


def _tab_potencias() -> None:
    modo_detallado = render_toggle_modo_detallado("potencias")
    meta = render_bloque_1_teoria_y_ejemplo("potencias", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    raw_def = practica.get("datos_iniciales", {})
    defaults = {
        "A": raw_def.get("A", [[3.556, -1.778, 0.0], [-1.778, 3.556, -1.778], [0.0, -1.778, 3.556]]),
        "x0": raw_def.get("x0", [1.0, 1.0, 1.0]),
        "tol_pct": float(raw_def.get("tol", raw_def.get("tol_pct", 0.05))),
        "max_iter": int(raw_def.get("max_iter", 30)),
    }
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_potencias(defaults, modo_detallado=modo_detallado)


_ESCENARIOS_FADDEEV: tuple[dict[str, Any], ...] = (
    {
        "id": "tp4_catedra",
        "nombre": "TP4 Ej. Cátedra: Matriz simétrica 3×3",
        "nota": "Caso oficial TP4 para extraer polinomio característico, autovalores e inversa por trazas.",
    },
    {
        "id": "parcial2_2026_t1",
        "nombre": "2do Parcial 2026 (Tema 1): 3 raíces reales en (0, 1.5) y B₃ - b₃I = 0",
        "A": [[1.0, 0.2, 0.0], [0.2, 0.8, 0.2], [0.0, 0.2, 0.5]],
        "nota": "Modelo del 2do Parcial 2026 (Tema 1): matriz tridiagonal positiva con 3 variaciones de signo en Descartes.",
    },
    {
        "id": "parcial2_2026_t3",
        "nombre": "2do Parcial 2026 (Temas 3 y 4): Simétrica 3×3 [[4,3,0],[3,6,2],[0,2,1]]",
        "A": [[4.0, 3.0, 0.0], [3.0, 6.0, 2.0], [0.0, 2.0, 1.0]],
        "nota": "Matriz reportada en el 2do Parcial 2026 (Temas 3 y 4) para hallar autovalores y verificar Cayley-Hamilton.",
    },
)


def _matriz_lazo_faddeev(beta: float) -> list[list[float]]:
    """Matriz 3x3 de acoplamiento en lazo de 3 etapas con realimentación beta."""
    bf = float(beta)
    return [
        [0.6, 0.2, bf],
        [0.2, 0.5, 0.2],
        [0.0, 0.2, 0.4],
    ]


def _formatear_espectro_faddeev(autovalores: list[Any]) -> str:
    """Resume todos los autovalores reales y pares complejos conjugados en una línea."""
    partes: list[str] = []
    vistos_comp: set[tuple[float, float]] = set()
    for eig in autovalores:
        if isinstance(eig, complex) and abs(eig.imag) >= 1e-6:
            re_r = 0.0 if abs(eig.real) < 1e-9 else float(eig.real)
            im_abs = float(abs(eig.imag))
            firma = (round(re_r, 4), round(im_abs, 4))
            if firma not in vistos_comp:
                vistos_comp.add(firma)
                partes.append(f"{re_r:.3f}±{im_abs:.3f}j")
        else:
            val_r = float(eig.real if isinstance(eig, complex) else eig)
            if abs(val_r) < 1e-9:
                val_r = 0.0
            partes.append(f"{val_r:.3f}")
    return ", ".join(partes)


def _construir_tabla_trazas_faddeev(historial_pasos: list[dict]) -> list[dict]:
    """Formatea los n pasos de trazas y coeficientes b_k con encabezados 100% en español."""
    return [
        {
            "Paso k": hp["k"],
            "Recurrencia": hp["formula"],
            "Traza tr(B_k)": round(float(hp["traza"]), 6),
            "Coeficiente b_k": round(float(hp["b_k"]), 6),
        }
        for hp in historial_pasos
    ]


def _al_cambiar_preset_fad(defaults: dict) -> None:
    id_sel = st.session_state.get("ui_fad_preset", "tp4_catedra")
    esc = next((e for e in _ESCENARIOS_FADDEEV if e["id"] == id_sel), None)
    if esc is None:
        return
    src_A = defaults["A"] if id_sel == "tp4_catedra" else esc["A"]
    st.session_state["sis_fad_A"] = [list(fila) for fila in src_A]
    for clave_ui in ("sis_fad__editor_df", "ui_fad_row_0", "ui_fad_row_1", "ui_fad_row_2"):
        st.session_state.pop(clave_ui, None)


@st.fragment
def _frag_faddeev(defaults: dict, modo_detallado: bool) -> None:
    st.markdown("### 3. Simulador Protagonista")
    contexto = st.radio(
        "Escenario de simulación:",
        options=["real_lazo", "libre_catedra"],
        format_func=lambda op: (
            "Caso Real: Lazo de Control en 3 Etapas (Ingesta, Proceso, Salida)"
            if op == "real_lazo"
            else "Matriz Libre 3×3, TP4 Ej. 9 y 2do Parcial 2026"
        ),
        horizontal=True,
        key="ui_fad_contexto",
    )
    es_caso_real = contexto == "real_lazo"

    error_parseo: Exception | None = None
    if es_caso_real:
        st.caption(
            r"Lazo $A(\beta) = [[0.6, 0.2, \beta], [0.2, 0.5, 0.2], [0, 0.2, 0.4]]$: "
            r"$\beta$ solo cambia el último coeficiente $b_3 = \det(A)$, desplazando $P(\lambda)$ hasta crear modos complejos o singularidad."
        )
        if "sis_fad_real_beta" not in st.session_state:
            st.session_state["sis_fad_real_beta"] = 0.0

        claves_ui_real = ("ui_fad_real_beta",)
        b1, b2, b3 = st.columns(3)
        if b1.button("Sin realimentación (β = 0 → 0.8, 0.5, 0.2)", key="btn_fad_real_real", use_container_width=True):
            _aplicar_preset_sistemas({"sis_fad_real_beta": 0.0}, claves_ui_real)
            rerun_fragmento_seguro()
        if b2.button("Lazo negativo (β = -0.5 → modos complejos)", key="btn_fad_real_comp", use_container_width=True):
            _aplicar_preset_sistemas({"sis_fad_real_beta": -0.5}, claves_ui_real)
            rerun_fragmento_seguro()
        if b3.button("Singularidad (β = -2.0 → det A = 0, sin A⁻¹)", key="btn_fad_real_sing", use_container_width=True):
            _aplicar_preset_sistemas({"sis_fad_real_beta": -2.0}, claves_ui_real)
            rerun_fragmento_seguro()

        beta_val = float(
            st.slider(
                "Retroalimentación Salida → Ingesta (β):",
                min_value=-2.20,
                max_value=1.00,
                value=float(st.session_state["sis_fad_real_beta"]),
                step=0.05,
                key="ui_fad_real_beta",
            )
        )
        st.session_state["sis_fad_real_beta"] = beta_val
        A = _matriz_lazo_faddeev(beta_val)
        etiquetas_cols = ["Ingesta", "Proceso", "Salida"]
    else:
        if "sis_fad_A" not in st.session_state:
            st.session_state["sis_fad_A"] = [list(r) for r in defaults["A"]]
        ids_fad = [e["id"] for e in _ESCENARIOS_FADDEEV]
        nom_fad = {e["id"]: e["nombre"] for e in _ESCENARIOS_FADDEEV}
        if "ui_fad_preset" not in st.session_state or st.session_state["ui_fad_preset"] not in ids_fad:
            st.session_state["ui_fad_preset"] = ids_fad[0]

        fad_id = st.selectbox(
            "Escenario de estudio (TP4 y Parciales):",
            options=ids_fad,
            format_func=lambda k_f: nom_fad[k_f],
            key="ui_fad_preset",
            on_change=_al_cambiar_preset_fad,
            args=(defaults,),
        )
        esc_fad = next(e for e in _ESCENARIOS_FADDEEV if e["id"] == fad_id)
        st.caption(esc_fad["nota"])

        A_cur = st.session_state["sis_fad_A"]
        n_m = len(A_cur)
        if modo_detallado:
            cols_A_es = [f"Coef. col. {i+1}" for i in range(n_m)]
            df_edit = render_data_editor_arandu(
                pd.DataFrame(A_cur, columns=cols_A_es),
                use_container_width=True,
                hide_index=True,
                disabled=False,
                key="sis_fad__editor_df",
            )
            A = df_edit[cols_A_es].astype(float).values.tolist()
            st.session_state["sis_fad_A"] = A
        else:
            cols_r = st.columns(n_m)
            A_tmp = []
            for idx_r in range(n_m):
                def_txt = ", ".join(f"{v:g}" for v in A_cur[idx_r])
                txt_r = cols_r[idx_r].text_input(
                    f"Fila {idx_r + 1} [aᵢ₁, aᵢ₂, aᵢ₃]:",
                    value=def_txt,
                    key=f"ui_fad_row_{idx_r}",
                )
                try:
                    A_tmp.append(_parsear_vector_libre(txt_r, n_m))
                except Exception as exc:
                    if error_parseo is None:
                        error_parseo = exc
            if error_parseo is None:
                A = A_tmp
                st.session_state["sis_fad_A"] = A
            else:
                A = A_cur
        etiquetas_cols = [f"x{i+1}" for i in range(len(A))]

    try:
        if error_parseo is not None:
            raise error_parseo
        res = metodo_faddeev_leverrier(A)
    except Exception as exc:
        render_excepcion_validacion(
            exc,
            campo="Matriz A (Faddeev-Leverrier)",
            sugerencia_defecto="Ingresá una matriz cuadrada numérica finita de orden n >= 2.",
            ejemplo_defecto="4, 1, 0",
        )
        return

    eigs = res["autovalores"]
    reales = res["autovalores_reales"]
    n_comp = len(eigs) - len(reales)
    sub_m1 = (
        f"{len(reales)} raíces reales puras"
        if n_comp == 0
        else f"{len(reales)} real{'es' if len(reales) != 1 else ''} + {n_comp} complejas (oscila)"
    )

    det_disp = 0.0 if abs(res["determinante"]) < 1e-12 else float(res["determinante"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Espectro completo (λᵢ)", _formatear_espectro_faddeev(eigs), sub_m1, delta_color="off")
    m2.metric(
        "Determinante det(A)",
        f"{det_disp:.4f}",
        "Invertible (bₙ ≠ 0) ✓" if res["es_invertible"] else "Singular (bₙ = 0 → sin A⁻¹)",
        delta_color="off",
    )
    m3.metric(
        "Cayley-Hamilton ‖Bₙ−bₙI‖",
        f"{res['norma_cayley_hamilton']:.1e}",
        f"Error A⁻¹: {res['error_inversa']:.1e}" if res["error_inversa"] is not None else "Inversa A⁻¹ no existe",
        delta_color="off",
    )

    st.info(
        f"**Polinomio característico:** `{res['polinomio_texto']}`  \n"
        f"**Regla de Descartes:** {res['descartes_polinomio']['resumen']}"
    )
    registrar_contexto_ia(
        "Polinomio Característico y Autovalores (Faddeev-Leverrier)",
        f"- **Escenario:** {'Lazo 3 Etapas' if es_caso_real else 'Libre / TP4 Ej. 9'}\n"
        f"- **Polinomio:** `{res['polinomio_texto']}`\n"
        f"- **Descartes:** {res['descartes_polinomio']['resumen']}\n"
        f"- **det(A):** `{res['determinante']:.6f}` | **‖Bₙ - bₙ·I‖:** `{res['norma_cayley_hamilton']:.2e}`\n"
        f"- **Autovalores:** `{eigs}`",
        clave="sis_fad_res",
    )

    curva = res["curva_polinomio"]
    k_vis = (
        render_slider_paso_k("Destacar hasta la raíz real k:", len(reales), key="vis_fad_k")
        if (modo_detallado and reales)
        else len(reales)
    )

    c_g1, c_g2 = st.columns([1.35, 1.0])
    with c_g1:
        fig = crear_figura_base_plotly("Polinomio Característico P(λ) = 0", "λ", "P(λ)", altura=320)
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
        if reales and k_vis > 0:
            fig.add_trace(
                go.Scatter(
                    x=reales[:k_vis],
                    y=[0.0] * k_vis,
                    mode="markers",
                    name="Raíces reales",
                    marker=dict(
                        size=13,
                        color=PALETA_ARANDU["marfil_nodo"],
                        symbol="star",
                        line=dict(color=PALETA_ARANDU["terracota_oscuro"], width=1.8),
                    ),
                )
            )
        render_plotly_arandu(fig, key="plot_sis_fad")

    with c_g2:
        re_vals = [float(v.real if isinstance(v, complex) else v) for v in eigs]
        im_vals = [float(v.imag if isinstance(v, complex) else 0.0) for v in eigs]
        colores_eig = [
            PALETA_ARANDU["terracota"] if abs(im) >= 1e-6 else PALETA_ARANDU["salvia"]
            for im in im_vals
        ]
        fig_pl = crear_figura_base_plotly("Espectro en el Plano Complejo", "Re(λ)", "Im(λ)", altura=320)
        fig_pl.add_hline(y=0.0, line_color=PALETA_ARANDU["texto_atenuado"], line_width=1.0)
        fig_pl.add_vline(x=0.0, line_color=PALETA_ARANDU["texto_atenuado"], line_width=1.0)
        if es_caso_real:
            theta = np.linspace(0.0, 2.0 * np.pi, 120)
            fig_pl.add_trace(
                go.Scatter(
                    x=np.cos(theta).tolist(),
                    y=np.sin(theta).tolist(),
                    mode="lines",
                    name="|λ| = 1",
                    line=dict(color=PALETA_ARANDU["texto_atenuado"], dash="dot", width=1.4),
                )
            )
        fig_pl.add_trace(
            go.Scatter(
                x=re_vals,
                y=im_vals,
                mode="markers",
                name="Autovalores λᵢ",
                marker=dict(
                    size=12,
                    color=colores_eig,
                    symbol="diamond",
                    line=dict(color=PALETA_ARANDU["marfil_nodo"], width=1.4),
                ),
            )
        )
        x_lo = min(re_vals + ([-1.05] if es_caso_real else [0.0]))
        x_hi = max(re_vals + ([1.05] if es_caso_real else [0.0]))
        pad_x = max(0.35, 0.22 * (x_hi - x_lo))
        y_abs = max([abs(y) for y in im_vals] + ([1.05] if es_caso_real else [0.5]))
        pad_y = max(0.35, 0.25 * y_abs)
        fig_pl.update_xaxes(range=[x_lo - pad_x, x_hi + pad_x])
        fig_pl.update_yaxes(range=[-(y_abs + pad_y), y_abs + pad_y])
        render_plotly_arandu(fig_pl, key="plot_sis_fad_plano")

    if modo_detallado:
        pasos_fad_vivos = [
            "**Algoritmo matricial en $n$ pasos:** Para $k = 1 \\dots n$ computamos la matriz auxiliar $B_k$ y extraemos $b_k = \\frac{1}{k}\\operatorname{tr}(B_k)$."
        ]
        for hp in res.get("historial_pasos", []):
            pasos_fad_vivos.append(
                f"**Paso $k = {hp['k']}$:** `{hp['formula']}` $\\implies \\operatorname{{tr}}(B_{{{hp['k']}}}) = {hp['traza']:.6g}$, "
                f"$b_{{{hp['k']}}} = \\frac{{{hp['traza']:.6g}}}{{{hp['k']}}} = {hp['b_k']:.6g}$."
            )
        inv_txt = (
            f"$A^{{-1}} = \\frac{{1}}{{b_n}}(B_{{n-1}} - b_{{n-1}}I)$ calculada con error `{res['error_inversa']:.2e}`"
            if res["inversa"] is not None
            else "Matriz singular ($b_n = 0$), no existe inversa."
        )
        pasos_fad_vivos.append(
            f"**Descartes y Cayley-Hamilton:** {res['descartes_polinomio']['resumen']} Verificación $\\|B_n - b_n I\\|_F = {res['norma_cayley_hamilton']:.2e}$."
        )
        pasos_fad_vivos.append(
            f"**Polinomio característico y espectro:** `{res['polinomio_texto']}`, $\\det(A) = {res['determinante']:.6f}$, autovalores `{eigs}`. {inv_txt}."
        )
        render_deduccion_paso_a_paso_viva(
            "Faddeev-Leverrier (Trazas y Polinomio)",
            pasos_fad_vivos,
            clave_ia="sis_fad_viva",
            expanded=True,
        )
        render_dataframe_acotado(
            _construir_tabla_trazas_faddeev(res.get("historial_pasos", [])),
            max_filas=100,
            key_prefix="tbl_sis_fad_coef",
            hide_index=True,
        )

        with st.expander("Inspeccionar Matrices Auxiliares B_k Paso a Paso"):
            hist_bk = res.get("historial_pasos", [])
            if hist_bk:
                k_bk = render_slider_paso_k("Inspeccionar matriz auxiliar B_k (paso k):", len(hist_bk), key="vis_fad_bk")
                hp_bk = hist_bk[k_bk - 1]
                st.caption(f"Paso {hp_bk['k']}: {hp_bk['formula']} | tr(B_{hp_bk['k']}) = {hp_bk['traza']:.6g} | b_{hp_bk['k']} = {hp_bk['b_k']:.6g}")
                render_dataframe_acotado(
                    _construir_tabla_matriz_es(hp_bk["B_k"], etiquetas_cols),
                    max_filas=100,
                    key_prefix="tbl_sis_fad_bk",
                    hide_index=True,
                )

        if res["inversa"] is not None:
            with st.expander("Ver Matriz Inversa A⁻¹ obtenida por Faddeev"):
                render_dataframe_acotado(
                    _construir_tabla_matriz_es(res["inversa"], etiquetas_cols),
                    max_filas=300,
                    key_prefix="tbl_sis_fad_inv",
                    hide_index=True,
                )


def _tab_faddeev() -> None:
    modo_detallado = render_toggle_modo_detallado("faddeev_leverrier")
    meta = render_bloque_1_teoria_y_ejemplo("faddeev_leverrier", modo_detallado=modo_detallado)
    practica = meta.get("practica_profesora", {})
    defaults = practica.get(
        "datos_iniciales",
        {"A": [[3.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 3.0]]},
    )
    if modo_detallado:
        render_practica_catedra_info(
            titulo_practica=practica.get("titulo"),
            enunciado_practica=practica.get("enunciado"),
            solucion_referencia=practica.get("solucion_referencia"),
        )
    st.divider()
    _frag_faddeev(defaults, modo_detallado=modo_detallado)


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
