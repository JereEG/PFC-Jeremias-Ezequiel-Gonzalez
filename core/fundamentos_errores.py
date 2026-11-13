"""
core/fundamentos_errores.py
===========================
Módulo de cálculo puro para Fundamentos Numéricos, Series y Teoría de Errores
(Arandú: Métodos Computacionales v3.0).

Cubre:
- Series de Taylor (e^x, pi Gregory-Leibniz) con Criterio de Parada de Scarborough.
- Sucesión de Fibonacci (precisión arbitraria y razón áurea).
- Aritmética de punto flotante normalizado de k dígitos (redondeo simétrico vs. truncamiento).
- Pérdida de asociatividad: (u - v) - w vs. (u - w) - v.
- Cancelación catastrófica en ecuaciones cuadráticas y forma racionalizada.
- Evaluación polinómica estándar vs. Forma anidada de Horner (genérica y TP2).

Diseño: Funciones 100% puras, sin `input()` ni `print()`, listas para UI y comparación.
"""

from __future__ import annotations

import math
import time
from decimal import Decimal, getcontext
from typing import Any


def sumar_n_positivos(numeros: list[float]) -> dict[str, Any]:
    if not numeros:
        raise ValueError("La lista de números no puede estar vacía.")
    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()
    total = 0.0
    muestra: list[dict[str, Any]] = []
    for i, x in enumerate(numeros, start=1):
        if x <= 0:
            raise ValueError(f"Valor no positivo detectado en índice {i}: {x}. Se requiere X_i > 0.")
        total += x
        if i <= 100:
            muestra.append({"indice": i, "x_i": x, "suma_acumulada": total})
    return {
        "total": total,
        "n": len(numeros),
        "t_clock": time.perf_counter() - t_ini_clock,
        "t_cpu": time.process_time() - t_ini_cpu,
        "muestra_acumulada": muestra,
    }


def sumar_pares(numeros: list[int]) -> dict[str, Any]:
    if not numeros:
        raise ValueError("La lista de enteros no puede estar vacía.")
    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()
    suma_p = 0
    cant_p = 0
    clasificacion: list[dict[str, Any]] = []
    for i, x in enumerate(numeros, start=1):
        es_par = x % 2 == 0
        if es_par:
            suma_p += x
            cant_p += 1
        clasificacion.append({
            "indice": i,
            "x_i": x,
            "es_par": es_par,
            "estado": "Par (Se suma)" if es_par else "Impar (Se omite)",
            "aporte": x if es_par else 0,
        })
    n_total = len(numeros)
    return {
        "suma_pares": suma_p,
        "cant_pares": cant_p,
        "total_evaluados": n_total,
        "porcentaje_pares": (cant_p / n_total * 100.0) if n_total > 0 else 0.0,
        "t_clock": time.perf_counter() - t_ini_clock,
        "t_cpu": time.process_time() - t_ini_cpu,
        "clasificacion": clasificacion,
    }


def generar_fibonacci(n: int) -> dict[str, Any]:
    if n <= 0:
        raise ValueError("La cantidad de términos N debe ser un entero positivo.")
    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()
    if n == 1:
        fib = [0]
    elif n == 2:
        fib = [0, 1]
    else:
        fib = [0, 1]
        for _ in range(2, n):
            fib.append(fib[-1] + fib[-2])
    t_clock = time.perf_counter() - t_ini_clock
    t_cpu = time.process_time() - t_ini_cpu
    phi_exacto = (1.0 + math.sqrt(5.0)) / 2.0
    filas = []
    for i, val in enumerate(fib):
        if i >= 2 and fib[i - 1] != 0:
            cociente: float | None = fib[i] / fib[i - 1]
            dist_phi: float | None = abs(cociente - phi_exacto)
        else:
            cociente = None
            dist_phi = None
        filas.append({
            "orden": i + 1,
            "valor": val,
            "valor_str": str(val),
            "cifras": len(str(val)),
            "cociente": cociente,
            "distancia_phi": dist_phi,
        })
    return {
        "terminos": fib,
        "ultimo_termino": fib[-1],
        "cifras_ultimo": len(str(fib[-1])),
        "phi_exacto": phi_exacto,
        "t_clock": t_clock,
        "t_cpu": t_cpu,
        "tabla": filas,
    }


def resolver_cuadratica(a: float, b: float, c: float) -> dict[str, Any]:
    if abs(a) < 1e-15:
        raise ValueError("El coeficiente cuadrático 'a' no puede ser cero.")
    t_ini = time.perf_counter()
    disc = b**2 - 4.0 * a * c
    xv = -b / (2.0 * a)
    yv = a * (xv**2) + b * xv + c
    if disc < -1e-15:
        real_p = -b / (2.0 * a)
        imag_p = math.sqrt(-disc) / (2.0 * a)
        raices = (complex(real_p, imag_p), complex(real_p, -imag_p))
        tipo = "complejas"
        estado = "Raíces Complejas Conjugadas (Δ < 0)"
    elif abs(disc) <= 1e-15:
        x_unica = -b / (2.0 * a)
        raices = (x_unica, x_unica)
        tipo = "real_doble"
        estado = "Raíz Real Doble Única (Δ = 0)"
    else:
        sqrt_d = math.sqrt(disc)
        raices = ((-b + sqrt_d) / (2.0 * a), (-b - sqrt_d) / (2.0 * a))
        tipo = "reales_distintas"
        estado = "Dos Raíces Reales Distintas (Δ > 0)"
    return {
        "a": a, "b": b, "c": c, "discriminante": disc,
        "tipo": tipo, "estado": estado, "raices": raices,
        "vertice": (xv, yv), "duracion": time.perf_counter() - t_ini,
    }


def calcular_serie_euler(
    n_max: int,
    x_val: float = 1.0,
    cifras_scarborough: int = 3,
    precision_dec: int = 50,
) -> dict[str, Any]:
    """
    Calcula la aproximación de e^x = sum_{k=0}^{n_max} x^k / k! término a término.
    Calcula también el error relativo porcentual aproximado E_rp respecto al criterio de Scarborough.
    """
    if n_max < 0:
        raise ValueError("n_max debe ser un entero no negativo.")
    getcontext().prec = precision_dec
    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()

    es_scarborough_pct = 0.5 * (10.0 ** (2 - cifras_scarborough))
    val_real = math.exp(x_val)
    suma_float = 1.0
    fact_float = 1.0
    suma_dec = Decimal(1)
    fact_dec = Decimal(1)
    x_dec = Decimal(str(x_val))

    historial: list[dict[str, Any]] = [{
        "k": 0,
        "expresion": "1 / 0!",
        "termino": 1.0,
        "suma_parcial": suma_float,
        "valor_exacto": val_real,
        "error_absoluto": abs(val_real - suma_float),
        "error_relativo_pct": (abs(val_real - suma_float) / abs(val_real) * 100.0) if val_real != 0 else 0.0,
        "error_aprox_rp_pct": None,
        "cumple_scarborough": False,
    }]

    termino_scarborough = None
    for k in range(1, n_max + 1):
        suma_ant = suma_float
        fact_float *= k
        term_f = (x_val**k) / fact_float
        suma_float += term_f

        fact_dec *= Decimal(k)
        suma_dec += (x_dec**k) / fact_dec

        err_abs = abs(val_real - suma_float)
        err_rel = (err_abs / abs(val_real) * 100.0) if val_real != 0 else 0.0
        e_rp = (abs(suma_float - suma_ant) / abs(suma_float) * 100.0) if abs(suma_float) > 1e-15 else 0.0
        cumple_sc = e_rp < es_scarborough_pct
        if cumple_sc and termino_scarborough is None:
            termino_scarborough = k

        expr_str = f"{x_val:g}^{k} / {k}!" if x_val != 1.0 else f"1 / {k}!"
        historial.append({
            "k": k,
            "expresion": expr_str,
            "termino": term_f,
            "suma_parcial": suma_float,
            "valor_exacto": val_real,
            "error_absoluto": err_abs,
            "error_relativo_pct": err_rel,
            "error_aprox_rp_pct": e_rp,
            "cumple_scarborough": cumple_sc,
        })

    return {
        "n_max": n_max,
        "x_val": x_val,
        "cifras_scarborough": cifras_scarborough,
        "es_scarborough_pct": es_scarborough_pct,
        "termino_scarborough": termino_scarborough,
        "aproximacion_float": suma_float,
        "aproximacion_decimal_str": str(suma_dec),
        "valor_exacto": val_real,
        "error_absoluto_final": abs(val_real - suma_float),
        "error_relativo_final_pct": (abs(val_real - suma_float) / abs(val_real) * 100.0) if val_real != 0 else 0.0,
        "t_clock": time.perf_counter() - t_ini_clock,
        "t_cpu": time.process_time() - t_ini_cpu,
        "historial": historial,
    }


def calcular_pi_leibniz_tolerancia(decimales: int, max_iter: int = 500000) -> dict[str, Any]:
    if decimales <= 0:
        raise ValueError("La cantidad de decimales exactos debe ser mayor a cero.")
    tol = 0.5 * (10.0 ** (-decimales))
    pi_ref = math.pi
    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()
    pi_aprox = 0.0
    signo = 1.0
    k = 0
    error = 1.0
    while error >= tol and k < max_iter:
        pi_aprox += signo * (4.0 / (2 * k + 1))
        signo = -signo
        k += 1
        error = abs(pi_ref - pi_aprox)
    return {
        "decimales_solicitados": decimales,
        "tolerancia": tol,
        "terminos_utilizados": k,
        "pi_aproximado": pi_aprox,
        "pi_exacto": pi_ref,
        "error_absoluto": error,
        "convergente": error < tol,
        "t_clock": time.perf_counter() - t_ini_clock,
        "t_cpu": time.process_time() - t_ini_cpu,
    }


def calcular_pi_leibniz_historial(n_terminos: int, max_muestras: int = 200) -> dict[str, Any]:
    if n_terminos <= 0:
        raise ValueError("n_terminos debe ser un entero positivo.")
    pi_ref = math.pi
    pi_aprox = 0.0
    signo = 1.0
    historial: list[dict[str, Any]] = []
    paso_muestreo = max(1, n_terminos // max_muestras)
    t_ini = time.perf_counter()
    for k in range(n_terminos):
        termino = signo * (4.0 / (2 * k + 1))
        pi_aprox += termino
        signo = -signo
        err = abs(pi_ref - pi_aprox)
        if k < 25 or (k + 1) % paso_muestreo == 0 or k == n_terminos - 1:
            historial.append({
                "k": k + 1,
                "termino_expr": f"{'+' if k % 2 == 0 else '-'} 4/{2*k + 1}",
                "termino_num": termino,
                "pi_aprox": pi_aprox,
                "pi_exacto": pi_ref,
                "error_absoluto": err,
                "error_relativo_pct": (err / pi_ref) * 100.0,
            })
    return {
        "n_terminos": n_terminos,
        "pi_aproximado": pi_aprox,
        "pi_exacto": pi_ref,
        "error_absoluto": abs(pi_ref - pi_aprox),
        "error_relativo_pct": (abs(pi_ref - pi_aprox) / pi_ref) * 100.0,
        "duracion": time.perf_counter() - t_ini,
        "historial": historial,
    }


def estimar_cifras_significativas(er: float) -> int:
    if er <= 0.0:
        return 16
    if math.isinf(er) or math.isnan(er):
        return 0
    t = 0
    while er <= 0.5 * (10.0 ** (1.0 - (t + 1))) and t < 16:
        t += 1
    return t


def calcular_errores(p_exacto: float, p_aprox: float) -> dict[str, Any]:
    t_ini = time.perf_counter()
    ea = abs(p_exacto - p_aprox)
    if abs(p_exacto) > 1e-15:
        er = ea / abs(p_exacto)
        ep = er * 100.0
    else:
        er = float("inf") if ea > 0 else 0.0
        ep = float("inf") if ea > 0 else 0.0
    return {
        "p_exacto": p_exacto,
        "p_aprox": p_aprox,
        "error_absoluto": ea,
        "error_relativo": er,
        "error_porcentual": ep,
        "cifras_significativas": estimar_cifras_significativas(er),
        "tiempo_ejecucion": time.perf_counter() - t_ini,
    }


def obtener_casos_estudio_errores() -> list[dict[str, Any]]:
    casos = [
        ("a", "p = π, p* = 22/7", math.pi, 22.0 / 7.0),
        ("b", "p = π, p* = 3.1416", math.pi, 3.1416),
        ("c", "p = e, p* = 2.718", math.e, 2.718),
        ("d", "p = √2, p* = 1.414", math.sqrt(2.0), 1.414),
        ("e", "p = e¹⁰, p* = 22000", math.exp(10.0), 22000.0),
        ("f", "p = 8!, p* = 39900", float(math.factorial(8)), 39900.0),
    ]
    res = []
    for item, desc, p_ex, p_ap in casos:
        info = calcular_errores(p_ex, p_ap)
        info["item"] = item
        info["descripcion"] = desc
        res.append(info)
    return res


def fl_round(val: float, k: int = 3) -> tuple[float, int]:
    if math.isnan(val) or math.isinf(val) or val == 0.0:
        return val, 0
    sign = 1.0 if val > 0 else -1.0
    val_abs = abs(val)
    exp = math.floor(math.log10(val_abs)) + 1
    mantissa = val_abs / (10.0**exp)
    mantissa_r = round(mantissa, k)
    if mantissa_r >= 1.0:
        mantissa_r /= 10.0
        exp += 1
    return sign * mantissa_r * (10.0**exp), exp


def fl_trunc(val: float, k: int = 3) -> tuple[float, int]:
    if math.isnan(val) or math.isinf(val) or val == 0.0:
        return val, 0
    sign = 1.0 if val > 0 else -1.0
    val_abs = abs(val)
    exp = math.floor(math.log10(val_abs)) + 1
    mantissa = val_abs / (10.0**exp)
    factor = 10.0**k
    mantissa_t = math.floor(mantissa * factor + 1e-12) / factor
    return sign * mantissa_t * (10.0**exp), exp


def format_norm(val: float, k: int = 3) -> str:
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "-Inf" if val < 0 else "+Inf"
    if val == 0.0:
        return "0." + "0" * k + " x 10^0"
    sign = "-" if val < 0 else ""
    val_abs = abs(val)
    exp = math.floor(math.log10(val_abs)) + 1
    mantissa = val_abs / (10.0**exp)
    mantissa_r = round(mantissa, k)
    if mantissa_r >= 1.0:
        mantissa_r /= 10.0
        exp += 1
    return f"{sign}{mantissa_r:.{k}f} x 10^{exp}"


def format_norm_trunc(val: float, k: int = 3) -> str:
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "-Inf" if val < 0 else "+Inf"
    if val == 0.0:
        return "0." + "0" * k + " x 10^0"
    sign = "-" if val < 0 else ""
    val_abs = abs(val)
    exp = math.floor(math.log10(val_abs)) + 1
    mantissa = val_abs / (10.0**exp)
    factor = 10.0**k
    mantissa_t = math.floor(mantissa * factor + 1e-12) / factor
    return f"{sign}{mantissa_t:.{k}f} x 10^{exp}"


def operar_en_maquina(x: float, y: float, operacion: str, k: int = 3) -> dict[str, Any]:
    op = operacion.strip()
    if op in ["÷", "/", "División (÷)"] and abs(y) < 1e-15:
        raise ZeroDivisionError("División por cero en máquina.")
    xr, _ = fl_round(x, k)
    yr, _ = fl_round(y, k)
    xt, _ = fl_trunc(x, k)
    yt, _ = fl_trunc(y, k)

    if op in ["+", "Suma (+)"]:
        exacto = x + y
        res_r, _ = fl_round(xr + yr, k)
        res_t, _ = fl_trunc(xt + yt, k)
        simbolo = "+"
    elif op in ["-", "Resta (-)"]:
        exacto = x - y
        res_r, _ = fl_round(xr - yr, k)
        res_t, _ = fl_trunc(xt - yt, k)
        simbolo = "-"
    elif op in ["*", "x", "×", "Multiplicación (×)"]:
        exacto = x * y
        res_r, _ = fl_round(xr * yr, k)
        res_t, _ = fl_trunc(xt * yt, k)
        simbolo = "×"
    elif op in ["/", "÷", "División (÷)"]:
        exacto = x / y
        res_r, _ = fl_round(xr / yr, k)
        res_t, _ = fl_trunc(xt / yt, k)
        simbolo = "÷"
    else:
        raise ValueError(f"Operación no soportada: {operacion}")

    ea_r = abs(exacto - res_r)
    er_r = (ea_r / abs(exacto) * 100.0) if abs(exacto) > 1e-15 else 0.0
    ea_t = abs(exacto - res_t)
    er_t = (ea_t / abs(exacto) * 100.0) if abs(exacto) > 1e-15 else 0.0
    return {
        "x": x, "y": y, "operacion": simbolo, "k_digitos": k, "exacto": exacto,
        "redondeo": {"valor": res_r, "norm_str": format_norm(res_r, k), "error_absoluto": ea_r, "error_relativo_pct": er_r},
        "truncamiento": {"valor": res_t, "norm_str": format_norm_trunc(res_t, k), "error_absoluto": ea_t, "error_relativo_pct": er_t},
    }


def comparar_asociatividad_personalizada(
    u: float = 121.0,
    v: float = 119.0,
    w: float = 0.327,
    k: int = 3,
) -> dict[str, Any]:
    """Evalúa (u - v) - w vs. (u - w) - v en aritmética normalizada de k dígitos."""
    exacto = (u - v) - w
    ur, vr, wr = fl_round(u, k)[0], fl_round(v, k)[0], fl_round(w, k)[0]
    ut, vt, wt = fl_trunc(u, k)[0], fl_trunc(v, k)[0], fl_trunc(w, k)[0]

    # Camino 1: (u - v) - w
    p1_r = fl_round(ur - vr, k)[0]
    res1_r = fl_round(p1_r - wr, k)[0]
    p1_t = fl_trunc(ut - vt, k)[0]
    res1_t = fl_trunc(p1_t - wt, k)[0]

    # Camino 2: (u - w) - v
    p2_r = fl_round(ur - wr, k)[0]
    res2_r = fl_round(p2_r - vr, k)[0]
    p2_t = fl_trunc(ut - wt, k)[0]
    res2_t = fl_trunc(p2_t - vt, k)[0]

    def _err_pct(ap: float) -> float:
        return (abs(exacto - ap) / abs(exacto) * 100.0) if abs(exacto) > 1e-15 else 0.0

    return {
        "exacto": exacto,
        "k_digitos": k,
        "tabla": [
            {
                "agrupacion": f"({u:g} - {v:g}) - {w:g}",
                "exacto": exacto,
                "redondeo_norm": format_norm(res1_r, k),
                "redondeo_val": res1_r,
                "error_redondeo_pct": _err_pct(res1_r),
                "trunc_norm": format_norm_trunc(res1_t, k),
                "trunc_val": res1_t,
                "error_trunc_pct": _err_pct(res1_t),
                "paso_intermedio_redondeo": p1_r,
            },
            {
                "agrupacion": f"({u:g} - {w:g}) - {v:g}",
                "exacto": exacto,
                "redondeo_norm": format_norm(res2_r, k),
                "redondeo_val": res2_r,
                "error_redondeo_pct": _err_pct(res2_r),
                "trunc_norm": format_norm_trunc(res2_t, k),
                "trunc_val": res2_t,
                "error_trunc_pct": _err_pct(res2_t),
                "paso_intermedio_redondeo": p2_r,
            },
        ],
    }


def casos_estudio_asociatividad(k: int = 3) -> dict[str, Any]:
    """Ejecuta los 5 incisos del TP2 (Ej. 2 y 3) con parámetro k variable en vivo."""
    casos = [
        ("a", "133 + 0.921", 133.0 + 0.921,
         lambda fl: fl(fl(133.0, k)[0] + fl(0.921, k)[0], k)[0],
         "fl(fl(133) + fl(0.921))"),
        ("b", "133 - 0.499", 133.0 - 0.499,
         lambda fl: fl(fl(133.0, k)[0] - fl(0.499, k)[0], k)[0],
         "fl(fl(133) - fl(0.499))"),
        ("c", "(121 - 119) - 0.327", (121.0 - 119.0) - 0.327,
         lambda fl: fl(fl(fl(121.0, k)[0] - fl(119.0, k)[0], k)[0] - fl(0.327, k)[0], k)[0],
         "Agrupación óptima: términos cercanos primero"),
        ("d", "(121 - 0.327) - 119", (121.0 - 0.327) - 119.0,
         lambda fl: fl(fl(fl(121.0, k)[0] - fl(0.327, k)[0], k)[0] - fl(119.0, k)[0], k)[0],
         "Absorción del término pequeño (pérdida de asociatividad)"),
        ("e", "(2/9) × (9/7)", 2.0 / 7.0,
         lambda fl: fl(fl(2.0 / 9.0, k)[0] * fl(9.0 / 7.0, k)[0], k)[0],
         "fl(fl(2/9) × fl(9/7))"),
    ]
    tabla = []
    for item, op_str, exacto, fn, detalle in casos:
        vr = fn(fl_round)
        vt = fn(fl_trunc)
        ea_r = abs(exacto - vr)
        er_r = (ea_r / abs(exacto) * 100.0) if abs(exacto) > 1e-15 else 0.0
        ea_t = abs(exacto - vt)
        er_t = (ea_t / abs(exacto) * 100.0) if abs(exacto) > 1e-15 else 0.0
        tabla.append({
            "item": item, "operacion": op_str, "exacto": exacto,
            "redondeo_norm": format_norm(vr, k), "redondeo_val": vr, "error_rel_redondeo_pct": er_r,
            "trunc_norm": format_norm_trunc(vt, k), "trunc_val": vt, "error_rel_trunc_pct": er_t,
            "detalle_paso": detalle,
        })
    c_r = tabla[2]["redondeo_val"]
    d_r = tabla[3]["redondeo_val"]
    ec = tabla[2]["error_rel_redondeo_pct"]
    ed = tabla[3]["error_rel_redondeo_pct"]
    return {
        "k_digitos": k,
        "tabla": tabla,
        "analisis_asociatividad": {
            "p_teorico": tabla[2]["exacto"],
            "resultado_agrupado_c": c_r,
            "error_pct_c": ec,
            "resultado_agrupado_d": d_r,
            "error_pct_d": ed,
            "conclusion": (
                f"En k={k} dígitos, (121-119)-0.327 da {c_r:g} (Error {ec:.2f}%), mientras que "
                f"(121-0.327)-119 sufre absorción dando {d_r:g} (Error {ed:.2f}%)."
            ),
        },
    }


def analizar_cancelacion_cuadratica(
    a: float = 1.0,
    b: float = 62.10,
    c: float = 1.0,
    k: int = 4,
) -> dict[str, Any]:
    if abs(a) < 1e-15:
        raise ValueError("El coeficiente 'a' no puede ser cero.")
    disc_real = b**2 - 4.0 * a * c
    if disc_real <= 0:
        raise ValueError("Se requiere discriminante estrictamente positivo (b² - 4ac > 0).")

    if b >= 0:
        x2_exact = (-b - math.sqrt(disc_real)) / (2.0 * a)
        x1_exact = (-2.0 * c) / (b + math.sqrt(disc_real))
    else:
        x1_exact = (-b + math.sqrt(disc_real)) / (2.0 * a)
        x2_exact = (-2.0 * c) / (b - math.sqrt(disc_real))

    fl_a, _ = fl_round(a, k)
    fl_b, _ = fl_round(b, k)
    fl_c, _ = fl_round(c, k)
    fl_b_sq, _ = fl_round(fl_b**2, k)
    fl_4ac, _ = fl_round(4.0 * fl_a * fl_c, k)
    fl_disc, _ = fl_round(fl_b_sq - fl_4ac, k)
    if fl_disc < 0:
        fl_disc = 0.0
    fl_sqrt_disc, _ = fl_round(math.sqrt(fl_disc), k)
    fl_2a, _ = fl_round(2.0 * fl_a, k)
    fl_neg_2c, _ = fl_round(-2.0 * fl_c, k)

    num1_trad, _ = fl_round(-fl_b + fl_sqrt_disc, k)
    x1_trad = fl_round(num1_trad / fl_2a, k)[0] if fl_2a != 0 else float("nan")
    num2_trad, _ = fl_round(-fl_b - fl_sqrt_disc, k)
    x2_trad = fl_round(num2_trad / fl_2a, k)[0] if fl_2a != 0 else float("nan")

    den1_rat, _ = fl_round(fl_b + fl_sqrt_disc, k)
    x1_rat = fl_round(fl_neg_2c / den1_rat, k)[0] if den1_rat != 0 else float("nan")
    den2_rat, _ = fl_round(fl_b - fl_sqrt_disc, k)
    x2_rat = fl_round(fl_neg_2c / den2_rat, k)[0] if den2_rat != 0 else float("nan")

    metodos = [
        ("x1_tradicional", "x₁ Tradicional: (-b + √Δ)/(2a)", x1_exact, x1_trad, b > 0,
         "Inestable si b > 0: resta casi igual en numerador." if b > 0 else "Estable si b < 0."),
        ("x1_racionalizada", "x₁ Racionalizada: -2c/(b + √Δ)", x1_exact, x1_rat, b < 0,
         "Estable si b > 0: suma términos de igual signo." if b > 0 else "Inestable si b < 0."),
        ("x2_tradicional", "x₂ Tradicional: (-b - √Δ)/(2a)", x2_exact, x2_trad, b < 0,
         "Estable si b > 0: suma de términos negativos." if b > 0 else "Inestable si b < 0."),
        ("x2_racionalizada", "x₂ Racionalizada: -2c/(b - √Δ)", x2_exact, x2_rat, b > 0,
         "Inestable si b > 0: resta casi igual en denominador." if b > 0 else "Estable si b < 0."),
    ]
    tabla = []
    for clave, nombre, ex, ap, cancel, diag in metodos:
        ea = abs(ex - ap) if not math.isnan(ap) else float("nan")
        er_pct = (ea / abs(ex) * 100.0) if (abs(ex) > 1e-15 and not math.isnan(ea)) else float("nan")
        tabla.append({
            "clave": clave, "metodo": nombre, "exacto": ex,
            "aprox_num": ap, "aprox_norm": format_norm(ap, k),
            "error_absoluto": ea, "error_absoluto_norm": format_norm(ea, k),
            "error_relativo_pct": er_pct, "sufre_cancelacion": cancel, "diagnostico": diag,
        })

    return {
        "coeficientes": {"a": a, "b": b, "c": c},
        "k_digitos": k,
        "pasos_maquina": {
            "fl_a": fl_a, "fl_b": fl_b, "fl_c": fl_c,
            "fl_b_sq": fl_b_sq, "fl_4ac": fl_4ac, "fl_disc": fl_disc,
            "fl_sqrt_disc": fl_sqrt_disc, "fl_2a": fl_2a, "fl_neg_2c": fl_neg_2c,
        },
        "tabla_comparativa": tabla,
        "x1_exacto": x1_exact,
        "x2_exacto": x2_exact,
    }


def evaluar_polinomio_horner(
    x_val: float = 4.71,
    k: int = 3,
    coefs: list[float] | None = None,
) -> dict[str, Any]:
    """
    Evalúa un polinomio a_n x^n + ... + a_0 en x_val con k dígitos significativos,
    contrastando Forma Estándar Directa vs. Forma Anidada de Horner (truncamiento y redondeo).
    Por defecto usa el polinomio de TP2: [1.0, -6.0, 3.0, -0.149].
    """
    if coefs is None:
        coefs = [1.0, -6.0, 3.0, -0.149]
    if len(coefs) < 2:
        raise ValueError("El polinomio debe tener al menos grado 1 (2 coeficientes).")

    grado = len(coefs) - 1

    # 1. Exacto
    f_ex = 0.0
    for c in coefs:
        f_ex = f_ex * x_val + c

    # 2. Forma Estándar (calcula potencias x^j explícitamente)
    def _eval_estandar(fl_fn) -> float:
        xf = fl_fn(x_val, k)[0]
        potencias = [1.0, xf]
        for _ in range(2, grado + 1):
            potencias.append(fl_fn(potencias[-1] * xf, k)[0])
        acum = 0.0
        for idx, c in enumerate(coefs):
            p = grado - idx
            cf = fl_fn(c, k)[0]
            term = fl_fn(cf * potencias[p], k)[0] if p > 0 else cf
            acum = fl_fn(acum + term, k)[0]
        return acum

    # 3. Forma Anidada de Horner
    def _eval_horner(fl_fn) -> tuple[float, list[dict[str, Any]]]:
        xf = fl_fn(x_val, k)[0]
        acum = fl_fn(coefs[0], k)[0]
        pasos = [{"paso": 0, "operacion": f"Inicio b_{grado} = {acum:g}", "valor": acum}]
        for idx, c in enumerate(coefs[1:], start=1):
            cf = fl_fn(c, k)[0]
            prod = fl_fn(acum * xf, k)[0]
            acum = fl_fn(prod + cf, k)[0]
            pasos.append({
                "paso": idx,
                "operacion": f"fl(fl(acum × {xf:g}) + {cf:g})",
                "valor": acum,
            })
        return acum, pasos

    f_trad_tr = _eval_estandar(fl_trunc)
    f_trad_rd = _eval_estandar(fl_round)
    f_anid_tr, pasos_tr = _eval_horner(fl_trunc)
    f_anid_rd, pasos_rd = _eval_horner(fl_round)

    casos_comparacion = [
        ("Forma Estándar (Truncado)", f_trad_tr),
        ("Forma Estándar (Redondeado)", f_trad_rd),
        ("Forma Anidada de Horner (Truncado)", f_anid_tr),
        ("Forma Anidada de Horner (Redondeado)", f_anid_rd),
    ]
    tabla_errores = []
    for nombre, p_star in casos_comparacion:
        ea = abs(f_ex - p_star)
        er_pct = (ea / abs(f_ex) * 100.0) if abs(f_ex) > 1e-15 else 0.0
        tabla_errores.append({
            "metodo": nombre,
            "aproximacion": p_star,
            "aproximacion_norm": format_norm(p_star, k),
            "error_absoluto": ea,
            "error_relativo_pct": er_pct,
        })

    return {
        "x": x_val,
        "k_digitos": k,
        "coefs": coefs,
        "f_exacto": f_ex,
        "pasos_horner_trunc": pasos_tr,
        "pasos_horner_round": pasos_rd,
        "tabla_errores": tabla_errores,
    }

