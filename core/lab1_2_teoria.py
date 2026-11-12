"""
core/lab1_2_teoria.py
====================
Módulo de lógica matemática pura para los Laboratorios 1 y 2.

Cubre:
- Lab 1: Modelos algorítmicos, sumas de N números, sumas condicionales (pares),
  sucesión de Fibonacci (precisión arbitraria y convergencia al número áureo),
  resolución analítica de la cuadrática, series de Taylor para e^x y pi
  (Gregory-Leibniz con criterio de Scarborough), conteo de términos en series impares.
- Lab 2: Teoría de errores (absoluto, relativo, porcentual, cifras significativas),
  aritmética de punto flotante normalizado de k dígitos (redondeo simétrico vs. truncamiento),
  demostración de pérdida de asociatividad, cancelación catastrófica en raíces cuadráticas
  y racionalización, evaluación polinómica estándar vs. forma anidada de Horner.

Regla de diseño: Funciones 100% puras, sin `input()` ni `print()`, con retornos estructurados.
"""

from __future__ import annotations

import math
import time
from decimal import Decimal, getcontext
from typing import Any


# ==============================================================================
# SECCIÓN 1: LABORATORIO 1 — MODELOS ALGORÍTMICOS, TIEMPOS Y SERIES
# ==============================================================================

def sumar_n_positivos(numeros: list[float]) -> dict[str, Any]:
    """
    Calcula la suma de una lista de números positivos midiendo tiempos de Reloj y CPU.
    Valida que todos los elementos sean estrictamente positivos (X_i > 0).

    Retorna un diccionario con:
    - total: suma acumulada.
    - n: cantidad de elementos.
    - t_clock: tiempo de pared en segundos.
    - t_cpu: tiempo de proceso CPU en segundos.
    - muestra_acumulada: lista de diccionarios con el paso a paso de los primeros 100 elementos.
    """
    if not numeros:
        raise ValueError("La lista de números no puede estar vacía.")

    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()

    total = 0.0
    muestra: list[dict[str, Any]] = []
    guardar_muestra = len(numeros) <= 200 or True

    for i, x in enumerate(numeros, start=1):
        if x <= 0:
            raise ValueError(f"Valor no positivo detectado en índice {i}: {x}. Se requiere X_i > 0.")
        total += x
        if guardar_muestra and i <= 100:
            muestra.append({
                "indice": i,
                "x_i": x,
                "suma_acumulada": total,
            })

    t_clock = time.perf_counter() - t_ini_clock
    t_cpu = time.process_time() - t_ini_cpu

    return {
        "total": total,
        "n": len(numeros),
        "t_clock": t_clock,
        "t_cpu": t_cpu,
        "muestra_acumulada": muestra,
    }


def sumar_pares(numeros: list[int]) -> dict[str, Any]:
    """
    Suma únicamente los valores pares de una lista de enteros positivos y mide tiempos.

    Retorna un diccionario con:
    - suma_pares: suma de los elementos pares.
    - cant_pares: cantidad de números pares encontrados.
    - total_evaluados: cantidad total de números en la lista.
    - porcentaje_pares: porcentaje de números pares respecto al total.
    - t_clock: tiempo de reloj en segundos.
    - t_cpu: tiempo de CPU en segundos.
    - clasificacion: lista con la clasificación de paridad de cada elemento.
    """
    if not numeros:
        raise ValueError("La lista de enteros no puede estar vacía.")

    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()

    suma_p = 0
    cant_p = 0
    clasificacion: list[dict[str, Any]] = []

    for i, x in enumerate(numeros, start=1):
        es_par = (x % 2 == 0)
        aporte = x if es_par else 0
        if es_par:
            suma_p += x
            cant_p += 1

        clasificacion.append({
            "indice": i,
            "x_i": x,
            "es_par": es_par,
            "estado": "Par (Se suma)" if es_par else "Impar (Se omite)",
            "aporte": aporte,
        })

    t_clock = time.perf_counter() - t_ini_clock
    t_cpu = time.process_time() - t_ini_cpu

    n_total = len(numeros)
    pct_pares = (cant_p / n_total * 100.0) if n_total > 0 else 0.0

    return {
        "suma_pares": suma_p,
        "cant_pares": cant_p,
        "total_evaluados": n_total,
        "porcentaje_pares": pct_pares,
        "t_clock": t_clock,
        "t_cpu": t_cpu,
        "clasificacion": clasificacion,
    }


def generar_fibonacci(n: int) -> dict[str, Any]:
    """
    Genera los primeros N términos de la sucesión de Fibonacci (0, 1, 1, 2, 3, 5, ...).
    Calcula los cocientes consecutivos F_n / F_{n-1} y su aproximación a la Razón Áurea phi.

    Python maneja enteros con precisión arbitraria, evitando desbordamientos para N >= 200.
    """
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

    filas: list[dict[str, Any]] = []
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
    """
    Resuelve analíticamente la ecuación cuadrática ax^2 + bx + c = 0.
    Identifica raíces reales distintas, raíz real doble o raíces complejas conjugadas.
    Calcula además el vértice de la parábola (x_v, y_v).
    """
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
        x1 = (-b + sqrt_d) / (2.0 * a)
        x2 = (-b - sqrt_d) / (2.0 * a)
        raices = (x1, x2)
        tipo = "reales_distintas"
        estado = "Dos Raíces Reales Distintas (Δ > 0)"

    duracion = time.perf_counter() - t_ini

    return {
        "a": a,
        "b": b,
        "c": c,
        "discriminante": disc,
        "tipo": tipo,
        "estado": estado,
        "raices": raices,
        "vertice": (xv, yv),
        "duracion": duracion,
    }


def calcular_serie_euler(
    n_max: int,
    x_val: float = 1.0,
    precision_dec: int = 50,
) -> dict[str, Any]:
    """
    Calcula la aproximación de e^x = sum_{k=0}^{n_max} x^k / k! término a término.
    Genera la evolución numérica tanto en float estándar como en alta precisión (Decimal).
    """
    if n_max < 0:
        raise ValueError("n_max debe ser un entero no negativo.")

    getcontext().prec = precision_dec
    t_ini_clock = time.perf_counter()
    t_ini_cpu = time.process_time()

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
    }]

    for k in range(1, n_max + 1):
        fact_float *= k
        term_f = (x_val**k) / fact_float
        suma_float += term_f

        fact_dec *= Decimal(k)
        suma_dec += (x_dec**k) / fact_dec

        err_abs = abs(val_real - suma_float)
        err_rel = (err_abs / abs(val_real) * 100.0) if val_real != 0 else 0.0

        expr_str = f"{x_val:g}^{k} / {k}!" if x_val != 1.0 else f"1 / {k}!"
        historial.append({
            "k": k,
            "expresion": expr_str,
            "termino": term_f,
            "suma_parcial": suma_float,
            "valor_exacto": val_real,
            "error_absoluto": err_abs,
            "error_relativo_pct": err_rel,
        })

    t_clock = time.perf_counter() - t_ini_clock
    t_cpu = time.process_time() - t_ini_cpu

    return {
        "n_max": n_max,
        "x_val": x_val,
        "aproximacion_float": suma_float,
        "aproximacion_decimal_str": str(suma_dec),
        "valor_exacto": val_real,
        "error_absoluto_final": abs(val_real - suma_float),
        "error_relativo_final_pct": (abs(val_real - suma_float) / abs(val_real) * 100.0) if val_real != 0 else 0.0,
        "t_clock": t_clock,
        "t_cpu": t_cpu,
        "historial": historial,
    }


def calcular_pi_leibniz_tolerancia(
    decimales: int,
    max_iter: int = 500000,
) -> dict[str, Any]:
    """
    Calcula Pi mediante la serie de Gregory-Leibniz:
        pi = 4 * sum_{k=0}^{inf} (-1)^k / (2k + 1)
    iterando hasta cumplir el criterio de Scarborough: tol = 0.5 * 10^(-decimales).
    """
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
        termino = signo * (4.0 / (2 * k + 1))
        pi_aprox += termino
        signo = -signo
        k += 1
        error = abs(pi_ref - pi_aprox)

    t_clock = time.perf_counter() - t_ini_clock
    t_cpu = time.process_time() - t_ini_cpu

    convergente = error < tol

    return {
        "decimales_solicitados": decimales,
        "tolerancia": tol,
        "terminos_utilizados": k,
        "pi_aproximado": pi_aprox,
        "pi_exacto": pi_ref,
        "error_absoluto": error,
        "convergente": convergente,
        "t_clock": t_clock,
        "t_cpu": t_cpu,
    }


def calcular_pi_leibniz_historial(
    n_terminos: int,
    max_muestras: int = 200,
) -> dict[str, Any]:
    """
    Genera la evolución término a término de la serie de Gregory-Leibniz para Pi.
    Aplica submuestreo inteligente para no saturar memoria si n_terminos es grande.
    """
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

    duracion = time.perf_counter() - t_ini

    return {
        "n_terminos": n_terminos,
        "pi_aproximado": pi_aprox,
        "pi_exacto": pi_ref,
        "error_absoluto": abs(pi_ref - pi_aprox),
        "error_relativo_pct": (abs(pi_ref - pi_aprox) / pi_ref) * 100.0,
        "duracion": duracion,
        "historial": historial,
    }


def conteo_terminos_serie_impar(denominador_ultimo: int) -> dict[str, int]:
    """
    Determina la cantidad de términos N de la serie de fracciones con denominadores
    impares sucesivos: 1 + 1/3 + 1/5 + ... + 1/(2i+1), conociendo el último denominador.
    """
    if denominador_ultimo <= 0 or denominador_ultimo % 2 == 0:
        raise ValueError("El denominador del último término debe ser un entero positivo e impar.")

    indice_i = (denominador_ultimo - 1) // 2
    n_terminos = indice_i + 1

    return {
        "denominador_ultimo": denominador_ultimo,
        "indice_final_i": indice_i,
        "n_terminos": n_terminos,
    }


# ==============================================================================
# SECCIÓN 2: LABORATORIO 2 — TEORÍA DE ERRORES Y ARITMÉTICA DE MÁQUINA
# ==============================================================================

def calcular_errores(p_exacto: float, p_aprox: float) -> dict[str, Any]:
    """
    Calcula los errores absoluto, relativo y porcentual entre un valor exacto p
    y una aproximación p*. Estima además la cantidad de cifras significativas correctas.
    """
    t_ini = time.perf_counter()

    ea = abs(p_exacto - p_aprox)
    if abs(p_exacto) > 1e-15:
        er = ea / abs(p_exacto)
        ep = er * 100.0
    else:
        er = float("inf") if ea > 0 else 0.0
        ep = float("inf") if ea > 0 else 0.0

    cifras = estimar_cifras_significativas(er)
    t_ejec = time.perf_counter() - t_ini

    return {
        "p_exacto": p_exacto,
        "p_aprox": p_aprox,
        "error_absoluto": ea,
        "error_relativo": er,
        "error_porcentual": ep,
        "cifras_significativas": cifras,
        "tiempo_ejecucion": t_ejec,
    }


def estimar_cifras_significativas(er: float) -> int:
    """
    Calcula el mayor entero no negativo t para el cual p* aproxima a p
    con t cifras significativas según la definición analítica:
        E_r <= 0.5 * 10^(1 - t)  (equivalente a 5 * 10^(-t)).
    """
    if er <= 0.0:
        return 16
    if math.isinf(er) or math.isnan(er):
        return 0

    t = 0
    while er <= 0.5 * (10.0 ** (1.0 - (t + 1))) and t < 16:
        t += 1
    return t


def obtener_casos_estudio_errores() -> list[dict[str, Any]]:
    """
    Retorna los 6 casos de estudio canónicos del Trabajo Práctico 2 (Ejercicio 1)
    con sus cálculos de error exacto, relativo y cifras significativas.
    """
    casos = [
        ("a", "p = π,  p* = 22/7 (Aprox. Arquímedes)", math.pi, 22.0 / 7.0),
        ("b", "p = π,  p* = 3.1416", math.pi, 3.1416),
        ("c", "p = e,  p* = 2.718", math.e, 2.718),
        ("d", "p = √2, p* = 1.414", math.sqrt(2.0), 1.414),
        ("e", "p = e¹⁰, p* = 22000", math.exp(10.0), 22000.0),
        ("f", "p = 8!, p* = 39900", float(math.factorial(8)), 39900.0),
    ]

    resultados = []
    for item, desc, p_ex, p_ap in casos:
        info_err = calcular_errores(p_ex, p_ap)
        info_err["item"] = item
        info_err["descripcion"] = desc
        resultados.append(info_err)

    return resultados


def fl_round(val: float, k: int = 3) -> tuple[float, int]:
    """
    Representa un número en punto flotante decimal normalizado con redondeo simétrico a k dígitos:
        fl(val) = sign * 0.d1...dk * 10^exp
    Retorna la tupla (valor_redondeado, exponente).
    """
    if val == 0.0:
        return 0.0, 0

    sign = 1.0 if val > 0 else -1.0
    val_abs = abs(val)
    exp = math.floor(math.log10(val_abs)) + 1
    mantissa = val_abs / (10.0**exp)

    mantissa_r = round(mantissa, k)
    if mantissa_r >= 1.0:
        mantissa_r /= 10.0
        exp += 1

    val_fl = sign * mantissa_r * (10.0**exp)
    return val_fl, exp


def fl_trunc(val: float, k: int = 3) -> tuple[float, int]:
    """
    Representa un número en punto flotante decimal normalizado con truncamiento (corte) a k dígitos:
        fl(val) = sign * 0.d1...dk * 10^exp
    Retorna la tupla (valor_truncado, exponente).
    """
    if val == 0.0:
        return 0.0, 0

    sign = 1.0 if val > 0 else -1.0
    val_abs = abs(val)
    exp = math.floor(math.log10(val_abs)) + 1
    mantissa = val_abs / (10.0**exp)

    factor = 10.0**k
    mantissa_t = math.floor(mantissa * factor + 1e-12) / factor

    val_fl = sign * mantissa_t * (10.0**exp)
    return val_fl, exp


def format_norm(val: float, k: int = 3) -> str:
    """
    Retorna el string de un valor en notación científica normalizada con redondeo simétrico:
        ±0.d1...dk × 10^exp
    """
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
    """
    Retorna el string de un valor en notación científica normalizada con truncamiento:
        ±0.d1...dk × 10^exp
    """
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


def operar_en_maquina(
    x: float,
    y: float,
    operacion: str,
    k: int = 3,
) -> dict[str, Any]:
    """
    Ejecuta una operación aritmética de máquina en punto flotante normalizado a k dígitos.
    Calcula simultáneamente el resultado con redondeo simétrico y con truncamiento.

    operacion: '+' | '-' | '*' | '/' (o 'x', '÷')
    """
    op = operacion.strip()
    if op in ["÷", "/"]:
        if abs(y) < 1e-15:
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
        "x": x,
        "y": y,
        "operacion": simbolo,
        "k_digitos": k,
        "exacto": exacto,
        "redondeo": {
            "valor": res_r,
            "norm_str": format_norm(res_r, k),
            "error_absoluto": ea_r,
            "error_relativo_pct": er_r,
        },
        "truncamiento": {
            "valor": res_t,
            "norm_str": format_norm_trunc(res_t, k),
            "error_absoluto": ea_t,
            "error_relativo_pct": er_t,
        },
    }


def casos_estudio_asociatividad(k: int = 3) -> dict[str, Any]:
    """
    Ejecuta los casos del Trabajo Práctico 2 (Ejercicios 2 y 3) para demostrar
    el comportamiento de la aritmética de máquina y la pérdida de asociatividad.

    Casos analizados:
    a. 133 + 0.921
    b. 133 - 0.499
    c. (121 - 119) - 0.327
    d. (121 - 0.327) - 119  (Pérdida de asociatividad respecto a c)
    e. (2/9) * (9/7)
    """
    casos_evaluados = []

    # a. 133 + 0.921
    p_a = 133.0 + 0.921
    r_133, _ = fl_round(133.0, k)
    r_0921, _ = fl_round(0.921, k)
    ps_ra, _ = fl_round(r_133 + r_0921, k)

    t_133, _ = fl_trunc(133.0, k)
    t_0921, _ = fl_trunc(0.921, k)
    ps_ta, _ = fl_trunc(t_133 + t_0921, k)

    casos_evaluados.append({
        "item": "a",
        "operacion": "133 + 0.921",
        "exacto": p_a,
        "redondeo_num": ps_ra,
        "redondeo_norm": format_norm(ps_ra, k),
        "trunc_num": ps_ta,
        "trunc_norm": format_norm_trunc(ps_ta, k),
        "paso": f"fl({r_133:g} + {r_0921:g})",
    })

    # b. 133 - 0.499
    p_b = 133.0 - 0.499
    r_0499, _ = fl_round(0.499, k)
    ps_rb, _ = fl_round(r_133 - r_0499, k)

    t_0499, _ = fl_trunc(0.499, k)
    ps_tb, _ = fl_trunc(t_133 - t_0499, k)

    casos_evaluados.append({
        "item": "b",
        "operacion": "133 - 0.499",
        "exacto": p_b,
        "redondeo_num": ps_rb,
        "redondeo_norm": format_norm(ps_rb, k),
        "trunc_num": ps_tb,
        "trunc_norm": format_norm_trunc(ps_tb, k),
        "paso": f"fl({r_133:g} - {r_0499:g})",
    })

    # c. (121 - 119) - 0.327
    p_c = (121.0 - 119.0) - 0.327
    r_121, _ = fl_round(121.0, k)
    r_119, _ = fl_round(119.0, k)
    r_0327, _ = fl_round(0.327, k)
    st1_rc, _ = fl_round(r_121 - r_119, k)
    ps_rc, _ = fl_round(st1_rc - r_0327, k)

    t_121, _ = fl_trunc(121.0, k)
    t_119, _ = fl_trunc(119.0, k)
    t_0327, _ = fl_trunc(0.327, k)
    st1_tc, _ = fl_trunc(t_121 - t_119, k)
    ps_tc, _ = fl_trunc(st1_tc - t_0327, k)

    casos_evaluados.append({
        "item": "c",
        "operacion": "(121 - 119) - 0.327",
        "exacto": p_c,
        "redondeo_num": ps_rc,
        "redondeo_norm": format_norm(ps_rc, k),
        "trunc_num": ps_tc,
        "trunc_norm": format_norm_trunc(ps_tc, k),
        "paso": f"Paso 1: fl(121-119)={st1_rc:g} → fl({st1_rc:g} - {r_0327:g})",
    })

    # d. (121 - 0.327) - 119 (absorción del término pequeño)
    p_d = (121.0 - 0.327) - 119.0
    st1_rd, _ = fl_round(r_121 - r_0327, k)
    ps_rd, _ = fl_round(st1_rd - r_119, k)

    st1_td, _ = fl_trunc(t_121 - t_0327, k)
    ps_td, _ = fl_trunc(st1_td - t_119, k)

    casos_evaluados.append({
        "item": "d",
        "operacion": "(121 - 0.327) - 119",
        "exacto": p_d,
        "redondeo_num": ps_rd,
        "redondeo_norm": format_norm(ps_rd, k),
        "trunc_num": ps_td,
        "trunc_norm": format_norm_trunc(ps_td, k),
        "paso": f"Paso 1: fl(121-0.327)={st1_rd:g} → fl({st1_rd:g} - {r_119:g})",
    })

    # e. (2/9) * (9/7)
    p_e = 2.0 / 7.0
    r_29, _ = fl_round(2.0 / 9.0, k)
    r_97, _ = fl_round(9.0 / 7.0, k)
    ps_re, _ = fl_round(r_29 * r_97, k)

    t_29, _ = fl_trunc(2.0 / 9.0, k)
    t_97, _ = fl_trunc(9.0 / 7.0, k)
    ps_te, _ = fl_trunc(t_29 * t_97, k)

    casos_evaluados.append({
        "item": "e",
        "operacion": "(2/9) × (9/7)",
        "exacto": p_e,
        "redondeo_num": ps_re,
        "redondeo_norm": format_norm(ps_re, k),
        "trunc_num": ps_te,
        "trunc_norm": format_norm_trunc(ps_te, k),
        "paso": f"fl({r_29:g} × {r_97:g})",
    })

    filas_tabla = []
    for c in casos_evaluados:
        ea_r = abs(c["exacto"] - c["redondeo_num"])
        er_r = (ea_r / abs(c["exacto"]) * 100.0) if abs(c["exacto"]) > 1e-15 else 0.0

        ea_t = abs(c["exacto"] - c["trunc_num"])
        er_t = (ea_t / abs(c["exacto"]) * 100.0) if abs(c["exacto"]) > 1e-15 else 0.0

        filas_tabla.append({
            "item": c["item"],
            "operacion": c["operacion"],
            "exacto": c["exacto"],
            "redondeo_norm": c["redondeo_norm"],
            "redondeo_val": c["redondeo_num"],
            "error_rel_redondeo_pct": er_r,
            "trunc_norm": c["trunc_norm"],
            "trunc_val": c["trunc_num"],
            "error_rel_trunc_pct": er_t,
            "detalle_paso": c["paso"],
        })

    # Análisis de asociatividad entre c y d
    caso_c_r = casos_evaluados[2]["redondeo_num"]
    caso_d_r = casos_evaluados[3]["redondeo_num"]
    err_c_pct = abs(p_c - caso_c_r) / abs(p_c) * 100.0
    err_d_pct = abs(p_d - caso_d_r) / abs(p_d) * 100.0

    analisis_asociatividad = {
        "p_teorico": p_c,
        "resultado_agrupado_c": caso_c_r,
        "error_pct_c": err_c_pct,
        "resultado_agrupado_d": caso_d_r,
        "error_pct_d": err_d_pct,
        "conclusion": (
            f"Al calcular (121 - 119) - 0.327 primero se restan los números cercanos generando un número de orden 1, "
            f"obteniendo {caso_c_r:g} (Error = {err_c_pct:.2f}%). "
            f"En cambio, en (121 - 0.327) - 119, al restar 0.327 de 121 en k={k} dígitos, el 0.327 sufre absorción "
            f"arrojando {caso_d_r:g} (Error = {err_d_pct:.2f}%). Se comprueba que en máquina (A - B) - C != (A - C) - B."
        ),
    }

    return {
        "k_digitos": k,
        "tabla": filas_tabla,
        "analisis_asociatividad": analisis_asociatividad,
    }


def analizar_cancelacion_cuadratica(
    a: float = 1.0,
    b: float = 62.10,
    c: float = 1.0,
    k: int = 4,
) -> dict[str, Any]:
    """
    Analiza el fenómeno de Cancelación Catastrófica en la ecuación cuadrática ax^2 + bx + c = 0
    cuando b^2 >> 4ac, comparando la Fórmula Tradicional vs. la Fórmula Racionalizada.

    Caso canónico del práctico (ejer_4.py):
        x^2 + 62.10x + 1 = 0 con k = 4 dígitos.
    """
    if abs(a) < 1e-15:
        raise ValueError("El coeficiente 'a' no puede ser cero.")

    disc_real = b**2 - 4.0 * a * c
    if disc_real <= 0:
        raise ValueError("Se requiere discriminante estrictamente positivo (b^2 - 4ac > 0) para raíces reales.")

    # Raíces de referencia en alta precisión (usando fórmula estable para cada una)
    if b >= 0:
        x2_exact = (-b - math.sqrt(disc_real)) / (2.0 * a)
        x1_exact = (-2.0 * c) / (b + math.sqrt(disc_real))
    else:
        x1_exact = (-b + math.sqrt(disc_real)) / (2.0 * a)
        x2_exact = (-2.0 * c) / (b - math.sqrt(disc_real))

    # Cómputo paso a paso en aritmética de k dígitos
    fl_a, _ = fl_round(a, k)
    fl_b, _ = fl_round(b, k)
    fl_c, _ = fl_round(c, k)

    fl_b_sq, _ = fl_round(fl_b**2, k)
    fl_4ac, _ = fl_round(4.0 * fl_a * fl_c, k)
    fl_disc, _ = fl_round(fl_b_sq - fl_4ac, k)
    fl_sqrt_disc, _ = fl_round(math.sqrt(fl_disc), k)
    fl_2a, _ = fl_round(2.0 * fl_a, k)
    fl_neg_2c, _ = fl_round(-2.0 * fl_c, k)

    # Fórmula Tradicional
    num1_trad, _ = fl_round(-fl_b + fl_sqrt_disc, k)
    x1_trad, _ = fl_round(num1_trad / fl_2a, k)

    num2_trad, _ = fl_round(-fl_b - fl_sqrt_disc, k)
    x2_trad, _ = fl_round(num2_trad / fl_2a, k)

    # Fórmula Racionalizada
    den1_rat, _ = fl_round(fl_b + fl_sqrt_disc, k)
    x1_rat = fl_round(fl_neg_2c / den1_rat, k)[0] if den1_rat != 0 else float("nan")

    den2_rat, _ = fl_round(fl_b - fl_sqrt_disc, k)
    x2_rat = fl_round(fl_neg_2c / den2_rat, k)[0] if den2_rat != 0 else float("nan")

    pasos_maquina = {
        "fl_a": fl_a,
        "fl_b": fl_b,
        "fl_c": fl_c,
        "fl_b_sq": fl_b_sq,
        "fl_4ac": fl_4ac,
        "fl_disc": fl_disc,
        "fl_sqrt_disc": fl_sqrt_disc,
        "fl_2a": fl_2a,
        "fl_neg_2c": fl_neg_2c,
        "num1_trad": num1_trad,
        "den1_rat": den1_rat,
        "num2_trad": num2_trad,
        "den2_rat": den2_rat,
    }

    metodos = [
        ("x1_tradicional", "x₁ Tradicional: (-b + √Δ)/(2a)", x1_exact, x1_trad, True, "Inestable: resta -b + √Δ en numerador produciendo cancelación de cifras."),
        ("x1_racionalizada", "x₁ Racionalizada: -2c/(b + √Δ)", x1_exact, x1_rat, False, "Estable: suma b + √Δ en denominador preservando las cifras significativas."),
        ("x2_tradicional", "x₂ Tradicional: (-b - √Δ)/(2a)", x2_exact, x2_trad, False, "Estable: suma -b - √Δ con el mismo signo, sin cancelación."),
        ("x2_racionalizada", "x₂ Racionalizada: -2c/(b - √Δ)", x2_exact, x2_rat, True, "Inestable: resta b - √Δ en denominador produciendo cancelación de cifras."),
    ]

    tabla_comparativa = []
    for clave, nombre, ex, ap, cancelacion, diag in metodos:
        ea = abs(ex - ap)
        er_pct = (ea / abs(ex) * 100.0) if abs(ex) > 1e-15 else 0.0
        tabla_comparativa.append({
            "clave": clave,
            "metodo": nombre,
            "exacto": ex,
            "aprox_num": ap,
            "aprox_norm": format_norm(ap, k),
            "error_absoluto": ea,
            "error_absoluto_norm": format_norm(ea, k),
            "error_relativo_pct": er_pct,
            "sufre_cancelacion": cancelacion,
            "diagnostico": diag,
        })

    return {
        "coeficientes": {"a": a, "b": b, "c": c},
        "k_digitos": k,
        "pasos_maquina": pasos_maquina,
        "tabla_comparativa": tabla_comparativa,
        "x1_exacto": x1_exact,
        "x2_exacto": x2_exact,
    }


def evaluar_polinomio_horner(
    x_val: float = 4.71,
    k: int = 3,
) -> dict[str, Any]:
    """
    Evalúa el polinomio canónico del Trabajo Práctico 2 (Ejercicio 5):
        P(x) = x^3 - 6x^2 + 3x - 0.149
    en el punto x_val (4.71 por defecto) a k dígitos significativos, contrastando
    la Forma Estándar Directa frente a la Forma Anidada de Horner (con truncamiento y redondeo).
    """
    # 1. Valores Exactos en Doble Precisión
    x2_ex = x_val**2
    x3_ex = x_val**3
    t_6x2_ex = 6.0 * x2_ex
    t_3x_ex = 3.0 * x_val
    f_ex = x3_ex - t_6x2_ex + t_3x_ex - 0.149

    # 2. Forma Estándar Truncada
    x2_tr = fl_trunc(x_val**2, k)[0]
    x3_tr = fl_trunc(x2_tr * x_val, k)[0]
    t_6x2_tr = fl_trunc(6.0 * x2_tr, k)[0]
    t_3x_tr = fl_trunc(3.0 * x_val, k)[0]
    st1_tr = fl_trunc(x3_tr - t_6x2_tr + t_3x_tr, k)[0]
    f_trad_tr = fl_trunc(st1_tr - 0.149, k)[0]

    # 3. Forma Estándar Redondeada
    x2_rd = fl_round(x_val**2, k)[0]
    x3_rd = fl_round(x2_rd * x_val, k)[0]
    t_6x2_rd = fl_round(6.0 * x2_rd, k)[0]
    t_3x_rd = fl_round(3.0 * x_val, k)[0]
    st1_rd = fl_round(x3_rd - t_6x2_rd + t_3x_rd, k)[0]
    f_trad_rd = fl_round(st1_rd - 0.149, k)[0]

    # 4. Forma Anidada de Horner: ((x - 6)x + 3)x - 0.149
    # Truncamiento
    p1_tr = fl_trunc(x_val - 6.0, k)[0]
    p2_tr = fl_trunc(p1_tr * x_val, k)[0]
    p3_tr = fl_trunc(p2_tr + 3.0, k)[0]
    p4_tr = fl_trunc(p3_tr * x_val, k)[0]
    f_anid_tr = fl_trunc(p4_tr - 0.149, k)[0]

    # Redondeo
    p1_rd = fl_round(x_val - 6.0, k)[0]
    p2_rd = fl_round(p1_rd * x_val, k)[0]
    p3_rd = fl_round(p2_rd + 3.0, k)[0]
    p4_rd = fl_round(p3_rd * x_val, k)[0]
    f_anid_rd = fl_round(p4_rd - 0.149, k)[0]

    tabla_terminos = [
        {"aritmetica": "Exacto (Doble Precisión)", "x": x_val, "x2": x2_ex, "x3": x3_ex, "6x2": t_6x2_ex, "3x": t_3x_ex, "f_x": f_ex},
        {"aritmetica": f"{k} dígitos (Truncamiento)", "x": x_val, "x2": x2_tr, "x3": x3_tr, "6x2": t_6x2_tr, "3x": t_3x_tr, "f_x": f_trad_tr},
        {"aritmetica": f"{k} dígitos (Redondeo)", "x": x_val, "x2": x2_rd, "x3": x3_rd, "6x2": t_6x2_rd, "3x": t_3x_rd, "f_x": f_trad_rd},
    ]

    tabla_horner_pasos = [
        {"aritmetica": f"{k} dígitos (Truncamiento)", "p1": p1_tr, "p2": p2_tr, "p3": p3_tr, "p4": p4_tr, "f_x": f_anid_tr},
        {"aritmetica": f"{k} dígitos (Redondeo)", "p1": p1_rd, "p2": p2_rd, "p3": p3_rd, "p4": p4_rd, "f_x": f_anid_rd},
    ]

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
            "error_absoluto": ea,
            "error_relativo_pct": er_pct,
        })

    return {
        "x": x_val,
        "k_digitos": k,
        "f_exacto": f_ex,
        "tabla_terminos": tabla_terminos,
        "tabla_horner_pasos": tabla_horner_pasos,
        "tabla_errores": tabla_errores,
    }

