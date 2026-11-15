"""Cuadratura numérica adaptativa para paso variable Δh — Arandú."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

_SUBINDICES = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _subindice(numero: int) -> str:
    return str(int(numero)).translate(_SUBINDICES)


def _curva_polinomica(seg_x: Sequence[float], seg_y: Sequence[float], grado: int, n_puntos: int) -> tuple[list[float], list[float]]:
    if grado <= 1 or len(seg_x) <= 2:
        return [float(x) for x in seg_x], [float(y) for y in seg_y]
    coeficientes = np.polyfit(np.asarray(seg_x, dtype=float), np.asarray(seg_y, dtype=float), deg=grado)
    xs_eval = np.linspace(float(seg_x[0]), float(seg_x[-1]), n_puntos)
    ys_eval = np.polyval(coeficientes, xs_eval)
    return [float(x) for x in xs_eval], [float(y) for y in ys_eval]


def _integral_referencia_simpson(f: Callable[[float], float], a: float, b: float, subintervalos: int = 400) -> float:
    n = subintervalos if subintervalos % 2 == 0 else subintervalos + 1
    xs = np.linspace(a, b, n + 1)
    ys = np.asarray([float(f(float(x))) for x in xs], dtype=float)
    h = (b - a) / float(n)
    return float((h / 3.0) * (ys[0] + ys[-1] + 4.0 * np.sum(ys[1:-1:2]) + 2.0 * np.sum(ys[2:-2:2])))


def integrar_paso_variable(
    f: Callable[[float], float],
    x0: float,
    pasos_h: Sequence[float],
    integral_exacta_fn: Callable[[float, float], float] | None = None,
    tolerancia_paso: float = 1e-6,
) -> dict:
    """Integra un tramo con pasos variables aplicando Simpson 3/8, Simpson 1/3 o Trapecio."""
    pasos = [float(h) for h in pasos_h]
    if not pasos:
        raise ValueError("Se requiere al menos un subintervalo en pasos_h.")
    if any(h <= 0.0 for h in pasos):
        raise ValueError("Todas las longitudes de paso Δh deben ser positivas.")

    x0_val = float(x0)
    nodos_x: list[float] = [x0_val]
    for h in pasos:
        nodos_x.append(nodos_x[-1] + h)
    nodos_y: list[float] = [float(f(x)) for x in nodos_x]

    n = len(pasos)
    tramos: list[dict] = []
    i = 0
    tramo_idx = 1

    while i < n:
        m = 1
        while i + m < n and abs(pasos[i + m] - pasos[i]) <= tolerancia_paso:
            m += 1

        h = pasos[i]
        if m >= 3 and m % 2 == 1:
            span = 3
            area = (3.0 * h / 8.0) * (
                nodos_y[i] + 3.0 * nodos_y[i + 1] + 3.0 * nodos_y[i + 2] + nodos_y[i + 3]
            )
            metodo = "Simpson 3/8"
            motivo = (
                f"Δh{_subindice(i + 1)} = Δh{_subindice(i + 2)} = Δh{_subindice(i + 3)} = {h:.2f}: "
                f"bloque impar de 3 pasos iguales activa cúbica (Simpson 3/8)"
            )
            seg_x = nodos_x[i : i + 4]
            seg_y = nodos_y[i : i + 4]
            x_curva, y_curva = _curva_polinomica(seg_x, seg_y, grado=3, n_puntos=40)
        elif m >= 2:
            span = 2
            area = (h / 3.0) * (nodos_y[i] + 4.0 * nodos_y[i + 1] + nodos_y[i + 2])
            metodo = "Simpson 1/3"
            motivo = (
                f"Δh{_subindice(i + 1)} = Δh{_subindice(i + 2)} = {h:.2f}: "
                f"pasos iguales permiten ajustar parábola (Simpson 1/3)"
            )
            seg_x = nodos_x[i : i + 3]
            seg_y = nodos_y[i : i + 3]
            x_curva, y_curva = _curva_polinomica(seg_x, seg_y, grado=2, n_puntos=32)
        else:
            span = 1
            area = (h / 2.0) * (nodos_y[i] + nodos_y[i + 1])
            metodo = "Trapecio"
            if i + 1 < n and abs(pasos[i] - pasos[i + 1]) > tolerancia_paso:
                motivo = (
                    f"Δh{_subindice(i + 1)} = {h:.2f} ≠ Δh{_subindice(i + 2)} = {pasos[i + 1]:.2f}: "
                    f"cambio de paso obliga a cerrar con Trapecio"
                )
            else:
                motivo = (
                    f"Δh{_subindice(i + 1)} = {h:.2f} sin par contiguo de igual paso: "
                    f"se cierra con Trapecio"
                )
            seg_x = nodos_x[i : i + 2]
            seg_y = nodos_y[i : i + 2]
            x_curva, y_curva = _curva_polinomica(seg_x, seg_y, grado=1, n_puntos=2)

        tramos.append(
            {
                "tramo_id": tramo_idx,
                "inicio": float(seg_x[0]),
                "fin": float(seg_x[-1]),
                "nodos": f"x{_subindice(i)}..x{_subindice(i + span)}",
                "nodos_x": [float(x) for x in seg_x],
                "nodos_y": [float(y) for y in seg_y],
                "paso_h": float(h),
                "metodo": metodo,
                "motivo": motivo,
                "area": float(area),
                "x_curva": x_curva,
                "y_curva": y_curva,
            }
        )
        i += span
        tramo_idx += 1

    xf_val = nodos_x[-1]
    integral_aprox = float(sum(t["area"] for t in tramos))
    if integral_exacta_fn is not None:
        integral_exacta = float(integral_exacta_fn(x0_val, xf_val))
    else:
        integral_exacta = _integral_referencia_simpson(f, x0_val, xf_val)

    error_absoluto = abs(integral_exacta - integral_aprox)
    error_relativo_pct = (
        (error_absoluto / abs(integral_exacta) * 100.0) if abs(integral_exacta) > 1e-14 else 0.0
    )

    return {
        "integral_aprox": integral_aprox,
        "integral_exacta": integral_exacta,
        "error_absoluto": float(error_absoluto),
        "error_relativo_pct": float(error_relativo_pct),
        "x0": x0_val,
        "xf": float(xf_val),
        "pasos_h": pasos,
        "nodos_x": nodos_x,
        "nodos_y": nodos_y,
        "tramos": tramos,
    }

