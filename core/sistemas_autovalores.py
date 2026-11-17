"""
core/sistemas_autovalores.py
============================
Módulo de cálculo puro para Sistemas Lineales y Autovalores/Autovectores
(Arandú: Métodos Computacionales v3.0).

Incluye:
- Eliminación de Gauss (con o sin pivoteo parcial) y sustitución hacia atrás.
- Método de Gauss-Jordan (resolución e inversa [A|I] -> [I|A⁻¹]).
- Factorización LU (Algoritmo de Doolittle).
- Método Iterativo de Gauss-Seidel (con dominancia diagonal y tasa de convergencia).
- Método de las Potencias (Autovalor máximo directo e inverso para mínimo).
- Método de Faddeev-Leverrier (polinomio característico, trazas, determinante,
  inversa exacta con fórmula A⁻¹ = (1/b_n)*(B_{n-1} - b_{n-1}I) y espectro completo).

Diseño: Funciones 100% puras, sin `input()` ni `print()`.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np


def eliminacion_gauss(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray,
    pivoteo: bool = True,
    decimales: int = 4,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    b = np.array(b_vector, dtype=float).flatten()
    n = len(A)
    if n < 1 or A.shape != (n, n) or len(b) != n:
        raise ValueError(f"Dimensiones incompatibles: A es {A.shape} y b tiene longitud {len(b)}.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError("Todos los coeficientes de A y b deben ser valores reales finitos.")

    M = np.hstack([A, b.reshape(-1, 1)])
    pasos: list[dict[str, Any]] = [{
        "paso": 0,
        "descripcion": "Sistema aumentado inicial [A | b]",
        "matriz": np.round(M.copy(), decimales).tolist(),
        "operaciones": [],
    }]
    intercambios = 0

    for k in range(n - 1):
        operaciones = []
        if pivoteo:
            fila_max = k + int(np.argmax(np.abs(M[k:, k])))
            if fila_max != k:
                M[[k, fila_max]] = M[[fila_max, k]]
                intercambios += 1
                operaciones.append(f"Intercambio de filas: F_{k+1} ↔ F_{fila_max+1}")

        pivote = M[k, k]
        if abs(pivote) < 1e-14:
            raise ValueError(f"Pivote nulo en ({k+1}, {k+1}). La matriz es singular o requiere pivoteo.")

        for i in range(k + 1, n):
            factor = M[i, k] / pivote
            if abs(factor) > 1e-15:
                operaciones.append(f"F_{i+1} = F_{i+1} - ({factor:.{decimales}f}) · F_{k+1}")
                M[i, k:] -= factor * M[k, k:]
                M[i, k] = 0.0

        pasos.append({
            "paso": k + 1,
            "descripcion": f"Eliminación en columna {k+1}",
            "matriz": np.round(M.copy(), decimales).tolist(),
            "operaciones": operaciones,
        })

    if abs(M[n - 1, n - 1]) < 1e-14:
        raise ValueError("El sistema no tiene solución única (matriz singular).")

    x = np.zeros(n, dtype=float)
    pasos_sustitucion = []
    for i in range(n - 1, -1, -1):
        suma_conocidos = sum(M[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i, n] - suma_conocidos) / M[i, i]
        pasos_sustitucion.append({
            "variable": f"x_{i+1}",
            "ecuacion": f"({M[i, n]:.{decimales}f} - {suma_conocidos:.{decimales}f}) / {M[i, i]:.{decimales}f}",
            "valor": float(x[i]),
        })
    pasos_sustitucion.reverse()
    if not np.all(np.isfinite(x)):
        raise ValueError("La sustitución regresiva produjo valores no finitos.")

    Ax = np.dot(A, x)
    residual = Ax - b
    return {
        "solucion": x.tolist(),
        "solucion_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
        "intercambios": intercambios,
        "pasos_escalonamiento": pasos,
        "pasos_sustitucion": pasos_sustitucion,
        "residual": residual.tolist(),
        "norma_error": float(np.linalg.norm(residual)),
        "duracion": time.perf_counter() - t_ini,
    }


def gauss_jordan(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray | None = None,
    pivoteo: bool = True,
    decimales: int = 4,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    if A.shape != (n, n):
        raise ValueError("La matriz A debe ser cuadrada.")

    if b_vector is not None:
        b = np.array(b_vector, dtype=float).flatten()
        if len(b) != n:
            raise ValueError("Longitud incompatible de b.")
        M = np.hstack([A, b.reshape(-1, 1)])
        modo = "sistema"
    else:
        M = np.hstack([A, np.eye(n)])
        modo = "inversa"

    pasos: list[dict[str, Any]] = [{
        "paso": 0,
        "descripcion": "Matriz ampliada inicial",
        "matriz": np.round(M.copy(), decimales).tolist(),
        "operaciones": [],
    }]

    for i in range(n):
        operaciones = []
        if pivoteo:
            fila_max = i + int(np.argmax(np.abs(M[i:, i])))
            if fila_max != i:
                M[[i, fila_max]] = M[[fila_max, i]]
                operaciones.append(f"Intercambio: F_{i+1} ↔ F_{fila_max+1}")

        pivote = M[i, i]
        if abs(pivote) < 1e-14:
            raise ValueError(f"Pivote singular o nulo en ({i+1}, {i+1}).")

        if abs(pivote - 1.0) > 1e-15:
            M[i, :] = M[i, :] / pivote
            operaciones.append(f"F_{i+1} = F_{i+1} / ({pivote:.{decimales}f})")

        for k in range(n):
            if k != i:
                factor = M[k, i]
                if abs(factor) > 1e-15:
                    M[k, :] -= factor * M[i, :]
                    M[k, i] = 0.0
                    operaciones.append(f"F_{k+1} = F_{k+1} - ({factor:.{decimales}f}) · F_{i+1}")

        pasos.append({
            "paso": i + 1,
            "descripcion": f"Reducción completa de la columna {i+1}",
            "matriz": np.round(M.copy(), decimales).tolist(),
            "operaciones": operaciones,
        })

    duracion = time.perf_counter() - t_ini
    if modo == "sistema":
        sol = M[:, -1].tolist()
        residual = np.dot(A, sol) - b
        return {
            "modo": "sistema",
            "solucion": sol,
            "solucion_dict": {f"x_{i+1}": s for i, s in enumerate(sol)},
            "pasos": pasos,
            "residual": residual.tolist(),
            "norma_error": float(np.linalg.norm(residual)),
            "duracion": duracion,
        }
    A_inv = M[:, n:].copy()
    return {
        "modo": "inversa",
        "inversa": A_inv.tolist(),
        "pasos": pasos,
        "norma_error": float(np.linalg.norm(np.dot(A, A_inv) - np.eye(n))),
        "duracion": duracion,
    }


def factorizacion_lu(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray | None = None,
    decimales: int = 4,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    if A.shape != (n, n):
        raise ValueError(f"La matriz A debe ser cuadrada. Recibido: {A.shape}")

    L = np.eye(n, dtype=float)
    U = A.copy()
    pasos_eliminacion = []

    for k in range(n - 1):
        pivote = U[k, k]
        if abs(pivote) < 1e-14:
            raise ValueError(f"Pivote nulo en U[{k+1},{k+1}]. Doolittle directo sin permutación no puede continuar.")
        factores_paso = []
        for i in range(k + 1, n):
            factor = U[i, k] / pivote
            L[i, k] = factor
            factores_paso.append({
                "fila_i": i + 1, "fila_k": k + 1, "factor": factor,
                "formula": f"L_{i+1},{k+1} = {U[i, k]:.{decimales}f} / {pivote:.{decimales}f} = {factor:.{decimales}f}",
            })
            U[i, k:] -= factor * U[k, k:]
            U[i, k] = 0.0
        pasos_eliminacion.append({
            "columna": k + 1,
            "factores": factores_paso,
            "U_actual": np.round(U.copy(), decimales).tolist(),
            "L_actual": np.round(L.copy(), decimales).tolist(),
        })

    if abs(U[n - 1, n - 1]) < 1e-14:
        raise ValueError("La matriz es singular (U[n, n] ≈ 0).")

    error_LU = float(np.linalg.norm(np.dot(L, U) - A))
    res: dict[str, Any] = {
        "L": L.tolist(), "U": U.tolist(), "error_LU": error_LU,
        "pasos_eliminacion": pasos_eliminacion,
    }

    if b_vector is not None:
        b = np.array(b_vector, dtype=float).flatten()
        if len(b) != n:
            raise ValueError("Longitud incompatible de b.")
        y = np.zeros(n, dtype=float)
        pasos_y = []
        for i in range(n):
            suma = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (b[i] - suma) / L[i, i]
            pasos_y.append({"variable": f"y_{i+1}", "ecuacion": f"({b[i]:.{decimales}f} - {suma:.{decimales}f}) / {L[i, i]:.{decimales}f}", "valor": float(y[i])})

        x = np.zeros(n, dtype=float)
        pasos_x = []
        for i in range(n - 1, -1, -1):
            suma = sum(U[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - suma) / U[i, i]
            pasos_x.append({"variable": f"x_{i+1}", "ecuacion": f"({y[i]:.{decimales}f} - {suma:.{decimales}f}) / {U[i, i]:.{decimales}f}", "valor": float(x[i])})
        pasos_x.reverse()

        residual = np.dot(A, x) - b
        res.update({
            "y": y.tolist(), "x": x.tolist(), "solucion": x.tolist(),
            "solucion_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
            "pasos_y": pasos_y, "pasos_x": pasos_x,
            "residual": residual.tolist(),
            "error_Ax": float(np.linalg.norm(residual)),
            "norma_error": float(np.linalg.norm(residual)),
        })

    res["duracion"] = time.perf_counter() - t_ini
    return res


def verificar_dominancia_diagonal(A_matriz: list[list[float]] | np.ndarray) -> dict[str, Any]:
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    es_dominante = True
    filas_info = []
    for i in range(n):
        diag = abs(A[i, i])
        suma_otros = sum(abs(A[i, j]) for j in range(n) if j != i)
        cumple = diag > suma_otros
        if not cumple:
            es_dominante = False
        filas_info.append({
            "fila": i + 1, "diagonal": diag, "suma_otros": suma_otros,
            "cumple": cumple, "diferencia": diag - suma_otros,
        })
    return {
        "es_dominante": es_dominante,
        "filas": filas_info,
        "diagnostico": (
            "La matriz ES estrictamente diagonal dominante (convergencia garantizada)."
            if es_dominante else
            "La matriz NO es estrictamente diagonal dominante. Gauss-Seidel podría oscilar o diverger."
        ),
    }


def gauss_seidel(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray,
    x0: list[float] | np.ndarray | None = None,
    tol: float = 0.01,
    max_iter: int = 100,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    b = np.array(b_vector, dtype=float).flatten()
    n = len(b)
    if A.shape != (n, n):
        raise ValueError("La matriz A debe ser cuadrada y coincidir con b.")

    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(f"Elemento diagonal nulo en a[{i+1},{i+1}]. Reordene las ecuaciones.")

    dominancia = verificar_dominancia_diagonal(A)
    x = np.ones(n, dtype=float) if x0 is None else np.array(x0, dtype=float).flatten()
    historial: list[dict[str, Any]] = []
    convergente = False

    for k in range(1, max_iter + 1):
        x_anterior = x.copy()
        for i in range(n):
            suma = b[i]
            for j in range(n):
                if j != i:
                    suma -= A[i, j] * x[j]
            x[i] = suma / A[i, i]

        if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.max(np.abs(x)) > 1e12:
            historial.append({
                "iteracion": k, "x": x_anterior.tolist(),
                "x_dict": {f"x_{i+1}": float(x_anterior[i]) for i in range(n)},
                "error_max": float("inf"), "norma_residuo": float("inf"), "cumple_tol": False,
            })
            break

        errores = np.abs(x - x_anterior)
        error_max = float(np.max(errores))
        residuo_k = float(np.linalg.norm(np.dot(A, x) - b))
        cumple = error_max < tol

        historial.append({
            "iteracion": k,
            "x": x.copy().tolist(),
            "x_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
            "errores_componentes": errores.tolist(),
            "error_max": error_max,
            "norma_residuo": residuo_k,
            "cumple_tol": cumple,
        })
        if cumple:
            convergente = True
            break

    residual = np.dot(A, x) - b
    return {
        "solucion": x.tolist(),
        "solucion_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
        "iteraciones": len(historial),
        "error_final": historial[-1]["error_max"] if historial else 0.0,
        "convergente": convergente,
        "dominancia": dominancia,
        "residual": residual.tolist(),
        "norma_residuo": float(np.linalg.norm(residual)),
        "norma_error": float(np.linalg.norm(residual)),
        "duracion": time.perf_counter() - t_ini,
        "historial": historial,
    }


def metodo_potencias_max(
    A_matriz: list[list[float]] | np.ndarray,
    x0: list[float] | np.ndarray | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    if A.shape != (n, n):
        raise ValueError("La matriz debe ser cuadrada.")

    x = np.ones(n, dtype=float) if x0 is None else np.array(x0, dtype=float).flatten()
    if np.all(np.abs(x) < 1e-15):
        raise ValueError("El vector inicial x0 no puede ser nulo.")

    lambda_old = 0.0
    lambda_new = 0.0
    historial: list[dict[str, Any]] = []
    convergente = False

    for k in range(1, max_iter + 1):
        y = np.dot(A, x)
        idx_max = int(np.argmax(np.abs(y)))
        lambda_new = float(y[idx_max])

        if abs(lambda_new) < 1e-15:
            historial.append({
                "iteracion": k, "y": y.tolist(), "lambda": 0.0,
                "x_norm": x.tolist(), "error_pct": 0.0, "cumple_tol": True,
            })
            convergente = True
            break

        x_norm = y / lambda_new
        err_pct = abs(lambda_new - lambda_old) / abs(lambda_new) * 100.0 if k > 1 else None
        cumple = err_pct is not None and err_pct < tol

        historial.append({
            "iteracion": k, "y": y.tolist(), "lambda": lambda_new,
            "x_norm": x_norm.tolist(), "error_pct": err_pct, "cumple_tol": cumple,
        })
        if cumple:
            convergente = True
            x = x_norm
            break
        lambda_old = lambda_new
        x = x_norm

    error_autovector = float(np.linalg.norm(np.dot(A, x) - lambda_new * x))
    return {
        "autovalor_max": lambda_new,
        "autovector": x.tolist(),
        "iteraciones": len(historial),
        "error_final_pct": historial[-1]["error_pct"] if historial and historial[-1]["error_pct"] is not None else 0.0,
        "convergente": convergente,
        "error_autovector": error_autovector,
        "duracion": time.perf_counter() - t_ini,
        "historial": historial,
    }


def metodo_potencias_min(
    A_matriz: list[list[float]] | np.ndarray,
    x0: list[float] | np.ndarray | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> dict[str, Any]:
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("La matriz es singular y no posee inversa (λ_min = 0).")

    res_inv = metodo_potencias_max(A_inv, x0=x0, tol=tol, max_iter=max_iter)
    lambda_inv = res_inv["autovalor_max"]
    if abs(lambda_inv) < 1e-15:
        raise ValueError("El autovalor dominante de A⁻¹ es nulo.")

    lambda_min = 1.0 / lambda_inv
    hist_min = []
    for row in res_inv["historial"]:
        l_inv = row["lambda"]
        l_aprox = (1.0 / l_inv) if abs(l_inv) > 1e-15 else float("inf")
        hist_min.append({
            "iteracion": row["iteracion"],
            "lambda_inversa": l_inv,
            "lambda": l_aprox,
            "x_norm": row["x_norm"],
            "error_pct": row["error_pct"],
            "cumple_tol": row["cumple_tol"],
        })

    x_vec = np.array(res_inv["autovector"], dtype=float)
    error_autovector = float(np.linalg.norm(np.dot(A, x_vec) - lambda_min * x_vec))
    return {
        "autovalor_min": lambda_min,
        "autovalor_inversa": lambda_inv,
        "autovector": res_inv["autovector"],
        "A_inversa": A_inv.tolist(),
        "iteraciones": res_inv["iteraciones"],
        "error_final_pct": res_inv["error_final_pct"],
        "error_autovector": error_autovector,
        "convergente": res_inv["convergente"],
        "duracion": time.perf_counter() - t_ini,
        "historial": hist_min,
    }


def metodo_faddeev_leverrier(A_matriz: list[list[float]] | np.ndarray) -> dict[str, Any]:
    """
    Calcula polinomio característico, trazas b_k, matrices B_k, determinante,
    inversa analítica A⁻¹ = (1/b_n)(B_{n-1} - b_{n-1}I) y autovalores/autovectores.
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    if A.shape != (n, n) or n < 1:
        raise ValueError("La matriz debe ser cuadrada de orden >= 1.")

    I = np.eye(n, dtype=float)
    B_prev = A.copy()
    b_coefs: list[float] = []
    B_matrices: list[list[list[float]]] = [A.copy().tolist()]
    historial_pasos = []

    b1 = float(np.trace(B_prev))
    b_coefs.append(b1)
    historial_pasos.append({"k": 1, "formula": "B₁ = A", "B_k": B_prev.tolist(), "traza": b1, "b_k": b1})

    for k in range(2, n + 1):
        Bk = np.dot(A, B_prev - b_coefs[-1] * I)
        traza = float(np.trace(Bk))
        bk = traza / k
        b_coefs.append(bk)
        B_matrices.append(Bk.tolist())
        historial_pasos.append({
            "k": k, "formula": f"B_{k} = A · (B_{k-1} - b_{k-1}·I)",
            "B_k": Bk.tolist(), "traza": traza, "b_k": bk,
        })
        B_prev = Bk

    # Polinomio monico: P_m(λ) = λ^n - b_1 λ^{n-1} - ... - b_n
    coef_poly = [1.0] + [-bk for bk in b_coefs]
    # Determinante: det(A) = (-1)^n * coef_poly[-1] = (-1)^(n-1) * b_n
    det_A = float(((-1) ** n) * coef_poly[-1])
    b_n = b_coefs[-1]
    norma_cayley_hamilton = float(np.linalg.norm(np.array(B_matrices[-1], dtype=float) - b_n * I))

    # Regla de los Signos de Descartes sobre P(+λ) y P(-λ)
    raices_nulas = 0
    for c_tail in reversed(coef_poly):
        if abs(c_tail) < 1e-11:
            raices_nulas += 1
        else:
            break
    signos_pos = [1 if c > 0 else -1 for c in coef_poly if abs(c) >= 1e-11]
    var_pos = sum(1 for a_s, b_s in zip(signos_pos, signos_pos[1:]) if a_s != b_s)
    signos_neg = [
        1 if (c * ((-1) ** (n - idx))) > 0 else -1
        for idx, c in enumerate(coef_poly)
        if abs(c) >= 1e-11
    ]
    var_neg = sum(1 for a_s, b_s in zip(signos_neg, signos_neg[1:]) if a_s != b_s)
    posibles_pos = list(range(var_pos, -1, -2))
    posibles_neg = list(range(var_neg, -1, -2))
    descartes_polinomio = {
        "raices_nulas": raices_nulas,
        "cambios_signo_pos": var_pos,
        "cambios_signo_neg": var_neg,
        "posibles_positivas": posibles_pos,
        "posibles_negativas": posibles_neg,
        "resumen": (
            f"P(+λ): {var_pos} cambios de signo (reales positivas posibles: {posibles_pos}); "
            f"P(-λ): {var_neg} cambios de signo (reales negativas posibles: {posibles_neg})"
            + (f"; {raices_nulas} raíz/raíces λ = 0" if raices_nulas > 0 else "")
            + "."
        ),
    }

    if abs(b_n) > 1e-12:
        if n == 1:
            A_inv = np.array([[1.0 / b_n]])
        else:
            b_n_minus_1 = b_coefs[-2]
            B_n_minus_1 = np.array(B_matrices[-2], dtype=float)
            A_inv = (1.0 / b_n) * (B_n_minus_1 - b_n_minus_1 * I)
        error_inversa = float(np.linalg.norm(np.dot(A, A_inv) - I))
        A_inv_list = A_inv.tolist()
    else:
        A_inv_list = None
        error_inversa = None

    autovalores_crudos = np.roots(coef_poly)
    autovalores = sorted(autovalores_crudos, key=lambda val: abs(val), reverse=True)

    autovectores = []
    for eig in autovalores:
        try:
            A_comp = A.astype(complex) - eig * np.eye(n, dtype=complex)
            _, _, vh = np.linalg.svd(A_comp)
            vec = vh[-1, :].conj()
            norm_v = np.linalg.norm(vec)
            if norm_v > 1e-14:
                vec = vec / norm_v
            if np.all(np.abs(vec.imag) < 1e-10):
                vec_salida: list[Any] = vec.real.tolist()
            else:
                vec_salida = [complex(round(c.real, 6), round(c.imag, 6)) for c in vec]
            autovectores.append(vec_salida)
        except Exception:
            autovectores.append(None)

    # Polinomio formateado
    signo_ext = (-1) ** n
    terminos_str = []
    for idx, c in enumerate(coef_poly):
        pot = n - idx
        c_adj = signo_ext * c
        if abs(c_adj) < 1e-12:
            continue
        if pot == n:
            terminos_str.append(f"{'-' if c_adj < 0 else ''}λ^{pot}")
        elif pot > 1:
            terminos_str.append(f"{'+' if c_adj >= 0 else '-'} {abs(c_adj):.4g}λ^{pot}")
        elif pot == 1:
            terminos_str.append(f"{'+' if c_adj >= 0 else '-'} {abs(c_adj):.4g}λ")
        else:
            terminos_str.append(f"{'+' if c_adj >= 0 else '-'} {abs(c_adj):.4g}")
    poly_str = " ".join(terminos_str).lstrip("+ ") + " = 0"

    # Malla de curva P(λ) para Plotly
    reales = [float(v.real) for v in autovalores if abs(v.imag) < 1e-8]
    if reales:
        lam_min_g, lam_max_g = min(reales), max(reales)
        pad = max(1.5, 0.25 * (lam_max_g - lam_min_g))
        lam_grid = np.linspace(lam_min_g - pad, lam_max_g + pad, 220)
    else:
        lam_grid = np.linspace(-10.0, 10.0, 220)
    with np.errstate(all="ignore"):
        p_vals = np.polyval([signo_ext * c for c in coef_poly], lam_grid)
    p_clean = [float(v) if np.isfinite(v) else None for v in p_vals.tolist()]

    return {
        "coeficientes_polinomio": coef_poly,
        "polinomio_texto": poly_str,
        "b_coeficientes": b_coefs,
        "determinante": det_A,
        "es_invertible": abs(det_A) > 1e-12,
        "inversa": A_inv_list,
        "error_inversa": error_inversa,
        "norma_cayley_hamilton": norma_cayley_hamilton,
        "descartes_polinomio": descartes_polinomio,
        "autovalores": [
            float(v.real) if abs(v.imag) < 1e-10 else complex(round(v.real, 6), round(v.imag, 6))
            for v in autovalores
        ],
        "autovalores_reales": reales,
        "autovectores": autovectores,
        "curva_polinomio": {"lambda": lam_grid.tolist(), "p_lambda": p_clean},
        "historial_pasos": historial_pasos,
        "duracion": time.perf_counter() - t_ini,
    }

