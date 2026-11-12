"""
core/lab4_sistemas.py
=====================
Módulo de cálculo puro para Sistemas de Ecuaciones Lineales y Autovalores/Autovectores (Laboratorio 4).

Incluye:
- Eliminación de Gauss (con o sin pivoteo parcial) y sustitución hacia atrás.
- Método de Gauss-Jordan (resolución de sistemas e inversión matricial paso a paso).
- Factorización LU (Algoritmo de Doolittle con sustitución progresiva y regresiva).
- Método Iterativo de Gauss-Seidel (verificación de dominancia diagonal y convergencia).
- Método de las Potencias (Autovalor Máximo dominante y Autovalor Mínimo vía matriz inversa).
- Método de Faddeev-Leverrier:
    * Coeficientes del Polinomio Característico P(lambda).
    * Autovalores analíticos / numéricos (raíces de P(lambda)).
    * Matriz Inversa analítica mediante matriz adjunta de Faddeev.
    * Autovectores asociados resolviendo (A - lambda*I)v = 0 vía SVD.

Regla estricta: Funciones puras, sin `input()` ni `print()`, atrapando singularidades
y divisiones por cero con excepciones descriptivas y retornos estructurados.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np


# ==============================================================================
# 1. ELIMINACIÓN DE GAUSS
# ==============================================================================

def eliminacion_gauss(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray,
    pivoteo: bool = True,
    decimales: int = 4,
) -> dict[str, Any]:
    """
    Resuelve el sistema Ax = b mediante el método de Eliminación de Gauss hacia adelante
    con sustitución hacia atrás.

    Parámetros:
    -----------
    A_matriz : Matriz de coeficientes (n x n).
    b_vector : Vector de términos independientes (n).
    pivoteo : bool. Si True, aplica pivoteo parcial por fila para estabilidad numérica.
    decimales : int. Cifras de redondeo en los reportes paso a paso.

    Retorna:
    --------
    dict con solución x, pasos de escalonamiento, vector residual r = Ax - b y norma de error.
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    b = np.array(b_vector, dtype=float).flatten()

    n = len(A)
    if A.shape != (n, n) or len(b) != n:
        raise ValueError(f"Dimensiones incompatibles: A tiene tamaño {A.shape} y b tiene longitud {len(b)}.")

    # Matriz aumentada [A | b]
    M = np.hstack([A, b.reshape(-1, 1)])
    pasos: list[dict[str, Any]] = [{
        "paso": 0,
        "descripcion": "Sistema aumentado inicial [A | b]",
        "matriz": np.round(M.copy(), decimales).tolist(),
        "operaciones": [],
    }]

    # Escalonamiento hacia adelante
    for k in range(n - 1):
        operaciones = []

        # Pivoteo parcial
        if pivoteo:
            fila_max = k + int(np.argmax(np.abs(M[k:, k])))
            if fila_max != k:
                M[[k, fila_max]] = M[[fila_max, k]]
                operaciones.append(f"Intercambio de filas: F_{k+1} ↔ F_{fila_max+1} (pivote mayor)")

        pivote = M[k, k]
        if abs(pivote) < 1e-14:
            raise ValueError(f"Pivote nulo o casi nulo detectado en posición ({k+1}, {k+1}). La matriz es singular o requiere reordenamiento.")

        for i in range(k + 1, n):
            factor = M[i, k] / pivote
            if abs(factor) > 1e-15:
                operaciones.append(f"F_{i+1} = F_{i+1} - ({factor:.{decimales}f}) · F_{k+1}")
                M[i, k:] -= factor * M[k, k:]
                M[i, k] = 0.0  # Anulación exacta para evitar ruido numérico

        pasos.append({
            "paso": k + 1,
            "descripcion": f"Eliminación en columna {k+1} debajo del pivote a_{k+1},{k+1}",
            "matriz": np.round(M.copy(), decimales).tolist(),
            "operaciones": operaciones,
        })

    # Verificación de rango / unicidad en la última fila
    if abs(M[n - 1, n - 1]) < 1e-14:
        raise ValueError("El sistema no tiene solución única (matriz de coeficientes singular).")

    # Sustitución hacia atrás
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

    # Verificación del residual Ax - b
    Ax = np.dot(A, x)
    residual = Ax - b
    norma_residual = float(np.linalg.norm(residual))

    duracion = time.perf_counter() - t_ini

    return {
        "solucion": x.tolist(),
        "solucion_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
        "pasos_escalonamiento": pasos,
        "pasos_sustitucion": pasos_sustitucion,
        "residual": residual.tolist(),
        "norma_error": norma_residual,
        "duracion": duracion,
    }


# ==============================================================================
# 2. MÉTODO DE GAUSS-JORDAN
# ==============================================================================

def gauss_jordan(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray | None = None,
    pivoteo: bool = True,
    decimales: int = 4,
) -> dict[str, Any]:
    """
    Aplica el Método de Gauss-Jordan para transformar la matriz en su forma escalonada
    reducida por filas (RREF).
    - Si se provee b, resuelve Ax = b.
    - Si b es None, calcula la Matriz Inversa [A | I] -> [I | A^-1].
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)

    if b_vector is not None:
        b = np.array(b_vector, dtype=float).flatten()
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

        # Pivoteo parcial
        if pivoteo:
            fila_max = i + int(np.argmax(np.abs(M[i:, i])))
            if fila_max != i:
                M[[i, fila_max]] = M[[fila_max, i]]
                operaciones.append(f"Intercambio: F_{i+1} ↔ F_{fila_max+1}")

        pivote = M[i, i]
        if abs(pivote) < 1e-14:
            raise ValueError(f"Pivote singular o casi nulo detectado en ({i+1}, {i+1}). La matriz no es invertible o el sistema carece de solución única.")

        # Normalización del pivote a 1
        if abs(pivote - 1.0) > 1e-15:
            M[i, :] = M[i, :] / pivote
            operaciones.append(f"Normalización: F_{i+1} = F_{i+1} / ({pivote:.{decimales}f})")

        # Eliminación arriba y abajo del pivote
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
        solucion = M[:, -1].tolist()
        Ax = np.dot(A, solucion)
        norma_err = float(np.linalg.norm(Ax - b))
        return {
            "modo": "sistema",
            "solucion": solucion,
            "solucion_dict": {f"x_{i+1}": s for i, s in enumerate(solucion)},
            "pasos": pasos,
            "norma_error": norma_err,
            "duracion": duracion,
        }
    else:
        A_inv = M[:, n:].copy()
        prod = np.dot(A, A_inv)
        norma_err = float(np.linalg.norm(prod - np.eye(n)))
        return {
            "modo": "inversa",
            "inversa": A_inv.tolist(),
            "pasos": pasos,
            "norma_error": norma_err,
            "duracion": duracion,
        }


# ==============================================================================
# 3. FACTORIZACIÓN LU (DOOLITTLE)
# ==============================================================================

def factorizacion_lu(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray | None = None,
    decimales: int = 4,
) -> dict[str, Any]:
    """
    Descompone la matriz A en el producto A = L · U (Doolittle, L con diagonal unitaria).
    Si se suministra b, resuelve sucesivamente:
        1. L · y = b  (Sustitución hacia adelante)
        2. U · x = y  (Sustitución hacia atrás)
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    if A.shape != (n, n):
        raise ValueError(f"La matriz A debe ser cuadrada. Dimensiones recibidas: {A.shape}")

    L = np.eye(n, dtype=float)
    U = A.copy()
    pasos_eliminacion = []

    for k in range(n - 1):
        pivote = U[k, k]
        if abs(pivote) < 1e-14:
            raise ValueError(f"Pivote nulo en U[{k+1},{k+1}]. La matriz requiere pivoteo y no admite factorización LU directa sin permutación.")

        factores_paso = []
        for i in range(k + 1, n):
            factor = U[i, k] / pivote
            L[i, k] = factor
            factores_paso.append({
                "fila_i": i + 1,
                "fila_k": k + 1,
                "factor": factor,
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

    # Verificación L · U = A
    LU = np.dot(L, U)
    error_LU = float(np.linalg.norm(LU - A))

    res: dict[str, Any] = {
        "L": L.tolist(),
        "U": U.tolist(),
        "error_LU": error_LU,
        "pasos_eliminacion": pasos_eliminacion,
    }

    if b_vector is not None:
        b = np.array(b_vector, dtype=float).flatten()

        # 1. Resolver L · y = b
        y = np.zeros(n, dtype=float)
        pasos_y = []
        for i in range(n):
            suma = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (b[i] - suma) / L[i, i]
            pasos_y.append({
                "variable": f"y_{i+1}",
                "ecuacion": f"({b[i]:.{decimales}f} - {suma:.{decimales}f}) / {L[i, i]:.{decimales}f}",
                "valor": float(y[i]),
            })

        # 2. Resolver U · x = y
        x = np.zeros(n, dtype=float)
        pasos_x = []
        for i in range(n - 1, -1, -1):
            suma = sum(U[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - suma) / U[i, i]
            pasos_x.append({
                "variable": f"x_{i+1}",
                "ecuacion": f"({y[i]:.{decimales}f} - {suma:.{decimales}f}) / {U[i, i]:.{decimales}f}",
                "valor": float(x[i]),
            })
        pasos_x.reverse()

        Ax = np.dot(A, x)
        error_Ax = float(np.linalg.norm(Ax - b))

        res.update({
            "y": y.tolist(),
            "x": x.tolist(),
            "solucion_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
            "pasos_y": pasos_y,
            "pasos_x": pasos_x,
            "error_Ax": error_Ax,
        })

    res["duracion"] = time.perf_counter() - t_ini
    return res


# ==============================================================================
# 4. MÉTODO ITERATIVO DE GAUSS-SEIDEL
# ==============================================================================

def verificar_dominancia_diagonal(A_matriz: list[list[float]] | np.ndarray) -> dict[str, Any]:
    """
    Verifica si una matriz es estrictamente diagonal dominante:
        |a_{ii}| > sum_{j != i} |a_{ij}|  para todo i.
    """
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
            "fila": i + 1,
            "diagonal": diag,
            "suma_otros": suma_otros,
            "cumple": cumple,
            "diferencia": diag - suma_otros,
        })

    return {
        "es_dominante": es_dominante,
        "filas": filas_info,
        "diagnostico": "La matriz ES estrictamente diagonal dominante (convergencia garantizada)." if es_dominante else "La matriz NO es estrictamente diagonal dominante. El método puede converger o divergir.",
    }


def gauss_seidel(
    A_matriz: list[list[float]] | np.ndarray,
    b_vector: list[float] | np.ndarray,
    x0: list[float] | np.ndarray | None = None,
    tol: float = 0.01,
    max_iter: int = 100,
) -> dict[str, Any]:
    """
    Resuelve el sistema Ax = b mediante el método iterativo de Gauss-Seidel:
        x_i^{(k+1)} = (b_i - sum_{j < i} a_{ij} x_j^{(k+1)} - sum_{j > i} a_{ij} x_j^{(k)}) / a_{ii}
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    b = np.array(b_vector, dtype=float).flatten()
    n = len(b)

    # Validar que ningún coeficiente diagonal sea nulo
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(f"Elemento diagonal nulo detectado en a[{i+1},{i+1}]. No se puede despejar la variable.")

    dominancia = verificar_dominancia_diagonal(A)

    x = np.ones(n, dtype=float) if x0 is None else np.array(x0, dtype=float).flatten()
    historial: list[dict[str, Any]] = []
    convergente = False

    for k in range(1, max_iter + 1):
        x_anterior = x.copy()
        despejes_paso = []

        for i in range(n):
            suma = b[i]
            for j in range(n):
                if j != i:
                    suma -= A[i, j] * x[j]
            x[i] = suma / A[i, i]
            despejes_paso.append(float(x[i]))

        errores = np.abs(x - x_anterior)
        error_max = float(np.max(errores))
        cumple = error_max < tol

        historial.append({
            "iteracion": k,
            "x": x.copy().tolist(),
            "x_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
            "errores_componentes": errores.tolist(),
            "error_max": error_max,
            "cumple_tol": cumple,
        })

        if cumple:
            convergente = True
            break

    duracion = time.perf_counter() - t_ini
    Ax = np.dot(A, x)
    norma_residuo = float(np.linalg.norm(Ax - b))

    return {
        "solucion": x.tolist(),
        "solucion_dict": {f"x_{i+1}": float(x[i]) for i in range(n)},
        "iteraciones": len(historial),
        "error_final": historial[-1]["error_max"] if historial else 0.0,
        "convergente": convergente,
        "dominancia": dominancia,
        "norma_residuo": norma_residuo,
        "duracion": duracion,
        "historial": historial,
    }


# ==============================================================================
# 5. MÉTODO DE LAS POTENCIAS (MÁXIMO Y MÍNIMO AUTOVALOR)
# ==============================================================================

def metodo_potencias_max(
    A_matriz: list[list[float]] | np.ndarray,
    x0: list[float] | np.ndarray | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> dict[str, Any]:
    """
    Calcula el autovalor dominante (de mayor magnitud) y su autovector asociado
    mediante el Método de las Potencias con normalización de norma infinito.
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    if A.shape != (n, n):
        raise ValueError("La matriz debe ser cuadrada.")

    x = np.ones(n, dtype=float) if x0 is None else np.array(x0, dtype=float).flatten()
    if np.all(np.abs(x) < 1e-15):
        raise ValueError("El vector inicial x0 no puede ser el vector nulo.")

    lambda_old = 0.0
    historial: list[dict[str, Any]] = []
    convergente = False

    for k in range(1, max_iter + 1):
        y = np.dot(A, x)
        # Normalización con la componente de máxima magnitud conservando el signo
        idx_max = int(np.argmax(np.abs(y)))
        lambda_new = float(y[idx_max])

        if abs(lambda_new) < 1e-15:
            historial.append({
                "iteracion": k,
                "y": y.tolist(),
                "lambda": 0.0,
                "x_norm": x.tolist(),
                "error_pct": 0.0,
            })
            break

        x_norm = y / lambda_new

        if k > 1:
            err_pct = abs(lambda_new - lambda_old) / abs(lambda_new) * 100.0
        else:
            err_pct = None

        cumple = (err_pct is not None and err_pct < tol)

        historial.append({
            "iteracion": k,
            "y": y.tolist(),
            "lambda": lambda_new,
            "x_norm": x_norm.tolist(),
            "error_pct": err_pct,
            "cumple_tol": cumple,
        })

        if cumple:
            convergente = True
            x = x_norm
            break

        lambda_old = lambda_new
        x = x_norm

    duracion = time.perf_counter() - t_ini
    Ax = np.dot(A, x)
    error_autovector = float(np.linalg.norm(Ax - lambda_new * x))

    return {
        "autovalor_max": lambda_new,
        "autovector": x.tolist(),
        "iteraciones": len(historial),
        "error_final_pct": historial[-1]["error_pct"] if historial and historial[-1]["error_pct"] is not None else 0.0,
        "convergente": convergente,
        "error_autovector": error_autovector,
        "duracion": duracion,
        "historial": historial,
    }


def metodo_potencias_min(
    A_matriz: list[list[float]] | np.ndarray,
    x0: list[float] | np.ndarray | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> dict[str, Any]:
    """
    Calcula el autovalor de menor magnitud aplicando el Método de las Potencias
    a la matriz inversa A^-1 (Método de las Potencias Inverso):
        lambda_min(A) = 1 / lambda_max(A^-1).
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("La matriz es singular y no posee inversa. El autovalor mínimo es 0.")

    res_inv = metodo_potencias_max(A_inv, x0=x0, tol=tol, max_iter=max_iter)
    lambda_inv = res_inv["autovalor_max"]

    if abs(lambda_inv) < 1e-15:
        raise ValueError("El autovalor de A^-1 es nulo.")

    lambda_min = 1.0 / lambda_inv
    duracion = time.perf_counter() - t_ini

    return {
        "autovalor_min": lambda_min,
        "autovalor_inversa": lambda_inv,
        "autovector": res_inv["autovector"],
        "A_inversa": A_inv.tolist(),
        "iteraciones": res_inv["iteraciones"],
        "error_final_pct": res_inv["error_final_pct"],
        "convergente": res_inv["convergente"],
        "duracion": duracion,
        "historial": res_inv["historial"],
    }


# ==============================================================================
# 6. MÉTODO DE FADDEEV-LEVERRIER
# ==============================================================================

def metodo_faddeev_leverrier(A_matriz: list[list[float]] | np.ndarray) -> dict[str, Any]:
    """
    Calcula el Polinomio Característico P(lambda), los Autovalores, la Matriz Inversa
    y los Autovectores asociados mediante el algoritmo de recurrencia de Faddeev-Leverrier:
        B_1 = A,          b_1 = tr(B_1)
        B_k = A(B_{k-1} - b_{k-1}*I),   b_k = tr(B_k) / k
    """
    t_ini = time.perf_counter()
    A = np.array(A_matriz, dtype=float)
    n = len(A)
    if A.shape != (n, n):
        raise ValueError("La matriz debe ser cuadrada.")

    I = np.eye(n, dtype=float)
    B_prev = A.copy()
    b_coefs: list[float] = []
    B_matrices: list[list[list[float]]] = [A.copy().tolist()]
    historial_pasos = []

    # Iteración 1
    b1 = float(np.trace(B_prev))
    b_coefs.append(b1)
    historial_pasos.append({
        "k": 1,
        "formula": "B_1 = A",
        "B_k": B_prev.tolist(),
        "traza": b1,
        "b_k": b1,
    })

    # Iteraciones k = 2..n
    for k in range(2, n + 1):
        Bk = np.dot(A, B_prev - b_coefs[-1] * I)
        traza = float(np.trace(Bk))
        bk = traza / k
        b_coefs.append(bk)
        B_matrices.append(Bk.tolist())

        historial_pasos.append({
            "k": k,
            "formula": f"B_{k} = A · (B_{k-1} - b_{k-1} · I)",
            "B_k": Bk.tolist(),
            "traza": traza,
            "b_k": bk,
        })
        B_prev = Bk

    # Coeficientes del polinomio característico:
    # P(lambda) = lambda^n - b_1*lambda^{n-1} - ... - b_n
    coef_poly = [1.0] + [-bk for bk in b_coefs]

    # Determinante: det(A) = (-1)^n * c_n = (-1)^{n-1} * b_n
    det_A = float(((-1) ** n) * coef_poly[-1])

    # Matriz Inversa por fórmula de Faddeev: A^-1 = (1 / det(A)) * (B_{n-1} - b_{n-1} * I)
    if abs(det_A) > 1e-12:
        b_n_minus_1 = b_coefs[-2]
        B_n_minus_1 = np.array(B_matrices[-2], dtype=float)
        adj_A = B_n_minus_1 - b_n_minus_1 * I
        A_inv = (1.0 / det_A) * adj_A
        error_inversa = float(np.linalg.norm(np.dot(A, A_inv) - I))
        A_inv_list = A_inv.tolist()
    else:
        A_inv_list = None
        error_inversa = None

    # Autovalores (raíces del polinomio característico)
    autovalores_crudos = np.roots(coef_poly)
    # Ordenar por magnitud descendente
    autovalores = sorted(autovalores_crudos, key=lambda val: abs(val), reverse=True)

    # Autovectores asociados vía SVD en (A - lambda * I)
    autovectores = []
    for eig in autovalores:
        try:
            A_comp = A.astype(complex) - eig * np.eye(n, dtype=complex)
            _, _, vh = np.linalg.svd(A_comp)
            vec = vh[-1, :].conj()
            # Normalizar para que la norma sea 1
            norm_v = np.linalg.norm(vec)
            if norm_v > 1e-14:
                vec = vec / norm_v
            # Si la parte imaginaria es insignificante, convertir a real
            if np.all(np.abs(vec.imag) < 1e-10):
                vec_salida = vec.real.tolist()
            else:
                vec_salida = [complex(round(c.real, 6), round(c.imag, 6)) for c in vec]
            autovectores.append(vec_salida)
        except Exception:
            autovectores.append(None)

    duracion = time.perf_counter() - t_ini

    return {
        "coeficientes_polinomio": coef_poly,
        "b_coeficientes": b_coefs,
        "determinante": det_A,
        "es_invertible": abs(det_A) > 1e-12,
        "inversa": A_inv_list,
        "error_inversa": error_inversa,
        "autovalores": [float(v.real) if abs(v.imag) < 1e-10 else complex(round(v.real, 6), round(v.imag, 6)) for v in autovalores],
        "autovectores": autovectores,
        "pasos_recurrentes": historial_pasos,
        "duracion": duracion,
    }

