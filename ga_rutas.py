"""
ga_rutas.py — Algoritmo genético para itinerarios turísticos (variante de orientación / TSP selectivo).

Cromosoma: permutación de los atractivos candidatos.
Decodificación: desde el punto de salida se recorren los genes en orden y se agrega cada atractivo
solo si, contando el traslado, la visita y el regreso al punto de salida, no se excede el tiempo
disponible ni el presupuesto. Así cualquier permutación produce un itinerario factible.

Fitness (adaptación de la fórmula de la propuesta, con tiempo y costo normalizados):
    F = S / (1 + w_t * T / T_max + w_c * C / B)
S = satisfacción acumulada de los atractivos visitados, T = minutos totales (traslado + visita),
C = costo total (entradas + transporte). Normalizar evita sumar minutos con pesos.

Operadores: selección por torneo, cruce PMX, mutación por inversión y elitismo.
"""
from __future__ import annotations
import itertools
import numpy as np


class Instancia:
    def __init__(self, minutos, km, inicio, candidatos, satisfaccion, visita_min, entrada_cop,
                 t_max, presupuesto, costo_km=600.0, w_t=1.0, w_c=1.0):
        self.minutos = np.asarray(minutos, dtype=float)
        self.km = np.asarray(km, dtype=float)
        self.inicio = int(inicio)
        self.candidatos = np.asarray(candidatos, dtype=int)
        self.sat = np.asarray(satisfaccion, dtype=float)
        self.visita = np.asarray(visita_min, dtype=float)
        self.entrada = np.asarray(entrada_cop, dtype=float)
        self.t_max, self.B = float(t_max), float(presupuesto)
        self.costo_km, self.w_t, self.w_c = float(costo_km), float(w_t), float(w_c)

    def decodificar(self, crom):
        """Devuelve (ruta de índices de nodo sin el inicio, minutos, costo, satisfacción)."""
        ruta, t, c, s, actual = [], 0.0, 0.0, 0.0, self.inicio
        for g in crom:
            nodo = self.candidatos[g]
            t_nuevo = t + self.minutos[actual, nodo] + self.visita[nodo]
            c_nuevo = c + self.km[actual, nodo] * self.costo_km + self.entrada[nodo]
            # debe alcanzar para volver al punto de salida
            if t_nuevo + self.minutos[nodo, self.inicio] <= self.t_max and \
               c_nuevo + self.km[nodo, self.inicio] * self.costo_km <= self.B:
                ruta.append(int(nodo)); t, c, s, actual = t_nuevo, c_nuevo, s + self.sat[nodo], nodo
        t += self.minutos[actual, self.inicio]; c += self.km[actual, self.inicio] * self.costo_km
        return ruta, t, c, s

    def fitness_de(self, t, c, s):
        return s / (1.0 + self.w_t * t / self.t_max + self.w_c * c / self.B) if s > 0 else 0.0

    def fitness(self, crom):
        _, t, c, s = self.decodificar(crom)
        return self.fitness_de(t, c, s)


# ---------------- operadores ----------------
def torneo(pob, fit, k, rng):
    idx = rng.choice(len(pob), size=k, replace=False)
    return pob[idx[np.argmax(fit[idx])]]


def pmx(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n + 1, size=2, replace=False))
    hijo = -np.ones(n, dtype=int)
    hijo[a:b] = p1[a:b]
    pos_p1 = {v: i for i, v in enumerate(p1)}
    for i in range(a, b):
        v = p2[i]
        if v in hijo[a:b]:
            continue
        j = i
        while a <= j < b:          # sigue el mapeo hasta encontrar una posición libre
            j = int(np.where(p2 == p1[j])[0][0])
        hijo[j] = v
    for i in range(n):
        if hijo[i] == -1:
            hijo[i] = p2[i]
    return hijo


def mutacion_inversion(c, rng):
    c = c.copy()
    a, b = sorted(rng.choice(len(c), size=2, replace=False))
    c[a:b + 1] = c[a:b + 1][::-1]
    return c


# ---------------- algoritmo genético ----------------
def algoritmo_genetico(inst: Instancia, tam_pob=80, generaciones=150, p_cruce=0.9, p_mut=0.3,
                       k_torneo=3, elite=2, semilla=42):
    rng = np.random.default_rng(semilla)
    n = len(inst.candidatos)
    pob = np.array([rng.permutation(n) for _ in range(tam_pob)])
    fit = np.array([inst.fitness(c) for c in pob])
    historia = [float(fit.max())]; media = [float(fit.mean())]
    evaluaciones = tam_pob
    for _ in range(generaciones):
        orden = np.argsort(-fit)
        nueva = [pob[i].copy() for i in orden[:elite]]
        while len(nueva) < tam_pob:
            p1, p2 = torneo(pob, fit, k_torneo, rng), torneo(pob, fit, k_torneo, rng)
            h = pmx(p1, p2, rng) if rng.random() < p_cruce else p1.copy()
            if rng.random() < p_mut:
                h = mutacion_inversion(h, rng)
            nueva.append(h)
        pob = np.array(nueva)
        fit = np.array([inst.fitness(c) for c in pob])
        evaluaciones += tam_pob - elite
        historia.append(float(fit.max())); media.append(float(fit.mean()))
    mejor = pob[np.argmax(fit)]
    ruta, t, c, s = inst.decodificar(mejor)
    return dict(ruta=ruta, minutos=t, costo=c, satisfaccion=s, fitness=float(fit.max()),
                historia=historia, media=media, evaluaciones=evaluaciones)


# ---------------- líneas base ----------------
def busqueda_aleatoria(inst: Instancia, evaluaciones, semilla=42):
    rng = np.random.default_rng(semilla)
    n = len(inst.candidatos); mejor, mejor_f, hist = None, -1.0, []
    for _ in range(evaluaciones):
        c = rng.permutation(n); f = inst.fitness(c)
        if f > mejor_f: mejor, mejor_f = c, f
        hist.append(mejor_f)
    ruta, t, cst, s = inst.decodificar(mejor)
    return dict(ruta=ruta, minutos=t, costo=cst, satisfaccion=s, fitness=mejor_f, historia=hist)


def voraz(inst: Instancia):
    """Vecino 'más rentable': en cada paso elige el atractivo factible con mayor satisfacción por minuto adicional."""
    pendientes = list(range(len(inst.candidatos))); orden = []; actual = inst.inicio; t = c = 0.0
    while pendientes:
        mejor_g, mejor_r = None, -1.0
        for g in pendientes:
            nodo = inst.candidatos[g]
            dt = inst.minutos[actual, nodo] + inst.visita[nodo]
            dc = inst.km[actual, nodo] * inst.costo_km + inst.entrada[nodo]
            if t + dt + inst.minutos[nodo, inst.inicio] <= inst.t_max and c + dc + inst.km[nodo, inst.inicio] * inst.costo_km <= inst.B:
                r = inst.sat[nodo] / max(dt, 1.0)
                if r > mejor_r: mejor_g, mejor_r = g, r
        if mejor_g is None: break
        nodo = inst.candidatos[mejor_g]
        t += inst.minutos[actual, nodo] + inst.visita[nodo]; c += inst.km[actual, nodo] * inst.costo_km + inst.entrada[nodo]
        actual = nodo; orden.append(mejor_g); pendientes.remove(mejor_g)
    crom = np.array(orden + [g for g in range(len(inst.candidatos)) if g not in orden])
    ruta, t, c, s = inst.decodificar(crom)
    return dict(ruta=ruta, minutos=t, costo=c, satisfaccion=s, fitness=inst.fitness_de(t, c, s))


def optimo_exhaustivo(inst: Instancia, max_paradas=None):
    """Recorre todos los subconjuntos ordenados (solo para instancias pequeñas, n <= 8)."""
    n = len(inst.candidatos); mejor = (-1.0, None)
    for k in range(1, (max_paradas or n) + 1):
        for perm in itertools.permutations(range(n), k):
            ruta, t, c, s = inst.decodificar(list(perm))
            if len(ruta) == k:
                f = inst.fitness_de(t, c, s)
                if f > mejor[0]: mejor = (f, (ruta, t, c, s))
    f, (ruta, t, c, s) = mejor
    return dict(ruta=ruta, minutos=t, costo=c, satisfaccion=s, fitness=f)
