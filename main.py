import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from algorithms import HeldKarp, GRASP

def matriz(n, seed=42):
    np.random.seed(seed)
    coords = np.random.randint(0, 100, size=(n, 2))
    dist_matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matriz[i][j] = dist_matriz[j][i] = int(dist)
    return dist_matriz

N_INICIAL = 3
N_FINAL = 22
GRASP_RUNS = 20

resultados = {
    "N": [],
    "Tempo_HK": [],
    "Tempo_GRASP": [],
    "Custo_HK": [],
    "Custo_GRASP": [],
    "Gap_Qualidade": {}
}

print(f"Iniciando análise comparativa de N={N_INICIAL} até N={N_FINAL}...")
print("-" * 60)
print(f"{'N':<5} | {'Tempo HK (s)':<15} | {'Tempo GRASP (s)':<15} | {'Gap Médio (%)':<15}")
print("-" * 60)

for n in range(N_INICIAL, N_FINAL + 1):
    dist_matriz = matriz(n, seed=n)

    hk = HeldKarp(dist_matriz)
    start_hk = time.perf_counter()
    custo_hk = hk.solve()
    end_hk = time.perf_counter()
    tempo_hk = end_hk - start_hk

    gaps_n = []
    start_gr = time.perf_counter()
    for _ in range(GRASP_RUNS):
        grasp = GRASP(dist_matriz, max_iterations=50)
        custo_grasp = grasp.solve()
        gap = ((custo_grasp - custo_hk) / custo_hk) * 100
        gaps_n.append(gap)
    end_gr = time.perf_counter()
    tempo_grasp = (end_gr - start_gr) / GRASP_RUNS

    resultados["N"].append(n)
    resultados["Tempo_HK"].append(tempo_hk)
    resultados["Tempo_GRASP"].append(tempo_grasp)
    resultados["Custo_HK"].append(custo_hk)
    resultados["Custo_GRASP"].append(np.mean([
        custo_hk * (1 + g / 100) for g in gaps_n
    ]))
    resultados["Gap_Qualidade"][n] = gaps_n

    print(f"{n:<5} | {tempo_hk:<15.5f} | {tempo_grasp:<15.5f} | {np.mean(gaps_n):<15.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(resultados["N"], resultados["Tempo_HK"], marker='o', label='Held-Karp', color='red')
ax1.plot(resultados["N"], resultados["Tempo_GRASP"], marker='s', label='GRASP', color='blue')
ax1.set_xlabel('Número de Cidades (N)')
ax1.set_ylabel('Tempo de Execução (segundos)')
ax1.set_title('Comparação de Desempenho Temporal')
ax1.legend()
ax1.grid(True)

dados_boxplot = [resultados["Gap_Qualidade"][n] for n in resultados["N"]]

ax2.boxplot(dados_boxplot, labels=resultados["N"], showfliers=True)
ax2.set_xlabel('Número de Cidades (N)')
ax2.set_ylabel('Erro Relativo (%) em relação ao Ótimo')
ax2.set_title('Distribuição do Gap do GRASP')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
