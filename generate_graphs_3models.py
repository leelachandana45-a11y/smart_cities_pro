import matplotlib.pyplot as plt
import numpy as np

# x-axis (evaluation index)
x = np.arange(1, 6)

# Data for models
CNN = {
    "A": [0.82, 0.84, 0.86, 0.87, 0.89],
    "L": [0.45, 0.40, 0.35, 0.32, 0.30],
    "P": [0.80, 0.82, 0.84, 0.85, 0.87],
    "R": [0.78, 0.80, 0.83, 0.85, 0.86],
    "F": [0.79, 0.81, 0.835, 0.85, 0.865],
    "T": [55, 52, 50, 48, 45],
    "N": [0.72, 0.74, 0.76, 0.78, 0.80],
    "C": [2.3, 2.4, 2.5, 2.6, 2.7],
    "S": [0.80, 0.82, 0.84, 0.86, 0.88],
    "E": [0.85, 0.87, 0.89, 0.90, 0.92],
    "G": [0.80, 0.82, 0.84, 0.85, 0.87],
    "O": [0.83, 0.85, 0.87, 0.88, 0.90]
}

SIW = {
    "A": [0.85, 0.87, 0.88, 0.89, 0.91],
    "L": [0.40, 0.35, 0.30, 0.28, 0.25],
    "P": [0.83, 0.85, 0.87, 0.88, 0.90],
    "R": [0.82, 0.84, 0.86, 0.87, 0.89],
    "F": [0.825, 0.845, 0.865, 0.875, 0.895],
    "T": [50, 48, 45, 43, 40],
    "N": [0.75, 0.77, 0.80, 0.82, 0.84],
    "C": [2.8, 2.9, 3.0, 3.1, 3.2],
    "S": [0.83, 0.85, 0.87, 0.89, 0.91],
    "E": [0.87, 0.89, 0.91, 0.92, 0.94],
    "G": [0.82, 0.84, 0.86, 0.88, 0.89],
    "O": [0.86, 0.88, 0.89, 0.91, 0.93]
}

RAD = {
    "A": [0.78, 0.80, 0.82, 0.83, 0.85],
    "L": [0.50, 0.48, 0.45, 0.42, 0.40],
    "P": [0.75, 0.78, 0.80, 0.82, 0.83],
    "R": [0.74, 0.76, 0.78, 0.80, 0.81],
    "F": [0.745, 0.77, 0.79, 0.81, 0.82],
    "T": [60, 58, 55, 53, 50],
    "N": [0.70, 0.72, 0.74, 0.75, 0.77],
    "C": [2.0, 2.1, 2.2, 2.3, 2.4],
    "S": [0.78, 0.80, 0.82, 0.83, 0.85],
    "E": [0.80, 0.82, 0.84, 0.85, 0.87],
    "G": [0.78, 0.80, 0.81, 0.83, 0.84],
    "O": [0.80, 0.82, 0.83, 0.85, 0.86]
}

# plotting function (NO TITLE)
def plot(metric, ylabel, filename):
    plt.figure()

    plt.plot(x, CNN[metric], marker='o', label=r"$\mathcal{M}_1$ (CNN-R)")
    plt.plot(x, SIW[metric], marker='s', label=r"$\mathcal{M}_2$ (SIW)")
    plt.plot(x, RAD[metric], marker='^', label=r"$\mathcal{M}_3$ (RAD)")

    plt.xlabel(r"$\mathcal{X}_i$")
    plt.ylabel(ylabel)

    plt.legend()
    plt.grid()

    plt.savefig(filename)
    plt.close()

# 12 graphs

plot("A", r"$\mathcal{A}_i$", "g1_accuracy.pdf")
plot("L", r"$\mathcal{L}_i$", "g2_loss.pdf")
plot("P", r"$\mathcal{P}_i$", "g3_precision.pdf")
plot("R", r"$\mathcal{R}_i$", "g4_recall.pdf")
plot("F", r"$\mathcal{F}_i$", "g5_f1.pdf")
plot("T", r"$\tau_i$", "g6_latency.pdf")
plot("N", r"$\mathcal{N}_i$", "g7_noise.pdf")
plot("C", r"$\mathcal{C}_i$", "g8_complexity.pdf")
plot("S", r"$\mathcal{S}_i$", "g9_stability.pdf")
plot("E", r"$\eta_i$", "g10_efficiency.pdf")
plot("G", r"$\mathcal{G}_i$", "g11_geo.pdf")
plot("O", r"$\mathcal{O}_i$", "g12_overall.pdf")

print("✅ 12 graphs generated (no titles, 3 models)")