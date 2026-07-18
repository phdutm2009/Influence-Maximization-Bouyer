import random
import numpy as np

def run_ic(G, seeds, p, simulations=1000):

    total_spread = 0
    total_coverage = 0

    spreads = []   # ذخیره spread هر اجرا

    for _ in range(simulations):

        active = set(seeds)
        new_active = list(seeds)

        while new_active:

            next_active = []

            for u in new_active:

                for v in G.neighbors(u):

                    if v not in active and random.random() < p:

                        active.add(v)
                        next_active.append(v)

            new_active = next_active

        spread = len(active)

        coverage = spread / G.number_of_nodes()

        total_spread += spread
        total_coverage += coverage

        spreads.append(spread)

    avg_spread = total_spread / simulations

    avg_coverage = (total_coverage / simulations) * 100

    variance_spread = np.var(spreads)

    return avg_spread, avg_coverage, variance_spread