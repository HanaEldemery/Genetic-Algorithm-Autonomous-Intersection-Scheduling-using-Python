# Genetic-Algorithm-Autonomous-Intersection-Scheduling-using-Python

This project applied a Genetic Algorithm (GA) to the problem of scheduling autonomous vehicles through a 4-way intersection without collisions. Drawing inspiration from biological evolution. Rather than refining a single solution over time, GAs maintain an entire population of candidate solutions and evolve them across generations through selection, crossover, and mutation.

Each candidate solution (chromosome) was represented as a vector of continuous start times — one per vehicle. A population of these schedules was randomly initialized within a feasible range, then evaluated using a weighted fitness function combining makespan and total waiting time. Across each generation, higher-quality solutions were more likely to be selected as parents, their schedules were combined via crossover to produce offspring, and small random mutations were applied to maintain diversity and avoid premature convergence.

Over successive generations, the population converged toward increasingly optimal solutions. This project deepened my understanding of population-based search, the balance between exploration and exploitation, and how domain-specific repair strategies can make evolutionary algorithms practical for constrained optimization problems.
