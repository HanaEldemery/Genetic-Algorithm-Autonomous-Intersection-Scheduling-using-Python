import random
import math
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

def visualize_intersection_graphical(cars, x0, start_times, finish_times, makespan):
    """
    Creates an animated graphical visualization of the intersection
    """
    # Create the sequence
    lane_count = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
    sequence = []
    for lane in x0:
        lane_count[lane] += 1
        car_index = lane_count[lane]
        found = False
        for car in cars:
            if car['lane'] == lane and car['i'] == car_index:
                sequence.append(car)
                found = True
                break
        # If not found (e.g., single car case), just append the first car with that lane
        if not found:
            for car in cars:
                if car['lane'] == lane:
                    sequence.append(car)
                    break

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+100+0")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw static intersection
    draw_intersection_base(ax)
    
    # Car colors for each lane
    colors = {'N': 'blue', 'S': 'red', 'E': 'green', 'W': 'orange'}
    
    # Initialize car objects
    car_patches = []
    car_texts = []
    for i, car in enumerate(sequence):
        color = colors[car['lane']]
        car_patch = patches.Circle((0, 0), 0.5, fc=color, ec='black', linewidth=2, alpha=0.8)
        ax.add_patch(car_patch)
        car_patches.append(car_patch)
        
        text = ax.text(0, 0, f"{car['lane']}_{car['dir']}", ha='center', va='center', 
                      fontsize=8, fontweight='bold', color='white')
        car_texts.append(text)
    
    # Status text
    time_text = ax.text(0, 13, '', ha='center', fontsize=14, fontweight='bold')
    status_text = ax.text(0, -13, '', ha='center', fontsize=10)
    
    max_time = int(makespan) + 1
    
    def get_car_position(lane, direction, progress):
        """
        Calculate car position based on lane, direction, and progress (0-1)
        progress: 0 = entering, 1 = exiting
        """
        # Starting positions (before intersection)
        starts = {
            'N': (0, 2),
            'S': (0, -2),
            'E': (2, 0),
            'W': (-2, 0)
        }
        
        # Ending positions (after intersection) based on direction
        ends = {
            ('N', 'R'): (-12, 0),   # North left to West
            ('N', 'S'): (0, -12),   # North straight to South
            ('N', 'L'): (12, 0),    # North right to East
            ('S', 'R'): (12, 0),    # South left to East
            ('S', 'S'): (0, 12),    # South straight to North
            ('S', 'L'): (-12, 0),   # South right to West
            ('E', 'R'): (0, 12),    # East left to North
            ('E', 'S'): (-12, 0),   # East straight to West
            ('E', 'L'): (0, -12),   # East right to South
            ('W', 'R'): (0, -12),   # West left to South
            ('W', 'S'): (12, 0),    # West straight to East
            ('W', 'L'): (0, 12)     # West right to North
        }
        
        start = starts[lane]
        end = ends[(lane, direction)]
        
        # For turns, create curved path
        if direction == 'L' or direction == 'R':
            # Create arc for turn
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Add curve to the path
            if progress < 0.5:
                t = progress * 2
                x = start[0] * (1 - t) + mid_x * t
                y = start[1] * (1 - t) + mid_y * t
            else:
                t = (progress - 0.5) * 2
                x = mid_x * (1 - t) + end[0] * t
                y = mid_y * (1 - t) + end[1] * t
        else:
            # Straight path
            x = start[0] * (1 - progress) + end[0] * progress
            y = start[1] * (1 - progress) + end[1] * progress
        
        return x, y
    
    def animate(frame):
        t = frame * 0.45  # Each frame = 0.45 time units
        
        if t > max_time:
            return car_patches + car_texts + [time_text, status_text]
        
        time_text.set_text(f'Time: {t:.1f} / {makespan}')
        
        waiting = 0
        moving = 0
        done = 0
        
        for i, car in enumerate(sequence):
            start = start_times[i]
            finish = finish_times[i]
            
            if t < start:
                # Waiting - hide car
                car_patches[i].set_visible(False)
                car_texts[i].set_visible(False)
                waiting += 1
            elif start <= t <= finish:
            #elif start <= t < finish:
                # Moving through intersection
                car_patches[i].set_visible(True)
                car_texts[i].set_visible(True)
                moving += 1
                
                # Calculate progress through intersection (0 to 1)
                progress = (t - start) / (finish - start)
                x, y = get_car_position(car['lane'], car['dir'], progress)
                
                car_patches[i].center = (x, y)
                car_texts[i].set_position((x, y))
            else:
                # Done - hide car
                car_patches[i].set_visible(False)
                car_texts[i].set_visible(False)
                done += 1
        
        status_text.set_text(f'Waiting: {waiting} | Moving: {moving} | Done: {done}')
        
        return car_patches + car_texts + [time_text, status_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=int(max_time * 2) + 10, 
                         interval=200, blit=True, repeat=False)
    
    plt.title('Intersection Traffic Simulation', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def draw_intersection_base(ax):
    """Draw the static intersection roads"""
    road_width = 4
    road_color = '#404040'
    line_color = 'yellow'
    
    # Horizontal road (East-West)
    ax.add_patch(patches.Rectangle((-15, -road_width/2), 30, road_width, 
                                   fc=road_color, ec='none'))
    
    # Vertical road (North-South)
    ax.add_patch(patches.Rectangle((-road_width/2, -15), road_width, 30, 
                                   fc=road_color, ec='none'))
    
    # Center lines
    ax.plot([-15, -road_width/2], [0, 0], 'y--', linewidth=1, alpha=0.5)
    ax.plot([road_width/2, 15], [0, 0], 'y--', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [-15, -road_width/2], 'y--', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [road_width/2, 15], 'y--', linewidth=1, alpha=0.5)
    
    # Intersection box
    ax.add_patch(patches.Rectangle((-road_width/2, -road_width/2), 
                                   road_width, road_width, 
                                   fc='#505050', ec='white', linewidth=2))
    
    # Labels
    ax.text(0, 14, 'NORTH', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(0, -14, 'SOUTH', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(14, 0, 'EAST', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(-14, 0, 'WEST', ha='center', fontsize=12, fontweight='bold', color='white')
    
    # Legend
    legend_y = 11
    ax.text(-11, legend_y, '●', color='blue', fontsize=20)
    ax.text(-10, legend_y, 'North', fontsize=9, va='center')
    ax.text(-6, legend_y, '●', color='red', fontsize=20)
    ax.text(-5, legend_y, 'South', fontsize=9, va='center')
    ax.text(-1, legend_y, '●', color='green', fontsize=20)
    ax.text(0, legend_y, 'East', fontsize=9, va='center')
    ax.text(4, legend_y, '●', color='orange', fontsize=20)
    ax.text(5, legend_y, 'West', fontsize=9, va='center')


CONFLICT_PAIRS = {
    ('S_R', 'W_S'), ('S_R', 'N_L'),
    ('S_S', 'E_R'), ('S_S', 'E_S'), ('S_S', 'E_L'), ('S_S', 'N_L'), ('S_S', 'W_S'), ('S_S', 'W_L'),
    ('S_L', 'W_L'), ('S_L', 'W_S'), ('S_L', 'N_R'), ('S_L', 'N_L'), ('S_L', 'N_S'), ('S_L', 'E_S'), ('S_L', 'E_L'),
    ('N_R', 'E_S'), ('N_R', 'S_L'),
    ('N_S', 'W_S'), ('N_S', 'W_R'), ('N_S', 'W_L'), ('N_S', 'S_L'), ('N_S', 'E_S'), ('N_S', 'E_L'),
    ('N_L', 'E_L'), ('N_L', 'E_S'), ('N_L', 'S_R'), ('N_L', 'S_L'), ('N_L', 'S_S'), ('N_L', 'W_S'), ('N_L', 'W_L'),
    ('E_R', 'S_S'), ('E_R', 'W_L'),
    ('E_S', 'S_S'), ('E_S', 'S_L'), ('E_S', 'W_L'), ('E_S', 'N_S'), ('E_S', 'N_L'), ('E_S', 'N_R'),
    ('E_L', 'N_L'), ('E_L', 'N_S'), ('E_L', 'W_R'), ('E_L', 'W_L'),('E_L', 'W_S'), ('E_L', 'S_S'), ('E_L', 'S_L'),
    ('W_R', 'N_S'), ('W_R', 'E_L'),
    ('W_S', 'N_S'), ('W_S', 'N_L'), ('W_S', 'E_L'), ('W_S', 'S_S'), ('W_S', 'S_L'), ('W_S', 'S_R'),
    ('W_L', 'S_L'), ('W_L', 'S_S'), ('W_L', 'E_R'), ('W_L', 'E_L'),('W_L', 'E_S'), ('W_L', 'N_S'), ('W_L', 'N_L'),
}

def calculate_makespan_and_wait(chromosome, cars, conflict_pairs):
    start_times = []
    finish_times = []
    lane_count = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
    sequence = []

    for lane in chromosome:
        lane_count[lane] += 1
        car_index = lane_count[lane]
        for car in cars:
            if car['lane'] == lane and car['i'] == car_index:
                sequence.append(car)
                break

    for i, car_i in enumerate(sequence):
        start_time = 0
        car_i_tag = f"{car_i['lane']}_{car_i['dir']}"

        for j in range(i):
            car_j = sequence[j]   # FIXED: define before using

            # Same lane blocking
            if car_i['lane'] == car_j['lane']:
                start_time = max(start_time, finish_times[j])

            # Conflict blocking
            car_j_tag = f"{car_j['lane']}_{car_j['dir']}"
            if (car_i_tag, car_j_tag) in conflict_pairs or (car_j_tag, car_i_tag) in conflict_pairs:
                start_time = max(start_time, finish_times[j])

        start_times.append(start_time)
        finish_times.append(start_time + car_i['tc'])

    makespan = max(finish_times)
    total_waiting_time = sum(start_times)
    return makespan, total_waiting_time, start_times, finish_times

def fitness_function(chromosome, cars):
    makespan, wait_time, _, _ = calculate_makespan_and_wait(chromosome, cars, CONFLICT_PAIRS)
    fitness = 0.5 * makespan + 0.5 * wait_time
    return fitness

def create_initial_population(cars, population_size):
    base_chromosome = [car['lane'] for car in cars]
    seen = set()        
    population = []     
    while len(population) < population_size:
        chromosome = tuple(random.sample(base_chromosome, len(base_chromosome)))
        if chromosome not in seen:
            seen.add(chromosome)          
            population.append(list(chromosome))
    return population

def davis_order_crossover(p1, p2):
    split_index = 3 
    #child 1
    prefix1 = p1[:split_index]
    needed1 = list(p1)
    for elem in prefix1:
        needed1.remove(elem)
    #start from split_index in p2 then wrap around
    remaining1 = []
    for i in range(len(p2)):
        idx = (split_index + i) % len(p2)
        if p2[idx] in needed1:
            remaining1.append(p2[idx])
            needed1.remove(p2[idx])
    child1 = prefix1 + remaining1
    #child 2
    prefix2 = p2[:split_index]
    needed2 = list(p2)
    for elem in prefix2:
        needed2.remove(elem)
    #start from split_index in p1, wrap around
    remaining2 = []
    for i in range(len(p1)):
        idx = (split_index + i) % len(p1)
        if p1[idx] in needed2:
            remaining2.append(p1[idx])
            needed2.remove(p1[idx])
    child2 = prefix2 + remaining2
    return child1, child2

def GA(cars):
    population_size = 5
    max_generations = 5
    elitism_rate=0.2
    crossover_rate=0.6
    mutation_rate=0.2
    
    # Track convergence - initialize with generation 0
    best_fitness_history = []
    avg_fitness_history = []
    worst_fitness_history = []
    
    #if there's only 1 car
    if len(cars) == 1:
        print("The car in lane: ", cars[0]['lane'], " will start moving at t = ", 0, " and finish at t = ", cars[0]['tc'])
        print("This will have a makespan of: ", cars[0]['tc'], " and a total wait time of: ", 0)  
        return
    
    #if all the cars are in the same lane
    lanes=[]
    for i in range(len(cars)):
        lanes.append(f"{cars[i]['lane']}")
    flag = False
    for i in range(len(lanes)):
        if i < len(lanes)-1:
            if cars[i]['lane'] != cars[i+1]['lane']:
                flag = True
    if flag == False:
        makespanx0, waitTimex0, startTimesx0, finishTimesx0 = calculate_makespan_and_wait(lanes, cars, CONFLICT_PAIRS)
        print("START TIMES: ", startTimesx0)
        for i in range(len(lanes)):
            print("The car in lane: ", lanes[i], " will start moving at t = ", startTimesx0[i], " and finish at t = ", finishTimesx0[i])
        print("This will have a makespan of: ", makespanx0, " and a total wait time of: ", waitTimex0)
        return {
    'best_chromosome': lanes,
    'best_fitness': fitness_function(lanes, cars),
    'makespan': makespanx0,
    'wait_time': waitTimex0,
    'start_times': startTimesx0,
    'finish_times': finishTimesx0
}
 
    
    #initial population
    population = create_initial_population(cars, population_size)
    print("Initial population: ")
    for i in range(len(population)):
        print(population[i])
    
    # Track initial population (generation 0)
    initial_fitness_scores = [fitness_function(chrom, cars) for chrom in population]
    best_fitness_history.append(min(initial_fitness_scores))
    avg_fitness_history.append(sum(initial_fitness_scores) / len(initial_fitness_scores))
    worst_fitness_history.append(max(initial_fitness_scores))
    
    print("Initial population fitness: ")
    for i in range(len(initial_fitness_scores)):
        print(initial_fitness_scores[i])

    for generation in range(max_generations):
        print(f"\n================ GENERATION {generation + 1} ====================")

        # --- Calculate fitness for current population ---
        fitness_scores = [fitness_function(chrom, cars) for chrom in population]
        print("Current population and fitness:")
        for i in range(len(population)):
            print(population[i], "Fitness:", fitness_scores[i])

        # Track convergence metrics
        best_fitness = min(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        worst_fitness = max(fitness_scores)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        worst_fitness_history.append(worst_fitness)

        # --- ELITISM ---
        elite_count = int(population_size * elitism_rate)
        new_population = []
        temp_population = population.copy()
        temp_fitness = fitness_scores.copy()
        while len(new_population) < elite_count:
            min_index = temp_fitness.index(min(temp_fitness))
            new_population.append(temp_population[min_index])
            temp_population.pop(min_index)
            temp_fitness.pop(min_index)
        print("Elites:")
        for i in range(len(new_population)):
            print(new_population[i])

        # --- CROSSOVER ---
        crossover_count = int(population_size * crossover_rate)
        # sort by ascending fitness so index 0 = best, 1 = next best, etc.
        sorted_pairs = sorted(zip(fitness_scores, population), key=lambda x: x[0])
        temp_fitness, temp_population = zip(*sorted_pairs)
        temp_population = list(temp_population)
        temp_fitness = list(temp_fitness)
        best_parent_index = temp_fitness.index(min(temp_fitness))
        best_parent = temp_population[best_parent_index]
        temp_population.pop(best_parent_index)
        temp_fitness.pop(best_parent_index)

        pair_index = 0  # to track which parent we're crossing with
        while (len(new_population) - elite_count) < crossover_count and pair_index < len(temp_population):
            next_parent = temp_population[pair_index]
            pair_index += 1

            # Perform Davis order crossover (elite × next parent)
            child1, child2 = davis_order_crossover(best_parent, next_parent)

            # Avoid duplicates
            child1_is_duplicate = child1 in new_population
            child2_is_duplicate = child2 in new_population

            # Add based on fitness or order
            if not child1_is_duplicate and not child2_is_duplicate:
                fitness1 = fitness_function(child1, cars)
                fitness2 = fitness_function(child2, cars)
                if fitness1 <= fitness2:
                    new_population.append(child1)
                    if (len(new_population) - elite_count) < crossover_count:
                        new_population.append(child2)
                else:
                    new_population.append(child2)
                    if (len(new_population) - elite_count) < crossover_count:
                        new_population.append(child1)
            elif not child1_is_duplicate:
                new_population.append(child1)
            elif not child2_is_duplicate:
                new_population.append(child2)

        print("After crossover:")
        for i in range(len(new_population)):
            print(new_population[i])

        # --- MUTATION ---
        mutation_count = int(population_size * mutation_rate)
        temp_population = population.copy()
        temp_fitness = fitness_scores.copy()
        while (len(new_population) - elite_count - crossover_count) < mutation_count:
            worst_parent_index = temp_fitness.index(max(temp_fitness))
            worst_parent = temp_population[worst_parent_index].copy()
            x = random.randint(0, len(worst_parent) - 1)
            y = x
            while x == y or worst_parent[x] == worst_parent[y]:
                y = random.randint(0, len(worst_parent) - 1)
            print("Mutation switches indices", x, "and", y, "in parent", worst_parent)
            worst_parent[x], worst_parent[y] = worst_parent[y], worst_parent[x]
            if worst_parent not in new_population:
                new_population.append(worst_parent)
                temp_population.pop(worst_parent_index)
                temp_fitness.pop(worst_parent_index)

        print("After mutation:")
        for i in range(len(new_population)):
            print(new_population[i])

        # --- Evaluate new generation fitness ---
        new_fitness_scores = [fitness_function(ch, cars) for ch in new_population]
        print("New fitness values:")
        for i in range(len(new_population)):
            print(new_population[i], "Fitness:", new_fitness_scores[i])

        # --- Prepare for next generation ---
        population = new_population.copy()

        # Track best chromosome in current generation
        best_fit = min(new_fitness_scores)
        best_index = new_fitness_scores.index(best_fit)
        best_chromosome = new_population[best_index]

        print(f"Best chromosome of generation {generation + 1}: {best_chromosome}, Fitness: {best_fit}")
        print("==============================================================")

    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    # generations list should include generation 0 and all subsequent generations
    generations = list(range(0, len(best_fitness_history)))
    
    plt.plot(generations, best_fitness_history, 'b-', linewidth=2, label='Best Fitness', marker='o')
    plt.plot(generations, avg_fitness_history, 'g-', linewidth=2, label='Average Fitness', marker='s')
    plt.plot(generations, worst_fitness_history, 'r-', linewidth=2, label='Worst Fitness', marker='^')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('GA Convergence Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(generations)  # Show all generation numbers on x-axis
    plt.tight_layout()
    plt.show()

    return {
        'best_chromosome': best_chromosome,
        'best_fitness': best_fit,
        'makespan': calculate_makespan_and_wait(best_chromosome, cars, CONFLICT_PAIRS)[0],
        'wait_time': calculate_makespan_and_wait(best_chromosome, cars, CONFLICT_PAIRS)[1],
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'worst_fitness_history': worst_fitness_history
    }

if __name__ == "__main__":

    cars1 = [
        {'lane': 'S', 'dir': 'L', 'tc': 5, 'i': 2},
        {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 2},
        {'lane': 'S', 'dir': 'R', 'tc': 2, 'i': 1},
        {'lane': 'N', 'dir': 'L', 'tc': 5, 'i': 1},
        {'lane': 'E', 'dir': 'S', 'tc': 6, 'i': 1},
        {'lane': 'W', 'dir': 'L', 'tc': 4, 'i': 1}
    ]
    cars4 = [
        {'lane': 'N', 'dir': 'R', 'tc': 5, 'i': 2},
        {'lane': 'N', 'dir': 'L', 'tc': 4, 'i': 1},
        {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 3},
        {'lane': 'S', 'dir': 'R', 'tc': 5, 'i': 2},
        {'lane': 'S', 'dir': 'L', 'tc': 4, 'i': 1},
        {'lane': 'S', 'dir': 'S', 'tc': 3, 'i': 3},
        {'lane': 'W', 'dir': 'R', 'tc': 5, 'i': 2},
        {'lane': 'W', 'dir': 'L', 'tc': 4, 'i': 1},
        {'lane': 'W', 'dir': 'S', 'tc': 3, 'i': 3},
        {'lane': 'E', 'dir': 'R', 'tc': 5, 'i': 2},
        {'lane': 'E', 'dir': 'L', 'tc': 4, 'i': 1},
        {'lane': 'E', 'dir': 'S', 'tc': 3, 'i': 3},    
    ]
    
    cars5 = [
        {'lane': 'N', 'dir': 'S', 'tc': 5, 'i': 4},
        {'lane': 'N', 'dir': 'S', 'tc': 4, 'i': 1},
        {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 3},
        {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 2}
    ]
    result = GA(cars5)
    best_chrom = result['best_chromosome']
    start_times = result['start_times'] if 'start_times' in result else calculate_makespan_and_wait(best_chrom, cars5, CONFLICT_PAIRS)[2]
    finish_times = result['finish_times'] if 'finish_times' in result else calculate_makespan_and_wait(best_chrom, cars5, CONFLICT_PAIRS)[3]
    makespan = result['makespan']

    visualize_intersection_graphical(cars5, best_chrom, start_times, finish_times, makespan)