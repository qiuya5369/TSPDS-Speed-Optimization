import time
import os
from solver import DroneDeliverySolver
import parameters as Pa


if __name__ == "__main__":
    # Configuration 1: instance file settings
    EXAMPLE_DIR = "example"

    # Generate instance file names automatically
    n_values = [2]
    EXAMPLE_FILES = []
    for n in n_values:
        first_part = 5 * n
        for i in range(1, 3):
            EXAMPLE_FILES.append(f"{first_part}_{n}_{i}.txt")

    # Configuration 2: number of runs per instance
    RUNS_PER_INSTANCE = 1

    # Configuration 3: display settings
    SHOW_DETAILED_PROGRESS = False

    all_instance_results = []

    # Main loop over all instance files
    for instance_idx, filename in enumerate(EXAMPLE_FILES, 1):
        filepath = f"{EXAMPLE_DIR}/{filename}"

        # Skip missing files
        if not os.path.exists(filepath):
            continue

        instance_results = []

        # Multiple runs for the same instance
        for run in range(1, RUNS_PER_INSTANCE + 1):
            start_time = time.perf_counter()

            solver = DroneDeliverySolver(
                filepath,
                seed=Pa.RANDOM_SEED + run,
                show_progress=SHOW_DETAILED_PROGRESS
            )
            final_solution = solver.solve()

            elapsed = time.perf_counter() - start_time

            instance_results.append({
                'instance': filename,
                'run': run,
                'total_cost': final_solution.total_cost,
                'time': elapsed
            })

        all_instance_results.extend(instance_results)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    instance_summary = {}
    for result in all_instance_results:
        instance_name = result['instance']
        if instance_name not in instance_summary:
            instance_summary[instance_name] = []
        instance_summary[instance_name].append(result)

    print(f"\nTotal instances: {len(instance_summary)}")
    print(f"Total runs: {len(all_instance_results)}\n")

    for instance_name, results in instance_summary.items():
        costs = [r['total_cost'] for r in results]
        times = [r['time'] for r in results]

        best_cost = min(costs)
        best_run = costs.index(best_cost) + 1
        worst_cost = max(costs)
        worst_run = costs.index(worst_cost) + 1
        avg_cost = sum(costs) / len(costs)
        avg_time = sum(times) / len(times)

        print(f"{instance_name}:")
        print(f"  Best cost:  {best_cost:.6f} (run {best_run})")
        print(f"  Worst cost: {worst_cost:.6f} (run {worst_run})")
        print(f"  Avg cost:   {avg_cost:.6f}")
        print(f"  Avg time:   {avg_time:.2f}s")
        print()
