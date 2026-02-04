import time
import os
from solver import DroneDeliverySolver
import parameters as Pa

if __name__ == "__main__":
    # Instance file directory
    EXAMPLE_DIR = "example"

    # Instance filenames
    n_values = [2]
    EXAMPLE_FILES = []
    for n in n_values:
        first_part = 5 * n
        for i in range(1, 2):
            EXAMPLE_FILES.append(f"{first_part}_{n}_{i}.txt")

    # Number of runs per instance
    RUNS_PER_INSTANCE = 2

    # Whether to print detailed iteration logs
    SHOW_DETAILED_PROGRESS = False

    all_instance_results = []

    # Loop over instances
    for instance_idx, filename in enumerate(EXAMPLE_FILES, 1):
        filepath = f"{EXAMPLE_DIR}/{filename}"

        if not os.path.exists(filepath):
            continue

        instance_results = []

        # Multiple runs for each instance
        for run in range(1, RUNS_PER_INSTANCE + 1):
            start_time = time.perf_counter()
            solver = DroneDeliverySolver(
                filepath,
                seed=Pa.RANDOM_SEED + run,
                show_progress=SHOW_DETAILED_PROGRESS,
            )
            final_solution = solver.solve()
            elapsed = time.perf_counter() - start_time

            instance_results.append(
                {
                    "instance": filename,
                    "run": run,
                    "total_time": final_solution.total_time,
                    "solve_time": elapsed,
                }
            )

        all_instance_results.extend(instance_results)

    # Summary statistics over all runs
    print(f"\n{'=' * 60}")
    print("Results Summary (Min Time Model)")
    print("=" * 60)

    instance_summary = {}
    for result in all_instance_results:
        instance_name = result["instance"]
        if instance_name not in instance_summary:
            instance_summary[instance_name] = []
        instance_summary[instance_name].append(result)

    print(f"\nTotal instances: {len(instance_summary)}")
    print(f"Total runs: {len(all_instance_results)}\n")

    for instance_name, results in instance_summary.items():
        times = [r["total_time"] for r in results]
        solve_times = [r["solve_time"] for r in results]
        best_time = min(times)
        best_run = times.index(best_time) + 1
        worst_time = max(times)
        worst_run = times.index(worst_time) + 1
        avg_time = sum(times) / len(times)
        avg_solve_time = sum(solve_times) / len(solve_times)

        print(f"{instance_name}:")
        print(f"  Best time:   {best_time:.6f}s (run {best_run})")
        print(f"  Worst time:  {worst_time:.6f}s (run {worst_run})")
        print(f"  Avg time:    {avg_time:.6f}s")
        print(f"  Avg solve:   {avg_solve_time:.2f}s")
        print()
