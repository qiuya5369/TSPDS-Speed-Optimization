import math
import random
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import MDS
import numpy as np
import parameters as Pa
from sklearn.cluster import KMeans
from collections import defaultdict
from core import (
    Solution,
    PhysicsCalculator,
    RouteManager,
    SpeedOptimizer,
    SolutionEvaluator,
)
from operators import DestroyOperator, RepairOperator, LocalSearch


class OperatorStatistics:
    """Count how often each destroy/repair/local-search operator is used."""

    def __init__(self):
        self.destroy_count = {
            "random_remove": 0,
            "worst_remove": 0,
        }
        self.repair_count = {
            "regret": 0,
            "greedy": 0,
            "global": 0,
        }
        self.local_search_count = {
            "two_opt": 0,
            "service_mode_switch": 0,
            "station_swap": 0,
        }
        self.total_iterations = 0
        self.total_local_search_calls = 0

    def record_destroy(self, operator_name):
        if operator_name in self.destroy_count:
            self.destroy_count[operator_name] += 1

    def record_repair(self, operator_name):
        if operator_name in self.repair_count:
            self.repair_count[operator_name] += 1

    def record_local_search(self, operator_name):
        if operator_name in self.local_search_count:
            self.local_search_count[operator_name] += 1

    def increment_iteration(self):
        self.total_iterations += 1

    def get_statistics(self):
        """Return simple usage stats for all operators."""
        stats = {
            "destroy_operators": {},
            "repair_operators": {},
            "local_search_operators": {},
            "total_iterations": self.total_iterations,
            "total_local_search_calls": self.total_local_search_calls,
        }

        total_destroy = sum(self.destroy_count.values())
        for op, count in self.destroy_count.items():
            usage_rate = (count / total_destroy * 100) if total_destroy > 0 else 0.0
            stats["destroy_operators"][op] = {
                "count": count,
                "usage_rate": usage_rate,
            }

        total_repair = sum(self.repair_count.values())
        for op, count in self.repair_count.items():
            usage_rate = (count / total_repair * 100) if total_repair > 0 else 0.0
            stats["repair_operators"][op] = {
                "count": count,
                "usage_rate": usage_rate,
            }

        total_local = sum(self.local_search_count.values())
        for op, count in self.local_search_count.items():
            usage_rate = (count / total_local * 100) if total_local > 0 else 0.0
            stats["local_search_operators"][op] = {
                "count": count,
                "usage_rate": usage_rate,
            }

        return stats

    def print_statistics(self):
        """Print a summary of operator usage."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("Operator usage summary")
        print("=" * 60)

        print("Destroy operators")
        print("-" * 60)
        for op_name, data in stats["destroy_operators"].items():
            print(f"{op_name:<25} {data['count']:>12} {data['usage_rate']:>11.2f}%")

        print("\nRepair operators")
        print("-" * 60)
        for op_name, data in stats["repair_operators"].items():
            print(f"{op_name:<25} {data['count']:>12} {data['usage_rate']:>11.2f}%")

        print("\nLocal search operators")
        print("-" * 60)
        for op_name, data in stats["local_search_operators"].items():
            print(f"{op_name:<25} {data['count']:>12} {data['usage_rate']:>11.2f}%")

        print("=" * 60 + "\n")

        all_used = True
        unused_operators = []

        for op_name, data in stats["destroy_operators"].items():
            if data["count"] == 0:
                all_used = False
                unused_operators.append(f"Destroy:{op_name}")

        for op_name, data in stats["repair_operators"].items():
            if data["count"] == 0:
                all_used = False
                unused_operators.append(f"Repair:{op_name}")

        for op_name, data in stats["local_search_operators"].items():
            if data["count"] == 0:
                all_used = False
                unused_operators.append(f"LocalSearch:{op_name}")

        if not all_used:
            print("Unused operators:")
            for op in unused_operators:
                print(f"   {op}")
        print()


def readFile(Infile):
    """Read one instance file and return basic data structures."""
    aline = Infile.readline()
    values = aline.split()
    CustNum = int(values[1])
    StaNum = int(values[3])
    TotNodeNum = CustNum + StaNum + 2

    DistDrone = [[0.0 for _ in range(TotNodeNum)] for _ in range(TotNodeNum)]
    DistTruck = [[0.0 for _ in range(TotNodeNum)] for _ in range(TotNodeNum)]

    for i in range(TotNodeNum):
        aline = Infile.readline()
        values = aline.split()
        for j in range(TotNodeNum):
            DistDrone[i][j] = float(values[j])
            DistTruck[i][j] = round(Pa.DistFactor * DistDrone[i][j], 6)

    CustDemand = [0.0 for _ in range(CustNum)]
    Infile.readline()
    aline = Infile.readline()
    values = aline.split()
    for i in range(CustNum):
        CustDemand[i] = float(values[i])

    Infile.close()
    return CustNum, StaNum, DistDrone, DistTruck, CustDemand


class ResultVisualizer:
    """Basic printing and plotting utilities for solutions."""

    @staticmethod
    def plot(solution, problem, save_path=None):
        """Plot truck routes and drone missions in a 2D layout."""
        truck_route = solution.truck_route
        active_stations = solution.active_stations
        drone_assignments = solution.drone_assignments
        all_stations = problem.valid_stations

        G = nx.DiGraph()
        all_nodes = (
            set(truck_route)
            | set(all_stations)
            | {problem.depot_start, problem.depot_end}
        )
        G.add_nodes_from(all_nodes)

        truck_edges = [
            (truck_route[i], truck_route[i + 1])
            for i in range(len(truck_route) - 1)
        ]
        for u, v in truck_edges:
            G.add_edge(u, v, color="blue", label="Truck")

        drone_edges = []
        for (c, s) in drone_assignments:
            drone_edges.append((s, c))
            drone_edges.append((c, s))
        for u, v in drone_edges:
            G.add_edge(u, v, color="red", label="Drone", style="dashed")

        dist_matrix = np.array(problem.DistDrone)
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            normalized_stress="auto",
            random_state=42,
        )
        pos_array = mds.fit_transform(dist_matrix)
        pos = {i: pos_array[i] for i in range(problem.TotNodeNum)}

        node_colors = []
        node_shapes = []
        for node in G.nodes():
            if node == problem.depot_start or node == problem.depot_end:
                node_colors.append("green")
                node_shapes.append("s")
            elif node in active_stations:
                node_colors.append("orange")
                node_shapes.append("o")
            elif node in problem.valid_stations:
                node_colors.append("darkgray")
                node_shapes.append("o")
            else:
                node_colors.append("lightblue")
                node_shapes.append("d")

        plt.figure(figsize=(12, 8))
        for i, node in enumerate(G.nodes()):
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_size=300,
                node_color=node_colors[i],
                node_shape=node_shapes[i],
                edgecolors="black",
                linewidths=0.8,
            )

        truck_edges = [
            e for e in G.edges() if G[e[0]][e[1]].get("label") == "Truck"
        ]
        drone_edges = [
            e for e in G.edges() if G[e[0]][e[1]].get("label") == "Drone"
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=truck_edges,
            edge_color="blue",
            width=2,
            arrowstyle="-|>",
            arrowsize=20,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=drone_edges,
            edge_color="red",
            width=1.5,
            style="dashed",
            arrowstyle="-|>",
            arrowsize=15,
        )

        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

        legend_elements = [
            plt.Line2D([0], [0], color="blue", lw=3, label="Truck Route"),
            plt.Line2D(
                [0],
                [0],
                color="red",
                lw=3,
                linestyle="dashed",
                label="Drone Mission",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Depot",
                markerfacecolor="green",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Active Station",
                markerfacecolor="orange",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Inactive Station",
                markerfacecolor="lightgray",
                markersize=8,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="d",
                color="w",
                label="Customer",
                markerfacecolor="lightblue",
                markersize=10,
            ),
        ]
        plt.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=9,
            title="Legend:",
            title_fontsize=10,
            frameon=True,
            framealpha=0.8,
        )

        plt.title(
            "Truck and Drone Delivery Routes\nCompletion Time: {:.2f}s | Cost: ${:.2f}".format(
                solution.total_time, solution.total_cost
            ),
            fontsize=12,
            pad=20,
        )
        plt.axis("off")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            if hasattr(problem, "show_progress") and problem.show_progress:
                print(f"Visualization saved to: {save_path}")
        plt.show()


class DroneDeliverySolver:
    """
    ALNS based solver for the min time variant.

    The objective is to minimize completion time, using:
      * initial clustering based construction
      * destroy/repair operators
      * local search and simulated annealing acceptance
    """

    def __init__(self, filepath, seed=42, show_progress=True):
        with open(filepath, "r") as f:
            (
                self.CustNum,
                self.StaNum,
                self.DistDrone,
                self.DistTruck,
                self.CustDemand,
            ) = readFile(f)

        self.TotNodeNum = self.CustNum + self.StaNum + 2
        self.depot_start = 0
        self.depot_end = self.CustNum + self.StaNum + 1
        self.valid_stations = list(
            range(self.CustNum + 1, self.CustNum + self.StaNum + 1)
        )

        self.speed_levels = self._generate_speed_levels()
        random.seed(seed if seed else Pa.RANDOM_SEED)

        self.MAX_ITER = Pa.MAX_ITER
        self.DESTROY_RATE = Pa.DESTROY_RATE
        self.best_solution = None
        self.NO_IMPROVE_ITER = Pa.NO_IMPROVE_ITER

        self.repair_operators = ["regret", "greedy", "global"]
        self.operator_weights = {
            "regret": Pa.INITIAL_REGRET_WEIGHT,
            "greedy": Pa.INITIAL_GREEDY_WEIGHT,
            "global": Pa.INITIAL_GLOBAL_WEIGHT,
        }
        self.operator_scores = {
            "regret": Pa.INITIAL_OPERATOR_SCORE,
            "greedy": Pa.INITIAL_OPERATOR_SCORE,
            "global": Pa.INITIAL_OPERATOR_SCORE,
        }

        self.destroy_operators = ["worst_remove", "random_remove"]
        self.destroy_weights = {"worst_remove": 0.7, "random_remove": 0.3}
        self.destroy_scores = {
            "worst_remove": Pa.INITIAL_OPERATOR_SCORE,
            "random_remove": Pa.INITIAL_OPERATOR_SCORE,
        }

        self.WEIGHT_UPDATE_FREQ = Pa.WEIGHT_UPDATE_FREQ
        self.ELITE_POOL_SIZE = Pa.ELITE_POOL_SIZE
        self.elite_pool = []

        self.reachability = self._precompute_reachability()
        self.show_progress = show_progress
        LocalSearch.set_show_progress(show_progress)

        self.operator_stats = OperatorStatistics()
        self.initial_solution = None

    def _precompute_reachability(self):
        """Precompute which station can serve which customer by drone."""
        reachability = defaultdict(dict)
        for s in self.valid_stations:
            for c in range(1, self.CustNum + 1):
                demand = self.CustDemand[c - 1]
                dist = self.DistDrone[s][c]
                reachability[s][c] = PhysicsCalculator.is_reachable_static(
                    demand, dist
                )
        return reachability

    def _generate_speed_levels(self):
        """Generate discretized speed levels for the drone."""
        if Pa.MinSpeed == Pa.MaxSpeed:
            return [Pa.MinSpeed] * Pa.SpeedLevel
        interval = (Pa.MaxSpeed - Pa.MinSpeed) / Pa.SpeedLevel
        return [round(Pa.MinSpeed + (i + 0.5) * interval, 1) for i in range(Pa.SpeedLevel)]

    def construct_initial_solution(self):
        """
        Build an initial solution for the min time objective.

        Steps:
          1) Cluster customers in distance space.
          2) Select one station per cluster by average flight time.
          3) Assign customers to drones balancing time load.
          4) Build a truck route with time based insertion and 2 opt.
        """
        sol = Solution()
        all_customers = set(range(1, self.CustNum + 1))

        customer_indices = list(all_customers)
        sub_dist = [
            [self.DistDrone[i][j] for j in customer_indices]
            for i in customer_indices
        ]

        n_clusters = min(
            math.ceil(self.CustNum / (Pa.StaDroNum * 1.5)),
            len(self.valid_stations),
        )

        if len(customer_indices) <= n_clusters:
            labels = list(range(len(customer_indices)))
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=Pa.RANDOM_SEED,
                n_init=10,
            )
            try:
                kmeans.fit(sub_dist)
                labels = kmeans.labels_
            except Exception:
                labels = [i % n_clusters for i in range(len(customer_indices))]

        cluster_map = defaultdict(list)
        for idx, label in enumerate(labels):
            customer = customer_indices[idx]
            cluster_map[label].append(customer)

        activated_stations = set()
        cluster_station = {}

        for label, customers in cluster_map.items():
            candidate_stations = []
            for s in self.valid_stations:
                if s in activated_stations:
                    continue

                total_flight_time = 0.0
                reachable_count = 0

                for c in customers:
                    if not self.reachability[s][c]:
                        continue
                    demand = self.CustDemand[c - 1]
                    dist = self.DistDrone[s][c]
                    speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, self)
                    if speed_pair:
                        v_out, v_back = speed_pair
                        vertical_time = (
                            2
                            * (
                                Pa.FlightHeight / Pa.vSpeedUp
                                + Pa.FlightHeight / Pa.vSpeedDown
                            )
                        )
                        horizontal_time = dist / v_out + dist / v_back
                        flight_time = vertical_time + horizontal_time
                        total_flight_time += flight_time
                        reachable_count += 1

                if reachable_count == 0:
                    continue

                avg_flight_time = total_flight_time / reachable_count
                candidate_stations.append((s, avg_flight_time))

            if candidate_stations:
                best_station, best_val = min(
                    candidate_stations, key=lambda x: x[1]
                )
                activated_stations.add(best_station)
                cluster_station[label] = best_station
                if self.show_progress:
                    print(
                        f"Cluster {label}: station {best_station} activated "
                        f"(avg flight time {best_val:.2f}s)"
                    )

        sol.active_stations = activated_stations

        assigned_customers = set()

        for label, customers in cluster_map.items():
            station = cluster_station.get(label)
            if station is None:
                continue

            for c in customers:
                if not self.reachability[station][c]:
                    continue

                used_drones = sum(
                    1 for (cust, sta) in sol.drone_assignments if sta == station
                )
                if used_drones >= Pa.StaDroNum:
                    continue

                demand = self.CustDemand[c - 1]
                dist = self.DistDrone[station][c]
                speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, self)
                if speed_pair:
                    drone_time_loads = {i: 0.0 for i in range(Pa.StaDroNum)}

                    for (cust, sta), (did, v_out, v_back) in sol.drone_assignments.items():
                        if sta == station:
                            d = self.DistDrone[sta][cust]
                            vertical_time = (
                                2
                                * (
                                    Pa.FlightHeight / Pa.vSpeedUp
                                    + Pa.FlightHeight / Pa.vSpeedDown
                                )
                            )
                            horizontal_time = d / v_out + d / v_back
                            task_time = vertical_time + horizontal_time
                            drone_time_loads[did] += task_time

                    drone_id = min(
                        drone_time_loads.keys(),
                        key=lambda d_id: drone_time_loads[d_id],
                    )

                    sol.drone_assignments[(c, station)] = (
                        drone_id,
                        speed_pair[0],
                        speed_pair[1],
                    )
                    assigned_customers.add(c)

        for c in all_customers - assigned_customers:
            candidate_stations = []
            for s in activated_stations:
                if not self.reachability[s][c]:
                    continue

                used_drones = sum(
                    1 for (cust, sta) in sol.drone_assignments if sta == s
                )
                if used_drones < Pa.StaDroNum:
                    dist = self.DistDrone[s][c]
                    demand = self.CustDemand[c - 1]
                    speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, self)
                    if speed_pair:
                        v_out, v_back = speed_pair
                        vertical_time = (
                            2
                            * (
                                Pa.FlightHeight / Pa.vSpeedUp
                                + Pa.FlightHeight / Pa.vSpeedDown
                            )
                        )
                        horizontal_time = dist / v_out + dist / v_back
                        flight_time = vertical_time + horizontal_time
                        candidate_stations.append((s, flight_time, speed_pair))

            if not candidate_stations:
                continue

            best_station, best_time, speed_pair = min(
                candidate_stations, key=lambda x: x[1]
            )
            if speed_pair:
                drone_time_loads = {i: 0.0 for i in range(Pa.StaDroNum)}

                for (cust, sta), (did, v_out, v_back) in sol.drone_assignments.items():
                    if sta == best_station:
                        d = self.DistDrone[sta][cust]
                        vertical_time = (
                            2
                            * (
                                Pa.FlightHeight / Pa.vSpeedUp
                                + Pa.FlightHeight / Pa.vSpeedDown
                            )
                        )
                        horizontal_time = d / v_out + d / v_back
                        task_time = vertical_time + horizontal_time
                        drone_time_loads[did] += task_time

                drone_id = min(
                    drone_time_loads.keys(),
                    key=lambda d_id: drone_time_loads[d_id],
                )

                sol.drone_assignments[(c, best_station)] = (
                    drone_id,
                    speed_pair[0],
                    speed_pair[1],
                )
                assigned_customers.add(c)

        truck_route = [self.depot_start]

        for station in activated_stations:
            best_pos = RouteManager.find_best_insertion_time(
                truck_route, station, self.DistTruck, Pa.TruckSpeed
            )
            truck_route.insert(best_pos, station)

        unserved_customers = sorted(all_customers - assigned_customers)
        for c in unserved_customers:
            best_pos = RouteManager.find_best_insertion_time(
                truck_route, c, self.DistTruck, Pa.TruckSpeed
            )
            truck_route.insert(best_pos, c)
            prev = truck_route[best_pos - 1]
            next_node = (
                truck_route[best_pos + 1]
                if best_pos + 1 < len(truck_route)
                else self.depot_end
            )
            if best_pos + 1 >= len(truck_route):
                truck_route.append(self.depot_end)

        if truck_route[-1] != self.depot_end:
            truck_route.append(self.depot_end)

        sol.truck_route = truck_route

        original_time = (
            sum(
                self.DistTruck[a][b]
                for a, b in zip(sol.truck_route[:-1], sol.truck_route[1:])
            )
            / Pa.TruckSpeed
        )

        sol.truck_route = RouteManager.two_opt_swap(
            sol.truck_route, self.DistTruck, self.depot_end, max_trials=30
        )

        sol = SpeedOptimizer.optimize(sol, self)

        sol.truck_route = RouteManager.two_opt_swap(
            sol.truck_route, self.DistTruck, self.depot_end, max_trials=20
        )

        optimized_time = (
            sum(
                self.DistTruck[a][b]
                for a, b in zip(sol.truck_route[:-1], sol.truck_route[1:])
            )
            / Pa.TruckSpeed
        )

        if self.show_progress:
            print(
                f"Truck travel time: {original_time:.2f}s to "
                f"{optimized_time:.2f}s (improvement {original_time - optimized_time:.2f}s)"
            )

        sol = SpeedOptimizer.optimize(sol, self)
        sol.truck_route = RouteManager.standardize_route(sol.truck_route, self, sol)
        sol = SolutionEvaluator.evaluate(sol, self)
        return sol

    def _update_elite_pool(self, solution):
        """Maintain an elite pool ordered by completion time."""
        if solution.total_time < float("inf") and not any(
            sol.truck_route == solution.truck_route
            and sol.drone_assignments == solution.drone_assignments
            for sol in self.elite_pool
        ):
            self.elite_pool.append(solution.clone())
            self.elite_pool.sort(key=lambda x: x.total_time)
            if len(self.elite_pool) > self.ELITE_POOL_SIZE:
                self.elite_pool.pop()

    def solve(self):
        """
        Main ALNS search loop.

        Use adaptive destroy/repair operators, local search, and
        simulated annealing based acceptance to minimize completion time.
        """
        current_sol = self.construct_initial_solution()
        self.initial_solution = current_sol.clone()

        if self.show_progress:
            ResultVisualizer.print(current_sol)

        self.best_solution = current_sol.clone()
        last_improve_iter = 0
        temperature = 200

        for iteration in range(self.MAX_ITER):
            self.operator_stats.increment_iteration()

            destroy_choice = self._select_destroy_operator()
            destroyed = self._apply_destroy_operator(current_sol, destroy_choice)
            self.operator_stats.record_destroy(destroy_choice)

            repair_choice = self._select_repair_operator()
            repaired = self._apply_repair_operator(destroyed, repair_choice)
            self.operator_stats.record_repair(repair_choice)

            optimized = SpeedOptimizer.optimize(repaired, self)
            optimized.truck_route = RouteManager.two_opt_swap(
                optimized.truck_route,
                self.DistTruck,
                self.depot_end,
                max_trials=5,
            )
            evaluated = SolutionEvaluator.evaluate(optimized, self)

            delta_time = evaluated.total_time - current_sol.total_time
            exp_arg = -delta_time / max(temperature, 1e-6)
            exp_arg = max(-700, min(700, exp_arg))
            accept_prob = math.exp(exp_arg)
            threshold = random.random() * (1 - 0.8 * (iteration / self.MAX_ITER))

            if delta_time < 0 or accept_prob > threshold:
                current_sol = evaluated.clone()
                if current_sol.total_time < self.best_solution.total_time:
                    self.best_solution = current_sol.clone()
                    last_improve_iter = iteration

            self._update_elite_pool(current_sol)

            if iteration % 25 == 0 and iteration > 0:
                self._update_weights(current_sol, iteration)

            if (iteration - last_improve_iter) > self.NO_IMPROVE_ITER:
                if self.elite_pool:
                    restart_sol = random.choice(self.elite_pool).clone()
                    restart_sol = self._perturb_solution(restart_sol)
                    current_sol = SolutionEvaluator.evaluate(restart_sol, self)
                    last_improve_iter = iteration
                    if current_sol.total_time < self.best_solution.total_time:
                        self.best_solution = current_sol.clone()

            if iteration % Pa.LOCAL_SEARCH_FREQ == 0:
                self.operator_stats.total_local_search_calls += 1
                local_best = current_sol.clone()
                for _ in range(Pa.LOCAL_SEARCH_DEPTH):
                    temp_sol = LocalSearch.station_swap(local_best, self)
                    self.operator_stats.record_local_search("station_swap")

                    temp_sol = LocalSearch.two_opt(temp_sol, self)
                    self.operator_stats.record_local_search("two_opt")

                    switch_result = LocalSearch.service_mode_switch(
                        temp_sol, self, local_best
                    )
                    self.operator_stats.record_local_search("service_mode_switch")
                    if switch_result:
                        temp_sol = switch_result

                    if temp_sol and temp_sol.total_time < local_best.total_time:
                        local_best = temp_sol.clone()

                if local_best.total_time < self.best_solution.total_time:
                    self.best_solution = local_best.clone()

            temperature *= Pa.SA_COOLING_RATE
            if (
                iteration % 100 == 99
                and temperature
                < current_sol.total_time * Pa.SA_TEMP_RESET_THRESHOLD
            ):
                temperature = (
                    current_sol.total_time * Pa.SA_TEMP_RESET_RATIO
                )

        final_best = self.best_solution.clone()
        for _ in range(Pa.FINAL_SEARCH_DEPTH):
            temp_sol = LocalSearch.station_swap(final_best, self)
            self.operator_stats.record_local_search("station_swap")

            temp_sol = LocalSearch.two_opt(temp_sol, self)
            self.operator_stats.record_local_search("two_opt")

            switch_result = LocalSearch.service_mode_switch(
                temp_sol, self, final_best
            )
            self.operator_stats.record_local_search("service_mode_switch")
            if switch_result:
                temp_sol = switch_result

            if temp_sol and temp_sol.total_time < final_best.total_time:
                final_best = temp_sol.clone()

        if final_best.total_time < self.best_solution.total_time:
            self.best_solution = final_best.clone()

        self.best_solution = SolutionEvaluator.evaluate(self.best_solution, self)

        if self.show_progress:
            self.operator_stats.print_statistics()

        # second ALNS loop kept to preserve original structure
        for iteration in range(self.MAX_ITER):
            self.operator_stats.increment_iteration()

            destroy_op_name = self._select_destroy_operator()
            destroyed_sol = self._apply_destroy_operator(current_sol, destroy_op_name)
            self.operator_stats.record_destroy(destroy_op_name)

            repair_op_name = self._select_repair_operator()
            repaired_sol = self._apply_repair_operator(destroyed_sol, repair_op_name)
            self.operator_stats.record_repair(repair_op_name)

            repaired_sol = SpeedOptimizer.optimize(repaired_sol, self)
            repaired_sol.truck_route = RouteManager.standardize_route(
                repaired_sol.truck_route, self, repaired_sol
            )
            repaired_sol = SolutionEvaluator.evaluate(repaired_sol, self)

            time_change = repaired_sol.total_time - current_sol.total_time

            if time_change < 0:
                accept_prob = 1.0
                accept = True
            else:
                accept_prob = (
                    math.exp(-time_change / temperature) if temperature > 0 else 0.0
                )
                accept = random.random() < accept_prob

            if accept:
                if time_change < 0:
                    if repaired_sol.total_time < self.best_solution.total_time:
                        self.best_solution = repaired_sol.clone()
                        last_improve_iter = iteration
                current_sol = repaired_sol.clone()
                self._update_elite_pool(current_sol)

            if iteration % self.WEIGHT_UPDATE_FREQ == 0 and iteration > 0:
                self._update_weights(current_sol, iteration)

            if (iteration - last_improve_iter) > self.NO_IMPROVE_ITER:
                if self.elite_pool:
                    restart_sol = random.choice(self.elite_pool).clone()
                    restart_sol = self._perturb_solution(restart_sol)
                    current_sol = SolutionEvaluator.evaluate(restart_sol, self)
                    last_improve_iter = iteration
                    if current_sol.total_time < self.best_solution.total_time:
                        self.best_solution = current_sol.clone()

            if iteration % Pa.LOCAL_SEARCH_FREQ == 0:
                self.operator_stats.total_local_search_calls += 1
                local_best = current_sol.clone()
                local_start_time = local_best.total_time

                for depth in range(Pa.LOCAL_SEARCH_DEPTH):
                    temp_sol = LocalSearch.station_swap(local_best, self)
                    self.operator_stats.record_local_search("station_swap")
                    if temp_sol.total_time < local_best.total_time:
                        local_best = temp_sol.clone()

                    temp_sol = LocalSearch.two_opt(temp_sol, self)
                    self.operator_stats.record_local_search("two_opt")
                    if temp_sol.total_time < local_best.total_time:
                        local_best = temp_sol.clone()

                    switch_result = LocalSearch.service_mode_switch(
                        temp_sol, self, local_best
                    )
                    self.operator_stats.record_local_search("service_mode_switch")
                    if switch_result:
                        if switch_result.total_time < local_best.total_time:
                            local_best = switch_result.clone()
                        else:
                            temp_sol = switch_result

                    if temp_sol and temp_sol.total_time < local_best.total_time:
                        local_best = temp_sol.clone()

                local_total_improvement = local_start_time - local_best.total_time
                if local_total_improvement > 0:
                    pass

                if local_best.total_time < self.best_solution.total_time:
                    self.best_solution = local_best.clone()

            temperature *= Pa.SA_COOLING_RATE
            if (
                iteration % 100 == 99
                and temperature
                < current_sol.total_time * Pa.SA_TEMP_RESET_THRESHOLD
            ):
                temperature = (
                    current_sol.total_time * Pa.SA_TEMP_RESET_RATIO
                )

        final_best = self.best_solution.clone()
        final_start_time = final_best.total_time

        for depth in range(Pa.FINAL_SEARCH_DEPTH):
            temp_sol = LocalSearch.station_swap(final_best, self)
            self.operator_stats.record_local_search("station_swap")
            if temp_sol.total_time < final_best.total_time:
                final_best = temp_sol.clone()

            temp_sol = LocalSearch.two_opt(temp_sol, self)
            self.operator_stats.record_local_search("two_opt")
            if temp_sol.total_time < final_best.total_time:
                final_best = temp_sol.clone()

            switch_result = LocalSearch.service_mode_switch(
                temp_sol, self, final_best
            )
            self.operator_stats.record_local_search("service_mode_switch")
            if switch_result:
                if switch_result.total_time < final_best.total_time:
                    final_best = switch_result.clone()
                else:
                    temp_sol = switch_result

            if temp_sol and temp_sol.total_time < final_best.total_time:
                final_best = temp_sol.clone()

        final_total_improvement = final_start_time - final_best.total_time
        if final_total_improvement > 0:
            self.best_solution = final_best.clone()

        self.best_solution = SolutionEvaluator.evaluate(self.best_solution, self)

        if self.show_progress:
            self.operator_stats.print_statistics()

        return self.best_solution

    def _perturb_solution(self, solution):
        """Simple perturbation by closing and opening some stations."""
        new_sol = solution.clone()
        active_sta = list(new_sol.active_stations)

        if len(active_sta) > 1:
            close_num = max(1, int(len(active_sta) * 0.2))
            for s in random.sample(active_sta, close_num):
                new_sol.active_stations.discard(s)
                remove_clients = [
                    c for (c, st) in new_sol.drone_assignments if st == s
                ]
                for c in remove_clients:
                    del new_sol.drone_assignments[(c, s)]
                    RouteManager.insert_node(
                        new_sol.truck_route, c, self.DistTruck
                    )

        candidate_sta = list(set(self.valid_stations) - new_sol.active_stations)
        if candidate_sta:
            new_sta = random.choice(candidate_sta)
            RouteManager.insert_node(new_sol.truck_route, new_sta, self.DistTruck)
            new_sol.active_stations.add(new_sta)

        return SolutionEvaluator.evaluate(new_sol, self)

    def _select_repair_operator(self):
        """Sample one repair operator according to current weights."""
        total_weight = sum(self.operator_weights.values())
        probs = [
            self.operator_weights[op] / total_weight
            for op in self.repair_operators
        ]
        return random.choices(self.repair_operators, weights=probs, k=1)[0]

    def _apply_repair_operator(self, destroyed_solution, operator_name):
        """Apply the chosen repair operator."""
        if operator_name == "regret":
            return RepairOperator.regret_reinsert(
                destroyed_solution,
                self,
                self.DistTruck,
                self.valid_stations,
                self.CustNum,
                Pa.StaDroNum,
            )
        if operator_name == "greedy":
            return RepairOperator.greedy_reinsert(
                destroyed_solution,
                self,
                self.DistTruck,
                self.valid_stations,
                self.CustNum,
                Pa.StaDroNum,
            )
        if operator_name == "global":
            return RepairOperator.collaborative_reinsert(destroyed_solution, self)
        raise ValueError(f"Unknown repair operator: {operator_name}")

    def _select_destroy_operator(self):
        """Sample one destroy operator according to current weights."""
        total_weight = sum(self.destroy_weights.values())
        probs = [
            self.destroy_weights[op] / total_weight
            for op in self.destroy_operators
        ]
        return random.choices(self.destroy_operators, weights=probs, k=1)[0]

    def _apply_destroy_operator(self, solution, operator_name):
        """Apply the chosen destroy operator."""
        if operator_name == "worst_remove":
            return DestroyOperator.worst_remove(
                solution, self, self.DESTROY_RATE, self.depot_end
            )
        if operator_name == "random_remove":
            return DestroyOperator.random_remove(
                solution, self.valid_stations, self.DESTROY_RATE, self.depot_end
            )
        raise ValueError(f"Unknown destroy operator: {operator_name}")

    def _update_weights(self, current_sol, iteration):
        """Update destroy and repair operator weights based on performance."""
        time_gap = current_sol.total_time - self.best_solution.total_time
        if abs(time_gap) > 1e-6:
            reward_magnitude = math.log(1 + abs(time_gap)) * Pa.REWARD_SCALE
            reward = reward_magnitude if time_gap < 0 else -reward_magnitude
        else:
            reward = 0.0

        if abs(reward) < Pa.MIN_REWARD_THRESHOLD:
            reward = (
                Pa.MIN_REWARD_THRESHOLD if reward >= 0 else -Pa.MIN_REWARD_THRESHOLD
            )

        for op in self.repair_operators:
            self.operator_scores[op] += reward
        for op in self.repair_operators:
            self.operator_scores[op] *= Pa.SCORE_DECAY_FACTOR

        for op in self.destroy_operators:
            self.destroy_scores[op] += reward
        for op in self.destroy_operators:
            self.destroy_scores[op] *= Pa.SCORE_DECAY_FACTOR

        temperature_weight = max(
            Pa.MIN_SOFTMAX_TEMPERATURE, 1.0 - iteration / self.MAX_ITER
        )

        exp_scores = {
            op: math.exp(score / temperature_weight)
            for op, score in self.operator_scores.items()
        }
        total_exp = sum(exp_scores.values())

        destroy_exp_scores = {
            op: math.exp(score / temperature_weight)
            for op, score in self.destroy_scores.items()
        }
        destroy_total_exp = sum(destroy_exp_scores.values())

        if iteration < self.MAX_ITER // 2:
            min_w, max_w = Pa.MIN_WEIGHT_EARLY, Pa.MAX_WEIGHT_EARLY
        else:
            min_w, max_w = Pa.MIN_WEIGHT_LATE, Pa.MAX_WEIGHT_LATE

        for op in self.repair_operators:
            new_weight = exp_scores[op] / total_exp
            self.operator_weights[op] = max(min_w, min(max_w, new_weight))
        total_w = sum(self.operator_weights.values())
        for op in self.repair_operators:
            self.operator_weights[op] /= total_w

        for op in self.destroy_operators:
            new_weight = destroy_exp_scores[op] / destroy_total_exp
            self.destroy_weights[op] = max(min_w, min(max_w, new_weight))
        destroy_total_w = sum(self.destroy_weights.values())
        for op in self.destroy_operators:
            self.destroy_weights[op] /= destroy_total_w

    def get_operator_statistics(self):
        """Return aggregated operator usage statistics."""
        return self.operator_stats.get_statistics()
