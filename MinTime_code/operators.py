import math
import random
from collections import defaultdict

import parameters as Pa
from core import PhysicsCalculator, RouteManager, SpeedOptimizer, SolutionEvaluator


class DestroyOperator:
    """
    Destroy operators for the min-time ALNS.
    They remove customers and related assignments to create partial solutions.
    """

    @classmethod
    def random_remove(cls, sol, valid_stations, remove_rate, depot_end):
        """Randomly remove non-depot nodes and related drone assignments."""
        new_sol = sol.clone()
        valid_nodes = [n for n in new_sol.truck_route if n not in {0, depot_end}]
        remove_num = int(len(valid_nodes) * remove_rate)
        removed = random.sample(valid_nodes, remove_num) if valid_nodes else []

        # Remove selected nodes from truck route but keep start and end depots
        new_sol.truck_route = [
            n for n in new_sol.truck_route
            if n not in removed or n in {0, depot_end}
        ]

        # Remove affected drone tasks
        del_keys = [
            (c, s) for (c, s) in new_sol.drone_assignments
            if c in removed or s in removed
        ]
        for k in del_keys:
            del new_sol.drone_assignments[k]

        # Rebuild active station set
        active_stations = set()
        for s in valid_stations:
            if any(k[1] == s for k in new_sol.drone_assignments):
                active_stations.add(s)
        new_sol.active_stations = active_stations
        return new_sol

    @classmethod
    def worst_remove(cls, sol, problem, remove_rate, depot_end):
        """
        Time-based destroy operator.

        For each customer, estimate the time reduction if it is removed:
        - drone tasks: impact on station-drone schedule and truck detour
        - truck-only: impact on truck travel time
        Customers with larger time impact are more likely to be removed.
        """
        new_sol = sol.clone()
        time_impact = {}

        # Group drone tasks by (station, drone_id)
        station_drone_tasks = defaultdict(list)
        for (cust, station), (drone_id, v_out, v_back) in new_sol.drone_assignments.items():
            dist = problem.DistDrone[station][cust]

            vertical_time_up = Pa.FlightHeight / Pa.vSpeedUp
            vertical_time_down = Pa.FlightHeight / Pa.vSpeedDown

            outbound_time = vertical_time_up + dist / v_out + vertical_time_down
            return_time = vertical_time_up + dist / v_back + vertical_time_down

            station_drone_tasks[(station, drone_id)].append(
                {
                    "customer": cust,
                    "outbound": outbound_time,
                    "return": return_time,
                    "total": outbound_time + return_time,
                }
            )

        # Compute time impact for each customer
        for cust in range(1, problem.CustNum + 1):
            drone_service = False
            station = None
            drone_id = None

            for (c, s) in new_sol.drone_assignments:
                if c == cust:
                    drone_service = True
                    station = s
                    drone_id = new_sol.drone_assignments[(c, s)][0]
                    break

            if drone_service:
                tasks = station_drone_tasks[(station, drone_id)]
                tasks_sorted = sorted(tasks, key=lambda x: x["total"], reverse=True)
                cust_idx = next(i for i, t in enumerate(tasks_sorted) if t["customer"] == cust)

                if cust_idx == 0:
                    # Customer is the farthest in this drone sequence
                    if len(tasks_sorted) > 1:
                        # Station still serves other customers
                        second_task = tasks_sorted[1]
                        current_task = tasks_sorted[0]
                        impact = second_task["return"] + current_task["outbound"]
                    else:
                        # Station serves only this customer
                        current_task = tasks_sorted[0]
                        impact = current_task["outbound"]

                        detour_savings = 0.0
                        if station in new_sol.truck_route:
                            idx = new_sol.truck_route.index(station)
                            if 0 < idx < len(new_sol.truck_route) - 1:
                                prev_node = new_sol.truck_route[idx - 1]
                                next_node = new_sol.truck_route[idx + 1]

                                current_detour = (
                                    problem.DistTruck[prev_node][station]
                                    + problem.DistTruck[station][next_node]
                                ) / Pa.TruckSpeed
                                direct_time = (
                                    problem.DistTruck[prev_node][next_node]
                                    / Pa.TruckSpeed
                                )
                                detour_savings = current_detour - direct_time

                        impact += detour_savings

                    time_impact[cust] = impact
                else:
                    # Non-farthest drone customer: low priority for removal
                    time_impact[cust] = 0.0
            else:
                # Served only by truck
                if cust not in new_sol.truck_route:
                    time_impact[cust] = 0.0
                    continue

                idx = new_sol.truck_route.index(cust)
                if idx == 0 or idx >= len(new_sol.truck_route) - 1:
                    time_impact[cust] = 0.0
                    continue

                prev_node = new_sol.truck_route[idx - 1]
                next_node = new_sol.truck_route[idx + 1]

                current_time = (
                    problem.DistTruck[prev_node][cust]
                    + problem.DistTruck[cust][next_node]
                ) / Pa.TruckSpeed
                direct_time = (
                    problem.DistTruck[prev_node][next_node] / Pa.TruckSpeed
                )
                detour_time = current_time - direct_time
                time_impact[cust] = detour_time

        if not time_impact:
            return new_sol

        # Build roulette wheel based on normalized time impact
        sorted_cust = sorted(time_impact.items(), key=lambda x: -x[1])
        max_impact = max(time_impact.values())
        min_impact = min(time_impact.values())

        prob_dist = []
        for cust, impact in sorted_cust:
            if impact < 0:
                normalized = 0.0
            elif max_impact - min_impact < 1e-6:
                normalized = 1.0
            else:
                normalized = (impact - min_impact) / (max_impact - min_impact + 1e-6)

            weight = (normalized + 1e-6) ** 2
            prob_dist.append((cust, weight))

        total_weight = sum(w for _, w in prob_dist)
        remove_num = max(1, int(len(prob_dist) * remove_rate))
        removed = []

        for _ in range(remove_num):
            if not prob_dist or total_weight < 1e-6:
                break

            pick = random.uniform(0, total_weight)
            current = 0.0
            for i, (cust, weight) in enumerate(prob_dist):
                current += weight
                if current >= pick:
                    removed.append(cust)
                    total_weight -= weight
                    del prob_dist[i]
                    break

        if not removed:
            return new_sol

        # Remove customers from truck route
        new_sol.truck_route = [
            n for n in new_sol.truck_route
            if n not in removed or n in {0, depot_end}
        ]

        # Remove corresponding drone tasks
        del_keys = [(c, s) for (c, s) in new_sol.drone_assignments if c in removed]
        for k in del_keys:
            del new_sol.drone_assignments[k]

        # Rebuild active station set and remove inactive stations from route
        active_stations = set()
        for s in problem.valid_stations:
            if any(k[1] == s for k in new_sol.drone_assignments):
                active_stations.add(s)

        inactive_stations = new_sol.active_stations - active_stations
        if inactive_stations:
            new_sol.truck_route = [
                n for n in new_sol.truck_route
                if n not in inactive_stations
            ]

        new_sol.active_stations = active_stations
        return new_sol


class RepairOperator:
    """
    Repair operators for the min-time ALNS.
    They reinsert missing customers using truck or drone service.
    """

    @classmethod
    def _calculate_truck_insertion_impact(cls, sol, cust, dist_truck):
        """Compute time increase and best position for inserting a customer on the truck route."""
        best_pos = RouteManager.find_best_insertion(sol.truck_route, cust, dist_truck)

        if best_pos <= 0 or best_pos >= len(sol.truck_route):
            return 0.0, best_pos

        prev = sol.truck_route[best_pos - 1]
        next_ = sol.truck_route[best_pos]

        time_delta = (
            dist_truck[prev][cust] + dist_truck[cust][next_] - dist_truck[prev][next_]
        ) / Pa.TruckSpeed

        return time_delta, best_pos

    @classmethod
    def _calculate_drone_insertion_impact(cls, sol, cust, station, problem, dist_truck):
        """
        Compute time impact of assigning a customer to a station by drone.

        Returns (time_impact, drone_id, v_out, v_back) or None if infeasible in energy.
        """
        demand = problem.CustDemand[cust - 1]
        dist = problem.DistDrone[station][cust]

        speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
        if speed_pair is None:
            return None

        v_out, v_back = speed_pair

        vertical_time_up = Pa.FlightHeight / Pa.vSpeedUp
        vertical_time_down = Pa.FlightHeight / Pa.vSpeedDown

        new_outbound = vertical_time_up + dist / v_out + vertical_time_down
        new_return = vertical_time_up + dist / v_back + vertical_time_down
        new_task_total = new_outbound + new_return

        station_activation_time = 0.0
        if station not in sol.active_stations:
            best_pos = RouteManager.find_best_insertion(
                sol.truck_route, station, dist_truck
            )
            if best_pos > 0 and best_pos < len(sol.truck_route):
                prev = sol.truck_route[best_pos - 1]
                next_ = sol.truck_route[best_pos]
                detour_dist = (
                    dist_truck[prev][station]
                    + dist_truck[station][next_]
                    - dist_truck[prev][next_]
                )
                station_activation_time = detour_dist / Pa.TruckSpeed

        best_time_impact = float("inf")
        best_drone_id = 0

        for drone_id in range(Pa.StaDroNum):
            current_tasks = []
            for (c, s), (did, v1, v2) in sol.drone_assignments.items():
                if s == station and did == drone_id:
                    d = problem.DistDrone[s][c]

                    vt_up = Pa.FlightHeight / Pa.vSpeedUp
                    vt_down = Pa.FlightHeight / Pa.vSpeedDown

                    outbound = vt_up + d / v1 + vt_down
                    ret = vt_up + d / v2 + vt_down
                    total = outbound + ret

                    current_tasks.append(
                        {
                            "customer": c,
                            "outbound": outbound,
                            "return": ret,
                            "total": total,
                        }
                    )

            if len(current_tasks) == 0:
                old_max_time = 0.0
            else:
                current_tasks_sorted = sorted(
                    current_tasks, key=lambda x: x["total"], reverse=True
                )
                cumulative_time = 0.0
                for i in range(len(current_tasks_sorted)):
                    task = current_tasks_sorted[-(i + 1)]
                    if i < len(current_tasks_sorted) - 1:
                        cumulative_time += task["outbound"] + task["return"]
                    else:
                        cumulative_time += task["outbound"]
                old_max_time = cumulative_time

            all_tasks = current_tasks + [
                {
                    "customer": cust,
                    "outbound": new_outbound,
                    "return": new_return,
                    "total": new_task_total,
                }
            ]
            all_tasks_sorted = sorted(all_tasks, key=lambda x: x["total"], reverse=True)

            cumulative_time = 0.0
            for i in range(len(all_tasks_sorted)):
                task = all_tasks_sorted[-(i + 1)]
                if i < len(all_tasks_sorted) - 1:
                    cumulative_time += task["outbound"] + task["return"]
                else:
                    cumulative_time += task["outbound"]
            new_max_time = cumulative_time

            time_delta = new_max_time - old_max_time
            if station not in sol.active_stations:
                total_impact = time_delta + station_activation_time
            else:
                total_impact = time_delta

            if total_impact < best_time_impact:
                best_time_impact = total_impact
                best_drone_id = drone_id

        return best_time_impact, best_drone_id, v_out, v_back

    @staticmethod
    def _cleanup_inactive_stations(sol):
        """Remove stations with no drone tasks and clean truck route."""
        active_stations = set()
        for s in sol.active_stations:
            if any(k[1] == s for k in sol.drone_assignments):
                active_stations.add(s)
            else:
                if s in sol.truck_route:
                    sol.truck_route.remove(s)
        sol.active_stations = active_stations

    @staticmethod
    def _calculate_station_activation_times(sol, problem):
        """
        Compute activation time (truck arrival time) for each active station
        along the current truck route.
        """
        activation_times = {}
        cumulative_time = 0.0

        for i in range(len(sol.truck_route) - 1):
            current = sol.truck_route[i]
            next_node = sol.truck_route[i + 1]
            travel_time = problem.DistTruck[current][next_node] / Pa.TruckSpeed
            cumulative_time += travel_time

            if next_node in sol.active_stations:
                activation_times[next_node] = cumulative_time

        return activation_times

    @staticmethod
    def _estimate_activation_time_at_position(sol, station, position, problem):
        """Estimate activation time if a station is inserted at a given position."""
        if position <= 0 or position >= len(sol.truck_route):
            return 0.0

        cumulative_time = 0.0
        for i in range(min(position, len(sol.truck_route) - 1)):
            current = sol.truck_route[i]
            next_node = sol.truck_route[i + 1] if i + 1 < len(sol.truck_route) else station

            if i == position - 1:
                travel_time = problem.DistTruck[current][station] / Pa.TruckSpeed
            else:
                travel_time = problem.DistTruck[current][next_node] / Pa.TruckSpeed

            cumulative_time += travel_time

        return cumulative_time

    @classmethod
    def greedy_reinsert(cls, sol, problem, dist_truck, valid_stations, cust_num, sta_dro_num):
        """
        Greedy repair operator.

        For each missing customer, compare truck insertion and all feasible
        drone assignments, and choose the option with minimum time impact.
        """
        new_sol = sol.clone()

        served = {c for (c, _) in new_sol.drone_assignments}
        served.update(n for n in new_sol.truck_route if 1 <= n <= cust_num)
        missing = set(range(1, cust_num + 1)) - served

        for c in sorted(missing):
            best_option = None
            best_time = float("inf")

            truck_time, truck_pos = cls._calculate_truck_insertion_impact(
                new_sol, c, dist_truck
            )
            if truck_time < best_time:
                best_time = truck_time
                best_option = ("truck", truck_pos)

            for s in valid_stations:
                if not problem.reachability[s][c]:
                    continue

                result = cls._calculate_drone_insertion_impact(
                    new_sol, c, s, problem, dist_truck
                )
                if result is None:
                    continue

                time_impact, drone_id, v_out, v_back = result
                if time_impact < best_time:
                    best_time = time_impact
                    best_option = ("drone", s, drone_id, v_out, v_back)

            if best_option is None:
                RouteManager.insert_node(new_sol.truck_route, c, dist_truck)
            elif best_option[0] == "truck":
                pos = best_option[1]
                new_sol.truck_route.insert(pos, c)
            else:
                _, s, drone_id, v_out, v_back = best_option

                if s not in new_sol.truck_route:
                    if len(new_sol.truck_route) > 1:
                        new_sol.truck_route.insert(1, s)
                    else:
                        new_sol.truck_route = RouteManager.insert_node(
                            new_sol.truck_route, s, dist_truck
                        )

                if s not in new_sol.active_stations:
                    new_sol.active_stations.add(s)

                new_sol.drone_assignments[(c, s)] = (drone_id, v_out, v_back)

        cls._cleanup_inactive_stations(new_sol)
        new_sol.truck_route = RouteManager.standardize_route(
            new_sol.truck_route, problem, new_sol
        )
        return new_sol

    @classmethod
    def regret_reinsert(cls, sol, problem, dist_truck, valid_stations, cust_num, sta_dro_num):
        """
        Regret-based repair operator.

        For each missing customer, build candidate options and compute
        regret = second_best_time - best_time, then insert customers
        in decreasing regret order.
        """
        new_sol = sol.clone()

        served = {c for (c, _) in new_sol.drone_assignments}
        served.update(n for n in new_sol.truck_route if 1 <= n <= cust_num)
        missing = set(range(1, cust_num + 1)) - served

        insertion_data = {}
        for c in missing:
            options = []

            truck_time, truck_pos = cls._calculate_truck_insertion_impact(
                new_sol, c, dist_truck
            )
            options.append(("truck", truck_time, truck_pos, None, None, None))

            for s in valid_stations:
                if not problem.reachability[s][c]:
                    continue

                result = cls._calculate_drone_insertion_impact(
                    new_sol, c, s, problem, dist_truck
                )
                if result is not None:
                    time_impact, drone_id, v_out, v_back = result
                    options.append(("drone", time_impact, s, drone_id, v_out, v_back))

            if not options:
                continue

            sorted_options = sorted(options, key=lambda x: x[1])
            best_option = sorted_options[0]
            if len(sorted_options) > 1:
                regret = sorted_options[1][1] - sorted_options[0][1]
            else:
                regret = 0.0

            insertion_data[c] = {
                "best": best_option,
                "regret": regret,
            }

        processing_order = sorted(
            insertion_data.items(),
            key=lambda x: (-x[1]["regret"], x[0]),
        )

        for c, data in processing_order:
            option_type, time_val, param1, param2, param3, param4 = data["best"]

            if option_type == "truck":
                pos = param1
                if 0 < pos <= len(new_sol.truck_route):
                    new_sol.truck_route.insert(pos, c)
                else:
                    RouteManager.insert_node(new_sol.truck_route, c, dist_truck)
            else:
                s = param1
                v_out = param3
                v_back = param4

                if s not in new_sol.truck_route:
                    if len(new_sol.truck_route) > 1:
                        new_sol.truck_route.insert(1, s)
                    else:
                        new_sol.truck_route = RouteManager.insert_node(
                            new_sol.truck_route, s, dist_truck
                        )
                if s not in new_sol.active_stations:
                    new_sol.active_stations.add(s)

                drone_tasks = defaultdict(list)
                for (cust, st), (did, v1, v2) in new_sol.drone_assignments.items():
                    if st == s:
                        d = problem.DistDrone[s][cust]
                        vt_up = Pa.FlightHeight / Pa.vSpeedUp
                        vt_down = Pa.FlightHeight / Pa.vSpeedDown
                        outbound = vt_up + d / v1 + vt_down
                        ret = vt_up + d / v2 + vt_down
                        total = outbound + ret
                        drone_tasks[did].append(
                            {"outbound": outbound, "return": ret, "total": total}
                        )

                new_d = problem.DistDrone[s][c]
                new_vt_up = Pa.FlightHeight / Pa.vSpeedUp
                new_vt_down = Pa.FlightHeight / Pa.vSpeedDown
                new_outbound = new_vt_up + new_d / v_out + new_vt_down
                new_return = new_vt_up + new_d / v_back + new_vt_down
                new_total = new_outbound + new_return

                best_drone_id = 0
                min_max_time = float("inf")

                for did in range(Pa.StaDroNum):
                    current_tasks = drone_tasks.get(did, [])
                    all_tasks = current_tasks + [
                        {
                            "outbound": new_outbound,
                            "return": new_return,
                            "total": new_total,
                        }
                    ]
                    all_tasks_sorted = sorted(
                        all_tasks, key=lambda x: x["total"], reverse=True
                    )

                    cumulative_time = 0.0
                    for i in range(len(all_tasks_sorted)):
                        task = all_tasks_sorted[-(i + 1)]
                        if i < len(all_tasks_sorted) - 1:
                            cumulative_time += task["outbound"] + task["return"]
                        else:
                            cumulative_time += task["outbound"]

                    if cumulative_time < min_max_time:
                        min_max_time = cumulative_time
                        best_drone_id = did

                drone_id = best_drone_id
                new_sol.drone_assignments[(c, s)] = (drone_id, v_out, v_back)

        cls._cleanup_inactive_stations(new_sol)
        new_sol.truck_route = RouteManager.standardize_route(
            new_sol.truck_route, problem, new_sol
        )
        return new_sol

    @classmethod
    def collaborative_reinsert(cls, sol, problem):
        """
            Collaborative repair operator.

            First balances drone assignments using estimated activation times,
            then inserts remaining customers into the truck route.
        """
        new_sol = sol.clone()

        served = {c for (c, _) in new_sol.drone_assignments}
        served.update(n for n in new_sol.truck_route if 1 <= n <= problem.CustNum)
        missing = list(set(range(1, problem.CustNum + 1)) - served)

        if not missing:
            return new_sol

        station_activation_times = cls._calculate_station_activation_times(
            new_sol, problem
        )

        for c in sorted(missing):
            best_option = None
            best_completion_time = float("inf")

            for s in problem.valid_stations:
                if not problem.reachability[s][c]:
                    continue

                demand = problem.CustDemand[c - 1]
                dist = problem.DistDrone[s][c]

                speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
                if speed_pair is None:
                    continue

                v_out, v_back = speed_pair

                vt_up = Pa.FlightHeight / Pa.vSpeedUp
                vt_down = Pa.FlightHeight / Pa.vSpeedDown
                new_outbound = vt_up + dist / v_out + vt_down
                new_return = vt_up + dist / v_back + vt_down
                new_total = new_outbound + new_return

                if s in station_activation_times:
                    activation_time = station_activation_times[s]
                else:
                    activation_time = cls._estimate_activation_time_at_position(
                        new_sol, s, 1, problem
                    )

                for drone_id in range(Pa.StaDroNum):
                    current_tasks = []
                    for (cust, st), (did, v1, v2) in new_sol.drone_assignments.items():
                        if st == s and did == drone_id:
                            d = problem.DistDrone[s][cust]
                            vt = (
                                Pa.FlightHeight / Pa.vSpeedUp
                                + Pa.FlightHeight / Pa.vSpeedDown
                            )
                            outbound = vt + d / v1
                            ret = vt + d / v2
                            total = outbound + ret
                            current_tasks.append(
                                {"outbound": outbound, "return": ret, "total": total}
                            )

                    all_tasks = current_tasks + [
                        {
                            "outbound": new_outbound,
                            "return": new_return,
                            "total": new_total,
                        }
                    ]
                    all_tasks_sorted = sorted(
                        all_tasks, key=lambda x: x["total"], reverse=True
                    )

                    cumulative_time = 0.0
                    for i in range(len(all_tasks_sorted)):
                        task = all_tasks_sorted[-(i + 1)]
                        if i < len(all_tasks_sorted) - 1:
                            cumulative_time += task["outbound"] + task["return"]
                        else:
                            cumulative_time += task["outbound"]

                    expected_completion = activation_time + cumulative_time

                    if expected_completion < best_completion_time:
                        best_completion_time = expected_completion
                        best_option = (s, drone_id, v_out, v_back)

            if best_option:
                s, drone_id, v_out, v_back = best_option

                if s not in new_sol.truck_route:
                    best_pos = RouteManager.find_best_insertion_time(
                        new_sol.truck_route, s, problem.DistTruck, Pa.TruckSpeed
                    )
                    new_sol.truck_route.insert(best_pos, s)
                    station_activation_times[s] = cls._estimate_activation_time_at_position(
                        new_sol, s, best_pos, problem
                    )

                if s not in new_sol.active_stations:
                    new_sol.active_stations.add(s)

                new_sol.drone_assignments[(c, s)] = (drone_id, v_out, v_back)
            else:
                RouteManager.insert_node(new_sol.truck_route, c, problem.DistTruck)

        cls._cleanup_inactive_stations(new_sol)
        new_sol.truck_route = RouteManager.standardize_route(
            new_sol.truck_route, problem, new_sol
        )
        return new_sol


class LocalSearch:
    """
    Local search operators for the min-time ALNS:
    - two_opt: truck route improvement
    - service_mode_switch: switch between truck and drone service
    - station_swap: remove and reassign stations
    """

    _show_progress = True

    @classmethod
    def set_show_progress(cls, show_progress):
        """Enable or disable optional progress reporting."""
        cls._show_progress = show_progress

    @staticmethod
    def two_opt(solution, problem, max_trials=100):
        """Randomized 2-opt on the truck route using travel time as objective."""
        new_sol = solution.clone()

        if len(new_sol.truck_route) < 4:
            return new_sol

        original_route = [
            n
            for n in new_sol.truck_route
            if n not in {problem.depot_start, problem.depot_end}
        ]
        if len(original_route) < 3:
            return new_sol

        best_route = original_route.copy()
        best_time = (
            sum(
                problem.DistTruck[a][b]
                for a, b in zip(
                    [problem.depot_start] + best_route,
                    best_route + [problem.depot_end],
                )
            )
            / Pa.TruckSpeed
        )

        for _ in range(max_trials):
            sample_range = list(range(1, len(original_route) - 1))
            if len(sample_range) < 2:
                continue

            i, j = sorted(random.sample(sample_range, 2))
            new_route = (
                original_route[:i]
                + original_route[i : j + 1][::-1]
                + original_route[j + 1 :]
            )
            current_time = (
                sum(
                    problem.DistTruck[a][b]
                    for a, b in zip(
                        [problem.depot_start] + new_route,
                        new_route + [problem.depot_end],
                    )
                )
                / Pa.TruckSpeed
            )

            if current_time < best_time - 1e-6:
                best_route = new_route
                best_time = current_time

        new_sol.truck_route = RouteManager.standardize_route(
            [problem.depot_start] + best_route + [problem.depot_end],
            problem,
            new_sol,
        )
        return SolutionEvaluator.evaluate(new_sol, problem)

    @staticmethod
    def service_mode_switch(current_sol, problem, best_solution):
        """
        Try switching service mode for customers (drone <-> truck)
        to reduce the overall completion time.
        """
        new_sol = current_sol.clone()
        original_time = current_sol.total_time

        current_drones = set(c for c, _ in new_sol.drone_assignments)
        best_drones = set(c for c, _ in best_solution.drone_assignments)
        diff_customers = list(current_drones.symmetric_difference(best_drones))
        other_customers = [
            c for c in range(1, problem.CustNum + 1) if c not in diff_customers
        ]
        all_customers = diff_customers + other_customers

        best_global_sol = new_sol.clone()
        best_global_time = original_time
        switches_made = 0
        attempts = 0

        for cust in all_customers:
            attempts += 1
            is_drone_served = any(c == cust for (c, _) in new_sol.drone_assignments)

            if is_drone_served:
                test_sol = new_sol.clone()
                current_station = None
                for (c, s) in test_sol.drone_assignments.keys():
                    if c == cust:
                        current_station = s
                        break
                if current_station is None:
                    continue

                del test_sol.drone_assignments[(cust, current_station)]

                station_has_tasks = any(
                    s == current_station
                    for (c, s) in test_sol.drone_assignments.keys()
                )
                if not station_has_tasks:
                    test_sol.active_stations.discard(current_station)
                    test_sol.truck_route = [
                        n for n in test_sol.truck_route if n != current_station
                    ]

                if cust not in test_sol.truck_route:
                    insert_pos = RouteManager.find_best_insertion(
                        test_sol.truck_route, cust, problem.DistTruck
                    )
                    test_sol.truck_route.insert(insert_pos, cust)

                test_sol = SpeedOptimizer.optimize(test_sol, problem)
                test_sol.truck_route = RouteManager.standardize_route(
                    test_sol.truck_route, problem, test_sol
                )
                test_sol = SolutionEvaluator.evaluate(test_sol, problem)

                if test_sol.total_time < best_global_time:
                    best_global_sol = test_sol.clone()
                    best_global_time = test_sol.total_time
                    new_sol = test_sol.clone()
                    switches_made += 1

            else:
                if cust not in new_sol.truck_route:
                    continue

                best_station_sol = None
                best_station_time = best_global_time

                for s in problem.valid_stations:
                    if not problem.reachability[s][cust]:
                        continue

                    demand = problem.CustDemand[cust - 1]
                    dist = problem.DistDrone[s][cust]
                    speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
                    if speed_pair is None:
                        continue
                    v_out, v_back = speed_pair

                    test_sol = new_sol.clone()
                    test_sol.truck_route = [
                        n for n in test_sol.truck_route if n != cust
                    ]

                    if s not in test_sol.active_stations:
                        if s not in test_sol.truck_route:
                            if len(test_sol.truck_route) > 1:
                                test_sol.truck_route.insert(1, s)
                            else:
                                insert_pos = RouteManager.find_best_insertion(
                                    test_sol.truck_route, s, problem.DistTruck
                                )
                                test_sol.truck_route.insert(insert_pos, s)
                        test_sol.active_stations.add(s)

                    drone_time_loads = {i: 0.0 for i in range(Pa.StaDroNum)}
                    for (c, st), (did, v1, v2) in test_sol.drone_assignments.items():
                        if st == s:
                            d = problem.DistDrone[s][c]
                            vt = (
                                Pa.FlightHeight / Pa.vSpeedUp
                                + Pa.FlightHeight / Pa.vSpeedDown
                            )
                            task_time = vt + d / v1 + vt + d / v2
                            drone_time_loads[did] += task_time

                    drone_id = min(
                        drone_time_loads.keys(), key=lambda d: drone_time_loads[d]
                    )
                    test_sol.drone_assignments[(cust, s)] = (drone_id, v_out, v_back)

                    test_sol = SpeedOptimizer.optimize(test_sol, problem)
                    test_sol.truck_route = RouteManager.standardize_route(
                        test_sol.truck_route, problem, test_sol
                    )
                    test_sol = SolutionEvaluator.evaluate(test_sol, problem)

                    if test_sol.total_time < best_station_time:
                        best_station_sol = test_sol.clone()
                        best_station_time = test_sol.total_time

                if best_station_sol and best_station_time < best_global_time:
                    best_global_sol = best_station_sol.clone()
                    best_global_time = best_station_time
                    new_sol = best_station_sol.clone()
                    switches_made += 1

        drone_customers = {}
        for (c, s), (did, v1, v2) in new_sol.drone_assignments.items():
            drone_customers[c] = (s, did, v1, v2)

        station_customers = defaultdict(list)
        for c, (s, did, v1, v2) in drone_customers.items():
            station_customers[s].append(c)

        active_station_list = sorted(new_sol.active_stations)
        swap_improved = True

        while swap_improved:
            swap_improved = False
            best_swap_sol = None
            best_swap_time = best_global_time

            for i in range(len(active_station_list)):
                for j in range(i + 1, len(active_station_list)):
                    s1 = active_station_list[i]
                    s2 = active_station_list[j]

                    custs_s1 = station_customers.get(s1, [])
                    custs_s2 = station_customers.get(s2, [])

                    for cA in custs_s1:
                        for cB in custs_s2:
                            demandA = problem.CustDemand[cA - 1]
                            distA_s2 = problem.DistDrone[s2][cA]
                            speedA = SpeedOptimizer._find_best_speeds(
                                demandA, distA_s2, problem
                            )
                            if speedA is None:
                                continue

                            demandB = problem.CustDemand[cB - 1]
                            distB_s1 = problem.DistDrone[s1][cB]
                            speedB = SpeedOptimizer._find_best_speeds(
                                demandB, distB_s1, problem
                            )
                            if speedB is None:
                                continue

                            test_sol = new_sol.clone()
                            del test_sol.drone_assignments[(cA, s1)]
                            del test_sol.drone_assignments[(cB, s2)]

                            drone_loads_s2 = defaultdict(float)
                            for (c, st), (did, v1, v2) in test_sol.drone_assignments.items():
                                if st == s2:
                                    d = problem.DistDrone[s2][c]
                                    vt = (
                                        Pa.FlightHeight / Pa.vSpeedUp
                                        + Pa.FlightHeight / Pa.vSpeedDown
                                    )
                                    drone_loads_s2[did] += vt + d / v1 + vt + d / v2
                            for did in range(Pa.StaDroNum):
                                if did not in drone_loads_s2:
                                    drone_loads_s2[did] = 0.0
                            dA_new = min(drone_loads_s2, key=drone_loads_s2.get)

                            drone_loads_s1 = defaultdict(float)
                            for (c, st), (did, v1, v2) in test_sol.drone_assignments.items():
                                if st == s1:
                                    d = problem.DistDrone[s1][c]
                                    vt = (
                                        Pa.FlightHeight / Pa.vSpeedUp
                                        + Pa.FlightHeight / Pa.vSpeedDown
                                    )
                                    drone_loads_s1[did] += vt + d / v1 + vt + d / v2
                            for did in range(Pa.StaDroNum):
                                if did not in drone_loads_s1:
                                    drone_loads_s1[did] = 0.0
                            dB_new = min(drone_loads_s1, key=drone_loads_s1.get)

                            vA_out, vA_back = speedA
                            vB_out, vB_back = speedB
                            test_sol.drone_assignments[(cA, s2)] = (
                                dA_new,
                                vA_out,
                                vA_back,
                            )
                            test_sol.drone_assignments[(cB, s1)] = (
                                dB_new,
                                vB_out,
                                vB_back,
                            )

                            test_sol = SpeedOptimizer.optimize(test_sol, problem)
                            test_sol = SolutionEvaluator.evaluate(test_sol, problem)

                            if test_sol.total_time < best_swap_time:
                                best_swap_time = test_sol.total_time
                                best_swap_sol = test_sol.clone()

            if best_swap_sol and best_swap_time < best_global_time:
                best_global_sol = best_swap_sol.clone()
                best_global_time = best_swap_time
                new_sol = best_swap_sol.clone()
                swap_improved = True

                drone_customers = {}
                for (c, s), (did, v1, v2) in new_sol.drone_assignments.items():
                    drone_customers[c] = (s, did, v1, v2)
                station_customers = defaultdict(list)
                for c, (s, did, v1, v2) in drone_customers.items():
                    station_customers[s].append(c)

        total_improvement = original_time - best_global_time
        if LocalSearch._show_progress:
            if total_improvement > 0:
                pass
            else:
                pass

        if best_global_time < original_time:
            return best_global_sol
        else:
            return None

    @staticmethod
    def station_swap(solution, problem):
        """
        Try removing each active station in turn and reassigning its customers,
        first to other stations, then to the truck route if needed.
        """
        if not solution.active_stations:
            return solution


        best_global_sol = solution.clone()
        best_global_time = solution.total_time
        best_removed_station = None

        for target_station in list(solution.active_stations):

            new_sol = solution.clone()
            affected_customers = [
                c for (c, s) in new_sol.drone_assignments if s == target_station
            ]

            new_sol.active_stations.discard(target_station)
            new_sol.truck_route = [
                n for n in new_sol.truck_route if n != target_station
            ]

            for c in affected_customers:
                if (c, target_station) in new_sol.drone_assignments:
                    del new_sol.drone_assignments[(c, target_station)]

            unassigned = set(affected_customers)

            best_assignment_sol = new_sol.clone()
            best_assignment_time = float("inf")

            for c in sorted(affected_customers):
                if c not in unassigned:
                    continue

                best_station_for_c = None
                best_station_time = best_assignment_time

                for s in new_sol.active_stations:
                    if not problem.reachability[s][c]:
                        continue

                    demand = problem.CustDemand[c - 1]
                    dist = problem.DistDrone[s][c]
                    speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
                    if not speed_pair:
                        continue

                    v_out, v_back = speed_pair
                    test_sol = best_assignment_sol.clone()

                    drone_time_loads = {i: 0.0 for i in range(Pa.StaDroNum)}
                    for (cust, st), (did, v1, v2) in test_sol.drone_assignments.items():
                        if st == s:
                            d = problem.DistDrone[s][cust]
                            vt = (
                                Pa.FlightHeight / Pa.vSpeedUp
                                + Pa.FlightHeight / Pa.vSpeedDown
                            )
                            task_time = vt + d / v1 + vt + d / v2
                            drone_time_loads[did] += task_time

                    drone_id = min(
                        drone_time_loads.keys(), key=lambda d: drone_time_loads[d]
                    )
                    test_sol.drone_assignments[(c, s)] = (drone_id, v_out, v_back)

                    test_sol = SpeedOptimizer.optimize(test_sol, problem)
                    test_sol.truck_route = RouteManager.standardize_route(
                        test_sol.truck_route, problem, test_sol
                    )
                    test_sol = SolutionEvaluator.evaluate(test_sol, problem)

                    if test_sol.total_time < best_station_time:
                        best_station_for_c = s
                        best_station_time = test_sol.total_time

                if best_station_for_c:
                    demand = problem.CustDemand[c - 1]
                    dist = problem.DistDrone[best_station_for_c][c]
                    speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
                    v_out, v_back = speed_pair

                    drone_time_loads = {i: 0.0 for i in range(Pa.StaDroNum)}
                    for (cust, st), (did, v1, v2) in best_assignment_sol.drone_assignments.items():
                        if st == best_station_for_c:
                            d = problem.DistDrone[st][cust]
                            vt = (
                                Pa.FlightHeight / Pa.vSpeedUp
                                + Pa.FlightHeight / Pa.vSpeedDown
                            )
                            task_time = vt + d / v1 + vt + d / v2
                            drone_time_loads[did] += task_time

                    drone_id = min(
                        drone_time_loads.keys(), key=lambda d: drone_time_loads[d]
                    )
                    best_assignment_sol.drone_assignments[(c, best_station_for_c)] = (
                        drone_id,
                        v_out,
                        v_back,
                    )
                    best_assignment_sol = SolutionEvaluator.evaluate(
                        best_assignment_sol, problem
                    )
                    best_assignment_time = best_assignment_sol.total_time
                    unassigned.remove(c)

            if unassigned:
                for c in sorted(unassigned):
                    insert_pos = RouteManager.find_best_insertion(
                        best_assignment_sol.truck_route, c, problem.DistTruck
                    )
                    best_assignment_sol.truck_route.insert(insert_pos, c)

            best_assignment_sol.truck_route = RouteManager.two_opt_swap(
                best_assignment_sol.truck_route,
                problem.DistTruck,
                problem.depot_end,
                max_trials=30,
            )
            best_assignment_sol = SpeedOptimizer.optimize(
                best_assignment_sol, problem
            )
            best_assignment_sol.truck_route = RouteManager.standardize_route(
                best_assignment_sol.truck_route,
                problem,
                best_assignment_sol,
            )
            best_assignment_sol = SolutionEvaluator.evaluate(
                best_assignment_sol, problem
            )

            if best_assignment_sol.total_time < best_global_time:
                best_global_sol = best_assignment_sol
                best_global_time = best_assignment_sol.total_time
                best_removed_station = target_station

                if LocalSearch._show_progress:
                    pass
            else:
                if LocalSearch._show_progress:
                    pass

        if LocalSearch._show_progress:
            total_improvement = solution.total_time - best_global_time
            if total_improvement > 0:
                pass
            else:
                pass

        return best_global_sol

    @staticmethod
    def full_local_search(solution, problem, best_solution, max_depth=3):
        """
        Run a sequence of local search moves:
        two_opt -> station_swap -> service_mode_switch,
        repeated up to max_depth iterations.
        """
        current_best = solution.clone()

        for _ in range(max_depth):
            improved = False

            temp_sol = LocalSearch.two_opt(current_best, problem)
            if temp_sol.total_time < current_best.total_time:
                current_best = temp_sol.clone()
                improved = True

            temp_sol = LocalSearch.station_swap(current_best, problem)
            if temp_sol and temp_sol.total_time < current_best.total_time:
                current_best = temp_sol.clone()
                improved = True

            switch_result = LocalSearch.service_mode_switch(
                current_best, problem, best_solution
            )
            if switch_result and switch_result.total_time < current_best.total_time:
                current_best = switch_result.clone()
                improved = True

            if not improved:
                break

        return current_best
