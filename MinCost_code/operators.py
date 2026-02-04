import math
import random
from collections import defaultdict

import parameters as Pa
from core import PhysicsCalculator, RouteManager, SpeedOptimizer, SolutionEvaluator


class DestroyOperator:
    """Destroy operators that partially remove customers/stations from a solution."""

    @classmethod
    def random_remove(cls, sol, valid_stations, remove_rate, depot_end):
        """Randomly remove non-depot nodes and related assignments."""
        new_sol = sol.clone()

        valid_nodes = [n for n in new_sol.truck_route if n not in {0, depot_end}]
        remove_num = int(len(valid_nodes) * remove_rate)
        removed = random.sample(valid_nodes, remove_num) if valid_nodes else []

        new_sol.truck_route = [
            n for n in new_sol.truck_route
            if n not in removed or n in {0, depot_end}
        ]

        del_keys = [
            (c, s) for (c, s) in new_sol.drone_assignments
            if c in removed or s in removed
        ]
        for k in del_keys:
            del new_sol.drone_assignments[k]

        active_stations = set()
        for s in valid_stations:
            if any(k[1] == s for k in new_sol.drone_assignments):
                active_stations.add(s)
        new_sol.active_stations = active_stations

        return new_sol

    @classmethod
    def worst_remove(cls, sol, problem, remove_rate, depot_end):
        """
        Cost-based removal operator.

        For each customer, estimate the cost saving if it is removed:
          - truck detour saving (if served by truck)
          - drone energy + dispatch cost saving (if served by drone)
          - station detour saving if the station becomes unused
        Customers with larger savings are more likely to be removed.
        """
        new_sol = sol.clone()
        cost_savings = {}

        for cust in range(1, problem.CustNum + 1):
            drone_service = False
            station = None
            for (c, s) in new_sol.drone_assignments:
                if c == cust:
                    drone_service = True
                    station = s
                    break

            if drone_service:
                dist = problem.DistDrone[station][cust]
                demand = problem.CustDemand[cust - 1]
                v_out = new_sol.drone_assignments[(cust, station)][1]
                v_back = new_sol.drone_assignments[(cust, station)][2]

                savings = Pa.DroneDepreciationCost
                energy_cost = (
                    PhysicsCalculator.calc_energy(demand, v_out, dist, False) +
                    PhysicsCalculator.calc_energy(0, v_back, dist, True)
                ) * Pa.EnergyCost
                savings += energy_cost

                station_customers = [c for (c, s) in new_sol.drone_assignments if s == station]
                if len(station_customers) == 1 and station in new_sol.truck_route:
                    idx = new_sol.truck_route.index(station)
                    if 0 < idx < len(new_sol.truck_route) - 1:
                        prev_node = new_sol.truck_route[idx - 1]
                        next_node = new_sol.truck_route[idx + 1]
                        current_cost = (
                            problem.DistTruck[prev_node][station] +
                            problem.DistTruck[station][next_node]
                        ) * Pa.UnitTruckCost
                        direct_cost = problem.DistTruck[prev_node][next_node] * Pa.UnitTruckCost
                        savings += (current_cost - direct_cost)

                cost_savings[cust] = savings
                continue

            if cust not in new_sol.truck_route:
                cost_savings[cust] = 0.0
                continue

            idx = new_sol.truck_route.index(cust)
            if idx == 0 or idx >= len(new_sol.truck_route) - 1:
                cost_savings[cust] = 0.0
                continue

            prev_node = new_sol.truck_route[idx - 1]
            next_node = new_sol.truck_route[idx + 1]
            current_cost = (
                problem.DistTruck[prev_node][cust] +
                problem.DistTruck[cust][next_node]
            ) * Pa.UnitTruckCost
            direct_cost = problem.DistTruck[prev_node][next_node] * Pa.UnitTruckCost
            cost_savings[cust] = (current_cost - direct_cost)

        if not cost_savings:
            return new_sol

        sorted_cust = sorted(cost_savings.items(), key=lambda x: -x[1])
        max_savings = max(cost_savings.values())
        min_savings = min(cost_savings.values())

        prob_dist = []
        for cust, savings in sorted_cust:
            if savings < 0:
                normalized = 0.0
            elif max_savings - min_savings < 1e-6:
                normalized = 1.0
            else:
                normalized = (savings - min_savings) / (max_savings - min_savings + 1e-6)

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

        new_sol.truck_route = [
            n for n in new_sol.truck_route
            if n not in removed or n in {0, depot_end}
        ]

        del_keys = [(c, s) for (c, s) in new_sol.drone_assignments if c in removed]
        for k in del_keys:
            del new_sol.drone_assignments[k]

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
    """Repair operators that reinsert missing customers with truck or drone service."""

    @classmethod
    def greedy_reinsert(cls, sol, problem, dist_truck, valid_stations, cust_num, sta_dro_num):
        """
        Greedy repair by customer index.

        For each missing customer, evaluate:
          - truck insertion cost
          - drone service via reachable stations (detour + energy + dispatch cost)
        Then apply the cheapest option for that customer.
        """
        new_sol = sol.clone()

        served = {c for (c, _) in new_sol.drone_assignments}
        served.update(n for n in new_sol.truck_route if 1 <= n <= cust_num)
        missing = set(range(1, cust_num + 1)) - served

        candidate_operations = defaultdict(list)
        for c in missing:
            truck_cost = RouteManager._calc_truck_insertion_cost(
                new_sol.truck_route, c, dist_truck
            )
            candidate_operations[c].append(('truck', truck_cost, None))

            for s in valid_stations:
                if not problem.reachability[s][c]:
                    continue
                result = cls._calculate_drone_cost(
                    new_sol, c, s, problem, dist_truck
                )
                if result:
                    cost_components, speed_pair = result
                    candidate_operations[c].append(
                        ('drone', sum(cost_components.values()), s, cost_components, speed_pair)
                    )

        processed_stations = set()
        for c in sorted(missing):
            if not candidate_operations.get(c):
                continue

            best_choice = min(candidate_operations[c], key=lambda x: x[1])

            if best_choice[0] == 'truck':
                RouteManager.insert_node(new_sol.truck_route, c, dist_truck)
            else:
                s = best_choice[2]
                speed_pair = best_choice[4]

                if s not in new_sol.truck_route:
                    new_sol.truck_route = RouteManager.insert_node(new_sol.truck_route, s, dist_truck)
                    processed_stations.add(s)

                if s not in new_sol.active_stations:
                    new_sol.active_stations.add(s)

                existing = sum(1 for (_, st) in new_sol.drone_assignments if st == s)
                new_sol.drone_assignments[(c, s)] = (existing, speed_pair[0], speed_pair[1])
                missing.discard(c)

        for c in missing:
            RouteManager.insert_node(new_sol.truck_route, c, dist_truck)

        active_stations = set()
        for s in new_sol.active_stations:
            if any(k[1] == s for k in new_sol.drone_assignments):
                active_stations.add(s)
            else:
                if s in new_sol.truck_route:
                    new_sol.truck_route.remove(s)
        new_sol.active_stations = active_stations

        new_sol.truck_route = RouteManager.standardize_route(
            new_sol.truck_route, problem, new_sol
        )
        return new_sol

    @classmethod
    def _calculate_drone_cost(cls, sol, cust, station, problem, dist_matrix):
        """Return drone cost components and best speeds, or None if infeasible."""
        demand = problem.CustDemand[cust - 1]
        dist = problem.DistDrone[station][cust]

        speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
        if not speed_pair:
            return None

        cost_components = {}

        if station not in sol.truck_route:
            temp_route = sol.truck_route.copy()
            insert_pos = RouteManager.find_best_insertion(temp_route, station, dist_matrix)
            delta = (
                dist_matrix[temp_route[insert_pos - 1]][station]
                + dist_matrix[station][temp_route[insert_pos]]
                - dist_matrix[temp_route[insert_pos - 1]][temp_route[insert_pos]]
            )
            cost_components['detour'] = delta * Pa.UnitTruckCost
        else:
            cost_components['detour'] = 0.0

        energy_cost = (
            PhysicsCalculator.calc_energy(demand, speed_pair[0], dist, False) +
            PhysicsCalculator.calc_energy(0, speed_pair[1], dist, True)
        ) * Pa.EnergyCost
        cost_components['energy'] = energy_cost

        cost_components['dispatch'] = Pa.DroneDepreciationCost
        return cost_components, speed_pair

    @classmethod
    def regret_reinsert(cls, sol, problem, dist_truck, valid_stations, cust_num, sta_dro_num):
        """
        Regret-1 repair.

        For each missing customer, compute:
          - best insertion cost
          - second-best insertion cost
        Regret = (second - best). Customers with larger regret are handled first.
        """
        new_sol = sol.clone()

        served = {c for (c, _) in new_sol.drone_assignments} | \
                 {n for n in new_sol.truck_route if 1 <= n <= cust_num}
        missing = set(range(1, cust_num + 1)) - served

        insertion_data = {}
        for c in missing:
            options = []

            best_truck_cost = float('inf')
            best_truck_pos = None
            for i in range(1, len(new_sol.truck_route)):
                prev = new_sol.truck_route[i - 1]
                next_ = new_sol.truck_route[i]
                delta = (
                    dist_truck[prev][c] +
                    dist_truck[c][next_] -
                    dist_truck[prev][next_]
                ) * Pa.UnitTruckCost
                if delta < best_truck_cost:
                    best_truck_cost = delta
                    best_truck_pos = i
            if best_truck_pos:
                options.append(('truck', best_truck_cost, best_truck_pos))

            for s in valid_stations:
                if not problem.reachability[s][c]:
                    continue
                result = cls._calculate_drone_cost(
                    new_sol, c, s, problem, dist_truck
                )
                if result:
                    cost_components, _ = result
                    options.append(('drone', sum(cost_components.values()), s))

            if not options:
                continue

            sorted_ops = sorted(options, key=lambda x: x[1])
            best_option = sorted_ops[0]
            c_min = best_option[1]
            c_second = sorted_ops[1][1] if len(sorted_ops) > 1 else c_min
            insertion_data[c] = {
                'best': best_option,
                'regret': c_second - c_min,
                'all_options': sorted_ops
            }

        required_stations = set()
        for _, data in insertion_data.items():
            if data['best'][0] == 'drone':
                s = data['best'][2]
                if s not in new_sol.truck_route or s not in new_sol.active_stations:
                    required_stations.add(s)

        for s in required_stations:
            if s not in new_sol.truck_route:
                insert_pos = RouteManager.find_best_insertion(new_sol.truck_route, s, dist_truck)
                new_sol.truck_route.insert(insert_pos, s)
            if s not in new_sol.active_stations:
                new_sol.active_stations.add(s)

        processing_order = sorted(
            insertion_data.items(),
            key=lambda x: (-x[1]['regret'], x[0])
        )

        for c, data in processing_order:
            opt_type, _, param = data['best']

            if opt_type == 'truck':
                new_sol.truck_route.insert(param, c)
            else:
                s = param

                if s not in new_sol.truck_route or s not in new_sol.active_stations:
                    if s not in new_sol.truck_route:
                        new_sol.truck_route = RouteManager.insert_node(new_sol.truck_route, s, dist_truck)
                    new_sol.active_stations.add(s)

                existing_tasks = [k for k in new_sol.drone_assignments if k[1] == s]
                drone_id = len(existing_tasks)

                demand = problem.CustDemand[c - 1]
                dist = problem.DistDrone[s][c]
                speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)

                if speed_pair:
                    new_sol.drone_assignments[(c, s)] = (drone_id, speed_pair[0], speed_pair[1])
                else:
                    new_sol.truck_route = RouteManager.insert_node(new_sol.truck_route, c, dist_truck)

            missing.discard(c)

        for c in missing:
            RouteManager.insert_node(new_sol.truck_route, c, dist_truck)

        new_sol.truck_route = RouteManager.standardize_route(
            new_sol.truck_route, problem, new_sol
        )

        valid_active_stations = new_sol.active_stations & set(new_sol.truck_route)
        invalid_assignments = [
            (c, s) for (c, s) in new_sol.drone_assignments
            if s not in valid_active_stations
        ]
        for c, s in invalid_assignments:
            del new_sol.drone_assignments[(c, s)]
            if c not in new_sol.truck_route:
                RouteManager.insert_node(new_sol.truck_route, c, dist_truck)

        new_sol.active_stations = valid_active_stations

        truck_nodes = set(new_sol.truck_route)
        active_stations = new_sol.active_stations
        drone_stations = {s for (_, s) in new_sol.drone_assignments}

        assert active_stations.issubset(truck_nodes)
        assert drone_stations.issubset(active_stations)

        return new_sol

    @classmethod
    def collaborative_reinsert(cls, solution, problem):
        """
        Two-stage repair.

        Stage 1: assign missing customers to already active stations if feasible.
        Stage 2: insert remaining customers into the truck route.
        """
        new_sol = solution.clone()

        served = {c for (c, _) in new_sol.drone_assignments} | \
                 {n for n in new_sol.truck_route if 1 <= n <= problem.CustNum}
        missing = set(range(1, problem.CustNum + 1)) - served
        if not missing:
            return new_sol

        assigned_in_phase1 = set()

        for c in sorted(missing):
            candidate_stations = []
            for s in new_sol.active_stations:
                if not problem.reachability[s][c]:
                    continue

                demand = problem.CustDemand[c - 1]
                dist = problem.DistDrone[s][c]
                speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
                if not speed_pair:
                    continue

                energy_cost = (
                    PhysicsCalculator.calc_energy(demand, speed_pair[0], dist, False) +
                    PhysicsCalculator.calc_energy(0, speed_pair[1], dist, True)
                ) * Pa.EnergyCost
                total_cost = energy_cost + Pa.DroneDepreciationCost
                candidate_stations.append((s, total_cost, speed_pair, dist))

            if not candidate_stations:
                continue

            best_station, _, (v_out, v_back), _ = min(candidate_stations, key=lambda x: x[1])
            drone_id = sum(1 for (_, s) in new_sol.drone_assignments if s == best_station)
            new_sol.drone_assignments[(c, best_station)] = (drone_id, v_out, v_back)
            assigned_in_phase1.add(c)

        remaining = missing - assigned_in_phase1
        if remaining:
            for c in sorted(remaining):
                best_pos = RouteManager.find_best_insertion(
                    new_sol.truck_route, c, problem.DistTruck
                )
                new_sol.truck_route.insert(best_pos, c)

        new_sol.truck_route = RouteManager.standardize_route(
            new_sol.truck_route, problem, new_sol
        )

        final_served = {c for (c, _) in new_sol.drone_assignments} | \
                       {n for n in new_sol.truck_route if 1 <= n <= problem.CustNum}
        assert len(final_served) == problem.CustNum
        return new_sol

    @staticmethod
    def _calc_energy_cost(v_out, v_back, distance, demand):
        """Legacy energy-cost helper kept for compatibility."""
        W_total = (Pa.W + demand) * Pa.g
        W_empty = Pa.W * Pa.g

        vertical = (
            (Pa.k1 * W_total * (Pa.vSpeedUp / 2 + math.sqrt((Pa.vSpeedUp / 2) ** 2 + W_total / (
                    Pa.k2 ** 2))) + Pa.c2 * W_total ** 1.5) * Pa.FlightHeight / Pa.vSpeedUp +
            (Pa.k1 * W_total * (Pa.vSpeedDown / 2 + math.sqrt((Pa.vSpeedDown / 2) ** 2 + W_total / (
                    Pa.k2 ** 2))) + Pa.c2 * W_total ** 1.5) * Pa.FlightHeight / Pa.vSpeedDown +
            (Pa.k1 * W_empty * (Pa.vSpeedUp / 2 + math.sqrt((Pa.vSpeedUp / 2) ** 2 + W_empty / (
                    Pa.k2 ** 2))) + Pa.c2 * W_empty ** 1.5) * Pa.FlightHeight / Pa.vSpeedUp +
            (Pa.k1 * W_empty * (Pa.vSpeedDown / 2 + math.sqrt((Pa.vSpeedDown / 2) ** 2 + W_empty / (
                    Pa.k2 ** 2))) + Pa.c2 * W_empty ** 1.5) * Pa.FlightHeight / Pa.vSpeedDown
        )

        power_out = ((Pa.c1 + Pa.c2) * ((W_total - Pa.c5 * (v_out * math.cos(math.radians(Pa.alpha))) ** 2) ** 2 +
                                        (Pa.c4 * v_out ** 2) ** 2) ** 0.75 + Pa.c4 * v_out ** 3)
        power_back = ((Pa.c1 + Pa.c2) * ((W_empty - Pa.c5 * (v_back * math.cos(math.radians(Pa.alpha))) ** 2) ** 2 +
                                         (Pa.c4 * v_back ** 2) ** 2) ** 0.75 + Pa.c4 * v_back ** 3)
        return (vertical + power_out * distance / v_out + power_back * distance / v_back) * Pa.EnergyCost


class LocalSearch:
    """Local improvement operators used inside ALNS."""

    _show_progress = True

    @classmethod
    def set_show_progress(cls, show_progress):
        """Enable or disable optional progress printing."""
        cls._show_progress = show_progress

    @staticmethod
    def two_opt(solution, problem, max_trials=100):
        """Randomized 2-opt on the truck route."""
        new_sol = solution.clone()

        if len(new_sol.truck_route) < 4:
            return new_sol

        original_route = [
            n for n in new_sol.truck_route
            if n not in {problem.depot_start, problem.depot_end}
        ]
        if len(original_route) < 3:
            return new_sol

        best_route = original_route.copy()
        best_cost = RouteManager.calculate_truck_cost(
            [problem.depot_start] + best_route + [problem.depot_end],
            problem.DistTruck
        )

        for _ in range(max_trials):
            sample_range = list(range(1, len(original_route) - 1))
            if len(sample_range) < 2:
                continue

            i, j = sorted(random.sample(sample_range, 2))
            new_route = original_route[:i] + original_route[i:j + 1][::-1] + original_route[j + 1:]
            full_route = [problem.depot_start] + new_route + [problem.depot_end]
            current_cost = RouteManager.calculate_truck_cost(full_route, problem.DistTruck)

            if current_cost < best_cost - 1e-6:
                best_route = new_route
                best_cost = current_cost

        new_sol.truck_route = RouteManager.standardize_route(
            [problem.depot_start] + best_route + [problem.depot_end],
            problem, new_sol
        )
        return SolutionEvaluator.evaluate(new_sol, problem)

    @staticmethod
    def service_mode_switch(current_sol, problem, best_solution):
        """
        Try switching service mode for selected customers.

        For each candidate customer:
          - Drone -> truck if truck insertion is cheaper than drone cost
          - Truck -> drone if a reachable station gives lower total cost
        """
        new_sol = current_sol.clone()

        candidate_customers = set()
        current_drones = set(c for c, _ in new_sol.drone_assignments)
        best_drones = set(c for c, _ in best_solution.drone_assignments)
        candidate_customers.update(current_drones.symmetric_difference(best_drones))

        num_random = max(2, int(problem.CustNum * 0.1))
        candidate_customers.update(
            random.sample(range(1, problem.CustNum + 1), min(num_random, problem.CustNum))
        )

        switches_made = 0

        for cust in sorted(candidate_customers):
            is_drone_served = any(c == cust for (c, _) in new_sol.drone_assignments)

            if is_drone_served:
                current_station = next(s for (c, s) in new_sol.drone_assignments if c == cust)

                dist = problem.DistDrone[current_station][cust]
                demand = problem.CustDemand[cust - 1]
                _, v_out, v_back = new_sol.drone_assignments[(cust, current_station)]
                drone_energy = (
                    PhysicsCalculator.calc_energy(demand, v_out, dist, False) +
                    PhysicsCalculator.calc_energy(0, v_back, dist, True)
                ) * Pa.EnergyCost
                drone_cost = drone_energy + Pa.DroneDepreciationCost

                station_savings = 0.0
                if sum(1 for (c2, s2) in new_sol.drone_assignments if s2 == current_station) == 1:
                    station_savings += Pa.StationCost
                    if current_station in new_sol.truck_route:
                        idx = new_sol.truck_route.index(current_station)
                        if 0 < idx < len(new_sol.truck_route) - 1:
                            prev = new_sol.truck_route[idx - 1]
                            next_node = new_sol.truck_route[idx + 1]
                            current_detour = (
                                problem.DistTruck[prev][current_station] +
                                problem.DistTruck[current_station][next_node]
                            ) * Pa.UnitTruckCost
                            direct = problem.DistTruck[prev][next_node] * Pa.UnitTruckCost
                            station_savings += (current_detour - direct)

                best_truck_cost = float('inf')
                best_pos = None
                temp_route = [n for n in new_sol.truck_route if n != cust]
                for i in range(1, len(temp_route)):
                    prev = temp_route[i - 1]
                    next_node = temp_route[i]
                    insert_cost = (
                        problem.DistTruck[prev][cust] +
                        problem.DistTruck[cust][next_node] -
                        problem.DistTruck[prev][next_node]
                    ) * Pa.UnitTruckCost
                    if insert_cost < best_truck_cost:
                        best_truck_cost = insert_cost
                        best_pos = i

                cost_change = best_truck_cost - drone_cost - station_savings
                if cost_change < -1e-6:
                    del new_sol.drone_assignments[(cust, current_station)]
                    if sum(1 for (c2, s2) in new_sol.drone_assignments if s2 == current_station) == 0:
                        new_sol.active_stations.discard(current_station)
                        if current_station in new_sol.truck_route:
                            new_sol.truck_route.remove(current_station)
                    if cust not in new_sol.truck_route:
                        new_sol.truck_route.insert(best_pos, cust)
                    new_sol = SolutionEvaluator.evaluate(new_sol, problem)
                    switches_made += 1
                continue

            if cust not in new_sol.truck_route:
                continue

            idx = new_sol.truck_route.index(cust)
            if idx == 0 or idx >= len(new_sol.truck_route) - 1:
                continue

            prev = new_sol.truck_route[idx - 1]
            next_node = new_sol.truck_route[idx + 1]
            truck_current = (
                problem.DistTruck[prev][cust] +
                problem.DistTruck[cust][next_node]
            ) * Pa.UnitTruckCost
            truck_after = problem.DistTruck[prev][next_node] * Pa.UnitTruckCost
            truck_savings = truck_current - truck_after

            best_drone_cost = float('inf')
            best_station = None
            best_speeds = None

            for s in problem.valid_stations:
                if not problem.reachability[s][cust]:
                    continue

                demand = problem.CustDemand[cust - 1]
                dist = problem.DistDrone[s][cust]
                speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
                if not speed_pair:
                    continue

                energy = (
                    PhysicsCalculator.calc_energy(demand, speed_pair[0], dist, False) +
                    PhysicsCalculator.calc_energy(0, speed_pair[1], dist, True)
                ) * Pa.EnergyCost
                drone_cost = energy + Pa.DroneDepreciationCost

                activation_cost = 0.0
                if s not in new_sol.active_stations:
                    activation_cost += Pa.StationCost
                    if s not in new_sol.truck_route:
                        temp_route = [n for n in new_sol.truck_route if n != cust]
                        insert_pos = RouteManager.find_best_insertion(temp_route, s, problem.DistTruck)
                        prev_s = temp_route[insert_pos - 1]
                        next_s = temp_route[insert_pos]
                        activation_cost += (
                            problem.DistTruck[prev_s][s] +
                            problem.DistTruck[s][next_s] -
                            problem.DistTruck[prev_s][next_s]
                        ) * Pa.UnitTruckCost

                total_drone_cost = drone_cost + activation_cost
                if total_drone_cost < best_drone_cost:
                    best_drone_cost = total_drone_cost
                    best_station = s
                    best_speeds = speed_pair

            if best_station:
                cost_change = best_drone_cost - truck_savings
                if cost_change < -1e-6:
                    new_sol.truck_route.remove(cust)
                    if best_station not in new_sol.active_stations:
                        if best_station not in new_sol.truck_route:
                            pos = RouteManager.find_best_insertion(
                                new_sol.truck_route, best_station, problem.DistTruck
                            )
                            new_sol.truck_route.insert(pos, best_station)
                        new_sol.active_stations.add(best_station)

                    drone_id = sum(1 for (_, s2) in new_sol.drone_assignments if s2 == best_station)
                    new_sol.drone_assignments[(cust, best_station)] = (
                        drone_id, best_speeds[0], best_speeds[1]
                    )
                    new_sol = SolutionEvaluator.evaluate(new_sol, problem)
                    switches_made += 1

        return new_sol if switches_made > 0 else None

    @staticmethod
    def full_local_search(solution, problem, best_solution, max_depth=3):
        """Apply route 2-opt and service-mode switching repeatedly."""
        current_best = solution.clone()
        for _ in range(max_depth):
            temp_sol = LocalSearch.two_opt(current_best, problem)
            temp_sol = LocalSearch.service_mode_switch(temp_sol, problem, best_solution) or temp_sol
            if temp_sol.total_cost < current_best.total_cost:
                current_best = temp_sol.clone()
        return current_best

    @staticmethod
    def station_swap(solution, problem):
        """
        Remove one active station and reassign its customers.

        Customers are first reassigned to other active stations when feasible,
        then inserted into the truck route if still unassigned.
        """
        new_sol = solution.clone()
        if not new_sol.active_stations:
            return new_sol

        target_station = random.choice(list(new_sol.active_stations))
        affected_customers = [c for (c, s) in new_sol.drone_assignments if s == target_station]

        new_sol.active_stations.discard(target_station)
        new_sol.truck_route = [n for n in new_sol.truck_route if n != target_station]
        for c in affected_customers:
            if (c, target_station) in new_sol.drone_assignments:
                del new_sol.drone_assignments[(c, target_station)]

        unassigned = set(affected_customers)

        for c in sorted(affected_customers):
            if c not in unassigned:
                continue

            feasible_stations = []
            for s in new_sol.active_stations:
                if not problem.reachability[s][c]:
                    continue

                demand = problem.CustDemand[c - 1]
                dist = problem.DistDrone[s][c]
                speed_pair = SpeedOptimizer._find_best_speeds(demand, dist, problem)
                if speed_pair:
                    energy_cost = (
                        PhysicsCalculator.calc_energy(demand, speed_pair[0], dist, False) +
                        PhysicsCalculator.calc_energy(0, speed_pair[1], dist, True)
                    ) * Pa.EnergyCost
                    feasible_stations.append((s, energy_cost, speed_pair))

            if feasible_stations:
                best_station, _, (v_out, v_back) = min(feasible_stations, key=lambda x: x[1])
                existing_tasks = [k for k in new_sol.drone_assignments if k[1] == best_station]
                drone_id = len(existing_tasks)
                new_sol.drone_assignments[(c, best_station)] = (drone_id, v_out, v_back)
                unassigned.remove(c)

        if unassigned:
            for c in sorted(unassigned):
                insert_pos = RouteManager.find_best_insertion(
                    new_sol.truck_route, c, problem.DistTruck
                )
                new_sol.truck_route.insert(insert_pos, c)

        new_sol.truck_route = RouteManager.two_opt_swap(
            new_sol.truck_route,
            problem.DistTruck,
            problem.depot_end,
            max_trials=30
        )
        new_sol = SpeedOptimizer.optimize(new_sol, problem)
        new_sol.truck_route = RouteManager.standardize_route(
            new_sol.truck_route, problem, new_sol
        )
        new_sol = SolutionEvaluator.evaluate(new_sol, problem)
        return new_sol
