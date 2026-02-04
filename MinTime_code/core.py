import math
import random
import parameters as Pa
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


class Solution:
    """Container for a truck-drone solution and all related metrics."""

    def __init__(self):
        self.truck_route = []
        self.drone_assignments = {}
        self.active_stations = set()
        self.total_cost = 0.0
        self.cost_breakdown = {
            'truck_cost': 0.0,
            'station_cost': 0.0,
            'energy_cost': 0.0,
            'depreciation_cost': 0.0
        }
        self.total_time = 0.0
        self.time_breakdown = {
            'truck_time': 0.0,
            'drone_time': 0.0,
            'max_station_completion': 0.0
        }
        self.flight_stats = {
            'total_drone_tasks': 0,
            'active_stations_count': 0,
            'avg_flight_distance': 0.0,
            'max_flight_distance': 0.0,
            'max_task_out_speed': 0.0,
            'max_task_back_speed': 0.0
        }
        # station -> activation time (truck arrival time at station)
        self.station_activation_times = {}
        # (station, drone_id) -> list of (cust, start, completion, end_at_station)
        self.drone_schedules = {}
        # customer_id -> service completion time
        self.customer_service_times = {}

    def clone(self):
        """Deep copy the solution object, including all metrics."""
        new_sol = Solution()
        new_sol.truck_route = list(self.truck_route)
        new_sol.drone_assignments = dict(self.drone_assignments)
        new_sol.active_stations = set(self.active_stations)
        new_sol.total_cost = self.total_cost
        new_sol.cost_breakdown = dict(self.cost_breakdown)
        new_sol.total_time = self.total_time
        new_sol.time_breakdown = dict(self.time_breakdown)
        new_sol.flight_stats = dict(self.flight_stats)
        new_sol.station_activation_times = dict(self.station_activation_times)
        new_sol.drone_schedules = {k: list(v) for k, v in self.drone_schedules.items()}
        new_sol.customer_service_times = dict(self.customer_service_times)
        return new_sol


class PhysicsCalculator:
    """Physical models for drone energy, power, and reachability."""

    @staticmethod
    def is_reachable(problem, station, cust):
        """Check if (station, customer) pair is reachable using precomputed flags."""
        return problem.reachability[station][cust]

    @staticmethod
    def is_reachable_static(demand, dist):
        """
        Static reachability check based on energy budget.

        Compute the minimum feasible speeds for outbound and return trips
        and check if they are within the allowed speed range.
        """
        W_total = (Pa.W + demand) * Pa.g
        W_empty = Pa.W * Pa.g

        # Vertical energy for takeoff and landing (loaded and empty)
        vertical = (
            PhysicsCalculator._calc_vertical_energy(W_total, Pa.vSpeedUp) +
            PhysicsCalculator._calc_vertical_energy(W_total, Pa.vSpeedDown) +
            PhysicsCalculator._calc_vertical_energy(W_empty, Pa.vSpeedUp) +
            PhysicsCalculator._calc_vertical_energy(W_empty, Pa.vSpeedDown)
        )

        # Minimum speeds implied by the remaining energy budget
        min_v_out = dist / ((Pa.EnergyCap - vertical) /
                            PhysicsCalculator._get_power(demand, Pa.MinSpeed))
        min_v_back = dist / ((Pa.EnergyCap - vertical) /
                             PhysicsCalculator._get_power(0, Pa.MinSpeed))

        return min_v_out <= Pa.MaxSpeed and min_v_back <= Pa.MaxSpeed

    @staticmethod
    def _calc_vertical_energy(weight, speed):
        """Compute vertical flight energy for a given weight and vertical speed."""
        term = Pa.k1 * weight * (
            speed / 2 + math.sqrt((speed / 2) ** 2 + weight / (Pa.k2 ** 2))
        ) + Pa.c2 * (weight) ** 1.5
        return term * Pa.FlightHeight / speed

    @staticmethod
    def _get_power(payload, speed):
        """Compute horizontal flight power for a given payload and speed."""
        W = (Pa.W + payload) * Pa.g
        alpha_rad = math.radians(Pa.alpha)
        term1 = W - Pa.c5 * (speed * math.cos(alpha_rad)) ** 2
        term2 = Pa.c4 * speed ** 2
        return (Pa.c1 + Pa.c2) * (term1 ** 2 + term2 ** 2) ** 0.75 + Pa.c4 * speed ** 3

    _vertical_cache = {}

    @classmethod
    def _get_vertical(cls, weight, is_return):
        """
        Cached vertical energy for takeoff and landing.

        Cache key uses rounded weight and a flag for loaded/empty returns.
        """
        key = (round(weight, 3), is_return)
        if key not in cls._vertical_cache:
            speed_up = cls._calc_vertical_energy(weight, Pa.vSpeedUp)
            speed_down = cls._calc_vertical_energy(weight, Pa.vSpeedDown)
            cls._vertical_cache[key] = speed_up + speed_down
        return cls._vertical_cache[key]

    @staticmethod
    @lru_cache(maxsize=10000)
    def calc_energy(payload, speed, distance, is_return):
        """Compute total drone energy (vertical + horizontal) for one leg."""
        speed = round(speed, 1)
        distance = round(distance, 2)

        if is_return:
            W = Pa.W * Pa.g
            vertical = PhysicsCalculator._get_vertical(W, True)
        else:
            W = (Pa.W + payload) * Pa.g
            vertical = PhysicsCalculator._get_vertical(W, False)

        alpha_rad = math.radians(Pa.alpha)
        cos_term = math.cos(alpha_rad)
        speed_sq = speed ** 2
        term1 = W - Pa.c5 * (speed * cos_term) ** 2
        term2 = Pa.c4 * speed_sq
        power = (Pa.c1 + Pa.c2) * (term1 ** 2 + term2 ** 2) ** 0.75 + Pa.c4 * speed ** 3
        horizontal = power * distance / speed
        return round(vertical + horizontal, 6)


class RouteManager:
    """Helper functions for truck route construction and local improvement."""

    @staticmethod
    def _calc_truck_insertion_cost(route, node, dist_matrix):
        """Compute truck insertion cost at the best position for a given node."""
        best_pos = RouteManager.find_best_insertion(route, node, dist_matrix)
        prev = route[best_pos - 1]
        next_ = route[best_pos]
        delta = dist_matrix[prev][node] + dist_matrix[node][next_] - dist_matrix[prev][next_]
        return delta * Pa.UnitTruckCost

    @staticmethod
    def find_best_insertion(route, node, dist_matrix):
        """Find best insertion position based on marginal distance increase."""
        best_pos, min_cost = 1, float('inf')
        for i in range(1, len(route)):
            prev, next_ = route[i - 1], route[i]
            delta = dist_matrix[prev][node] + dist_matrix[node][next_] - dist_matrix[prev][next_]
            if delta < min_cost:
                best_pos, min_cost = i, delta
        return best_pos

    @staticmethod
    def find_best_insertion_time(route, node, dist_matrix, truck_speed):
        """Find best insertion position based on marginal travel time."""
        best_pos, min_time_delta = 1, float('inf')
        for i in range(1, len(route)):
            prev, next_ = route[i - 1], route[i]
            time_delta = (
                dist_matrix[prev][node] +
                dist_matrix[node][next_] -
                dist_matrix[prev][next_]
            ) / truck_speed
            if time_delta < min_time_delta:
                best_pos, min_time_delta = i, time_delta
        return best_pos

    @staticmethod
    def find_best_customer_insertion(route, node, dist_matrix, active_stations):
        """
        Insert customer after the last active station.

        Search only after the last station on the route to keep
        station segments and customer segments separated.
        """
        last_sta_pos = -1
        for i, n in enumerate(route):
            if n in active_stations:
                last_sta_pos = i

        search_start = last_sta_pos + 1 if last_sta_pos >= 0 else 1
        if search_start < 1:
            search_start = 1

        best_pos, min_cost = search_start, float('inf')
        for i in range(search_start, len(route)):
            prev, next_ = route[i - 1], route[i]
            delta = dist_matrix[prev][node] + dist_matrix[node][next_] - dist_matrix[prev][next_]
            if delta < min_cost:
                best_pos, min_cost = i, delta
        return best_pos

    @classmethod
    def insert_node(cls, route, node, dist_matrix, active_stations=None):
        """
        Insert a node into the route at the best position.

        If active_stations is provided, use customer-specific insertion
        after the last station; otherwise use pure distance-based insertion.
        """
        if node in route:
            return route
        if active_stations is not None:
            pos = cls.find_best_customer_insertion(route, node, dist_matrix, active_stations)
        else:
            pos = cls.find_best_insertion(route, node, dist_matrix)
        route.insert(pos, node)
        return route

    @staticmethod
    def standardize_route(route, problem, solution):
        """
        Clean truck route and remove invalid nodes.

        Keep depot start, depot end, valid active stations, and customers,
        and remove duplicates while preserving order.
        """
        cleaned = [problem.depot_start]
        prev_node = problem.depot_start
        for node in route:
            if node in {problem.depot_start, problem.depot_end}:
                continue
            valid = (
                node in solution.active_stations and node in problem.valid_stations
            ) or (node in range(1, problem.CustNum + 1))
            if valid and node != prev_node:
                cleaned.append(node)
                prev_node = node
        cleaned.append(problem.depot_end)
        return cleaned

    @staticmethod
    def insertion_cost(route, node, dist_matrix, depot_end):
        """
        Heuristic insertion cost with look-ahead and balance penalty.

        Combines local delta, look-ahead to nearby nodes, and a simple
        left-right balance penalty to avoid very unbalanced splits.
        """
        min_cost = float('inf')
        route_len = len(route)
        total_length = (
            sum(dist_matrix[a][b] for a, b in zip(route[:-1], route[1:]))
            if route_len > 1 else 0
        )
        for i in range(1, route_len):
            prev, next_node = route[i - 1], route[i]
            delta = dist_matrix[prev][node] + dist_matrix[node][next_node] - dist_matrix[prev][next_node]
            look_ahead = 0.0

            # Forward look-ahead with exponential decay
            for j in range(1, 6):
                if i + j < route_len:
                    look_ahead += dist_matrix[node][route[i + j]] * (0.6 ** j)

            # Backward look-ahead with exponential decay
            for j in range(1, 4):
                if i - j >= 0:
                    look_ahead += dist_matrix[route[i - j]][node] * (0.4 ** j)

            # Special handling for insertion before depot end
            if next_node == depot_end:
                return_leg = dist_matrix[node][depot_end] * 0.5
                look_ahead += return_leg

            existing_left = sum(dist_matrix[a][b] for a, b in zip(route[:i], route[1:i]))
            existing_right = total_length - existing_left - dist_matrix[prev][next_node]
            new_left = existing_left + dist_matrix[prev][node]
            new_right = existing_right + dist_matrix[node][next_node]
            balance_penalty = abs(new_left / (new_left + new_right + 1e-6) - 0.5) * 2.0

            total_delta = delta + look_ahead + balance_penalty * 0.3
            min_cost = min(min_cost, total_delta)
        return min_cost

    @staticmethod
    def calculate_truck_cost(route, dist_matrix):
        """Compute truck route cost given the distance matrix and unit cost."""
        return sum(dist_matrix[a][b] * Pa.UnitTruckCost for a, b in zip(route[:-1], route[1:]))

    @classmethod
    def two_opt_swap(cls, route, dist_matrix, depot_end, max_trials=None):
        """Classic 2-opt local search on the truck route with an iteration cap."""
        if max_trials is None:
            max_trials = min(Pa.TWO_OPT_DEFAULT_TRIALS, len(route) * Pa.TWO_OPT_TRIALS_FACTOR)

        best_route = list(route)
        best_dist = sum(dist_matrix[a][b] for a, b in zip(best_route[:-1], best_route[1:]))
        improved = True
        trials = 0

        while improved and trials < max_trials:
            improved = False
            trials += 1
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    if j == depot_end or i == depot_end:
                        continue
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    new_dist = sum(dist_matrix[a][b] for a, b in zip(new_route[:-1], new_route[1:]))
                    if new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best_route

    @staticmethod
    def detect_subtours(route, depot_start, depot_end):
        """
        Detect illegal subtours in a truck route.

        Returns a list of subtours that are disconnected from the main
        depot-to-depot tour.
        """
        subtours = []
        visited = set()
        node_to_tour = {}
        current_tour = []

        for node in route:
            if node in {depot_start, depot_end}:
                if current_tour:
                    subtours.append(current_tour)
                    for n in current_tour:
                        node_to_tour[n] = len(subtours) - 1
                    current_tour = []
                continue

            if node in visited:
                if node in node_to_tour:
                    existing_tour_idx = node_to_tour[node]
                    current_tour = subtours[existing_tour_idx] + current_tour
                else:
                    split_pos = current_tour.index(node)
                    new_subtour = current_tour[split_pos:]
                    subtours.append(new_subtour)
                    for n in new_subtour:
                        node_to_tour[n] = len(subtours) - 1
                    current_tour = current_tour[:split_pos]
            else:
                current_tour.append(node)
                visited.add(node)

        if current_tour:
            subtours.append(current_tour)

        main_tour_indices = set()
        for i, tour in enumerate(subtours):
            if any(n in {depot_start, depot_end} for n in tour):
                main_tour_indices.add(i)
                for n in tour:
                    if n in node_to_tour:
                        main_tour_indices.add(node_to_tour[n])

        illegal_subtours = [
            tour for i, tour in enumerate(subtours)
            if i not in main_tour_indices and len(tour) >= 2
        ]
        return illegal_subtours


class SpeedOptimizer:
    """
    Speed optimization for drone tasks.

    Uses a discrete grid of speeds and searches from high to low to find
    the fastest feasible pair under the energy budget.
    """

    @staticmethod
    def optimize(sol, problem):
        """Optimize speeds for all drone tasks in parallel using threads."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for (c, s) in list(sol.drone_assignments.keys()):
                future = executor.submit(
                    SpeedOptimizer._optimize_single_task,
                    sol, c, s, problem
                )
                futures.append(future)
            for future in futures:
                future.result()
        return sol

    @staticmethod
    def _optimize_single_task(sol, c, s, problem):
        """Optimize speed for a single drone task or move it to truck if infeasible."""
        demand = problem.CustDemand[c - 1]
        dist = problem.DistDrone[s][c]

        best_speeds = SpeedOptimizer._find_best_speeds(demand, dist, problem)

        if best_speeds:
            sol.drone_assignments[(c, s)] = (sol.drone_assignments[(c, s)][0], *best_speeds)
        else:
            # Not enough energy: move this customer back to the truck route
            del sol.drone_assignments[(c, s)]
            RouteManager.insert_node(sol.truck_route, c, problem.DistTruck)

    @staticmethod
    def _find_best_speeds(demand, dist, problem):
        """
        Search for the fastest feasible (v_out, v_back) pair.

        Iterate from high speed to low speed and return the first pair
        that satisfies the energy constraint.
        """
        if demand < 0 or dist <= 0:
            return None

        for v_out in reversed(problem.speed_levels):
            for v_back in reversed(problem.speed_levels):
                if not (Pa.MinSpeed <= v_out <= Pa.MaxSpeed and
                        Pa.MinSpeed <= v_back <= Pa.MaxSpeed):
                    continue

                energy = (
                    PhysicsCalculator.calc_energy(demand, v_out, dist, False) +
                    PhysicsCalculator.calc_energy(0, v_back, dist, True)
                )

                if energy <= Pa.EnergyCap:
                    return (v_out, v_back)

        return None

    @staticmethod
    def get_min_energy(demand, dist, problem):
        """Return the minimum energy required among feasible speed pairs."""
        best_speeds = SpeedOptimizer._find_best_speeds(demand, dist, problem)
        if not best_speeds:
            return float('inf')
        v_out, v_back = best_speeds
        energy = (
            PhysicsCalculator.calc_energy(demand, v_out, dist, False) +
            PhysicsCalculator.calc_energy(0, v_back, dist, True)
        )
        return energy

    @staticmethod
    def _handle_invalid_assignment(sol, cust, station, problem):
        """Fallback handler for invalid drone assignments: move to truck."""
        del sol.drone_assignments[(cust, station)]
        if cust not in sol.truck_route:
            RouteManager.insert_node(sol.truck_route, cust, problem.DistTruck)
        if not any(k[1] == station for k in sol.drone_assignments):
            sol.active_stations.discard(station)
            if station in sol.truck_route:
                sol.truck_route.remove(station)


class SolutionEvaluator:
    """
    Evaluate a solution in terms of cost and time.

    Cost: truck cost, energy cost, depreciation, station cost (if any).
    Time: truck travel time, station activation times, and customer service times
    under serial drone schedules at each station.
    """

    @staticmethod
    def evaluate(sol, problem):
        """Update all cost and time fields of the solution."""
        sol.truck_route = RouteManager.standardize_route(sol.truck_route, problem, sol)

        # Cost calculation (unchanged from min-cost version)
        truck_cost = RouteManager.calculate_truck_cost(sol.truck_route, problem.DistTruck)
        energy_cost = SolutionEvaluator._calculate_energy_cost(sol, problem)
        depreciation_cost = len(sol.drone_assignments) * Pa.DroneDepreciationCost

        sol.cost_breakdown = {
            'truck_cost': round(truck_cost, 6),
            'station_cost': 0.0,
            'energy_cost': round(energy_cost, 6),
            'depreciation_cost': round(depreciation_cost, 6)
        }
        sol.total_cost = round(sum(sol.cost_breakdown.values()), 6)

        # Time calculation
        # Compute service completion times for all customers
        customer_times, station_times, schedules = SolutionEvaluator._calculate_all_customer_service_times(
            sol, problem
        )

        sol.customer_service_times = customer_times
        sol.station_activation_times = station_times
        sol.drone_schedules = schedules

        # System completion time is the maximum customer completion time
        # This matches constraint T >= t_i for all i in N_c in model B
        if customer_times:
            sol.total_time = round(max(customer_times.values()), 6)
        else:
            sol.total_time = 0.0

        # Time breakdown for diagnostic purposes
        truck_time = sum(
            problem.DistTruck[a][b] / Pa.TruckSpeed
            for a, b in zip(sol.truck_route[:-1], sol.truck_route[1:])
        )

        drone_served_customers = [c for (c, s) in sol.drone_assignments.keys()]
        if drone_served_customers:
            max_drone_time = max(
                customer_times[c]
                for c in drone_served_customers
                if c in customer_times
            )
        else:
            max_drone_time = 0.0

        sol.time_breakdown = {
            'truck_time': round(truck_time, 6),
            'drone_time': round(max_drone_time, 6),
            'max_station_completion': round(max(station_times.values()) if station_times else 0.0, 6)
        }

        sol.flight_stats = SolutionEvaluator._calculate_flight_stats(sol, problem)
        return sol

    @staticmethod
    def _calculate_energy_cost(sol, problem):
        """Compute total drone energy cost."""
        energy_cost = 0.0
        for (c, s), (_, v1, v2) in sol.drone_assignments.items():
            dist = problem.DistDrone[s][c]
            energy_cost += (
                PhysicsCalculator.calc_energy(problem.CustDemand[c - 1], v1, dist, False) +
                PhysicsCalculator.calc_energy(0, v2, dist, True)
            ) * Pa.EnergyCost
        return energy_cost

    @staticmethod
    def _calculate_all_customer_service_times(sol, problem):
        """
        Compute completion time for every customer.

        Steps:
        1) Compute truck arrival time at every node on the route.
        2) For truck-served customers, set completion time to truck arrival.
        3) For each station-drone pair, build a serial schedule of tasks
           and compute customer arrival and return times.
        """
        TASK_SORT_STRATEGY = 'ascending'

        # Step 1: truck arrival times along the route
        node_arrival_times = {}
        cumulative_time = 0.0

        for i in range(len(sol.truck_route)):
            current_node = sol.truck_route[i]
            node_arrival_times[current_node] = cumulative_time

            if i < len(sol.truck_route) - 1:
                next_node = sol.truck_route[i + 1]
                travel_time = problem.DistTruck[current_node][next_node] / Pa.TruckSpeed
                cumulative_time += travel_time

        # Step 2: truck-served customers and station activation times
        customer_service_times = {}
        station_activation_times = {}

        for node in sol.truck_route:
            if 1 <= node <= problem.CustNum:
                is_drone_served = any(c == node for (c, s) in sol.drone_assignments.keys())
                if not is_drone_served:
                    customer_service_times[node] = node_arrival_times[node]
            elif node in sol.active_stations:
                station_activation_times[node] = node_arrival_times[node]

        # Step 3: group drone tasks by (station, drone_id)
        station_drone_tasks = defaultdict(list)
        for (cust, station), (drone_id, v_out, v_back) in sol.drone_assignments.items():
            station_drone_tasks[(station, drone_id)].append({
                'customer': cust,
                'v_out': v_out,
                'v_back': v_back,
                'dist': problem.DistDrone[station][cust],
                'demand': problem.CustDemand[cust - 1]
            })

        # Step 4: serial schedule for each drone at each station
        drone_schedules = {}

        for (station, drone_id), tasks in station_drone_tasks.items():
            if station not in station_activation_times:
                continue

            station_ready_time = station_activation_times[station]
            schedules = []
            drone_current_time = station_ready_time

            tasks_sorted = SolutionEvaluator._sort_drone_tasks(tasks, TASK_SORT_STRATEGY)

            for task in tasks_sorted:
                cust = task['customer']
                v_out = task['v_out']
                v_back = task['v_back']
                dist = task['dist']

                vertical_time = (
                    Pa.FlightHeight / Pa.vSpeedUp +
                    Pa.FlightHeight / Pa.vSpeedDown
                )

                outbound_time = dist / v_out
                total_outbound = vertical_time + outbound_time

                task_start = drone_current_time
                customer_arrival = task_start + total_outbound
                customer_service_times[cust] = customer_arrival

                return_time = dist / v_back
                total_return = vertical_time + return_time

                task_end = customer_arrival + total_return

                schedules.append((cust, task_start, customer_arrival, task_end))
                drone_current_time = task_end

            drone_schedules[(station, drone_id)] = schedules

        return customer_service_times, station_activation_times, drone_schedules

    @staticmethod
    def _sort_drone_tasks(tasks, strategy='ascending'):
        """
        Sort drone tasks for scheduling.

        Supported strategies:
        - 'ascending': shortest total time first
        - 'descending': longest total time first
        - 'weighted': distance-time weighted priority
        - 'alternating': alternate between long and short tasks
        """
        tasks_with_time = []
        for task in tasks:
            vertical_time = 2 * (Pa.FlightHeight / Pa.vSpeedUp + Pa.FlightHeight / Pa.vSpeedDown)
            horizontal_time = task['dist'] / task['v_out'] + task['dist'] / task['v_back']
            total_time = vertical_time + horizontal_time

            tasks_with_time.append({
                **task,
                'total_time': total_time
            })

        if strategy == 'ascending':
            return sorted(tasks_with_time, key=lambda t: t['total_time'])

        elif strategy == 'descending':
            return sorted(tasks_with_time, key=lambda t: -t['total_time'])

        elif strategy == 'weighted':
            if len(tasks_with_time) > 0:
                max_dist = max(t['dist'] for t in tasks_with_time)
                max_time = max(t['total_time'] for t in tasks_with_time)

                for task in tasks_with_time:
                    normalized_dist = task['dist'] / max_dist if max_dist > 0 else 0.0
                    normalized_time = task['total_time'] / max_time if max_time > 0 else 0.0
                    task['priority'] = -(0.6 * normalized_dist + 0.4 * normalized_time)

                return sorted(tasks_with_time, key=lambda t: t['priority'])
            else:
                return tasks_with_time

        elif strategy == 'alternating':
            sorted_tasks = sorted(tasks_with_time, key=lambda t: t['total_time'])

            result = []
            left = 0
            right = len(sorted_tasks) - 1
            use_long = True

            while left <= right:
                if use_long:
                    result.append(sorted_tasks[right])
                    right -= 1
                else:
                    result.append(sorted_tasks[left])
                    left += 1
                use_long = not use_long

            return result

        else:
            return sorted(tasks_with_time, key=lambda t: t['total_time'])

    @staticmethod
    def _calculate_drone_time(sol, problem):
        """Compute the maximum single-task drone time over all assignments."""
        max_time = 0.0
        for (c, s), (_, v_out, v_back) in sol.drone_assignments.items():
            dist = problem.DistDrone[s][c]
            time = (
                2 * (Pa.FlightHeight / Pa.vSpeedUp + Pa.FlightHeight / Pa.vSpeedDown) +
                dist / v_out +
                dist / v_back
            )
            max_time = max(max_time, time)
        return max_time

    @staticmethod
    def _calculate_flight_stats(sol, problem):
        """Aggregate simple statistics of all drone flights."""
        if not sol.drone_assignments:
            return {
                'total_drone_tasks': 0,
                'active_stations_count': 0,
                'avg_flight_distance': 0.0,
                'max_flight_distance': 0.0,
                'max_task_out_speed': 0.0,
                'max_task_back_speed': 0.0
            }

        distances = []
        speeds_out = []
        speeds_back = []
        for (c, s), (_, v_out, v_back) in sol.drone_assignments.items():
            dist = problem.DistDrone[s][c]
            distances.append(dist)
            speeds_out.append(v_out)
            speeds_back.append(v_back)

        max_dist_idx = distances.index(max(distances))

        return {
            'total_drone_tasks': len(sol.drone_assignments),
            'active_stations_count': len(sol.active_stations),
            'avg_flight_distance': round(sum(distances) / len(distances), 2),
            'max_flight_distance': round(max(distances), 2),
            'max_task_out_speed': round(speeds_out[max_dist_idx], 2),
            'max_task_back_speed': round(speeds_back[max_dist_idx], 2)
        }
