import math
import random
import parameters as Pa
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


class Solution:
    """Container for a combined truck route + drone assignments solution."""

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
        self.time_breakdown = {'truck_time': 0.0, 'drone_time': 0.0}
        self.flight_stats = {
            'total_drone_tasks': 0,
            'active_stations_count': 0,
            'avg_flight_distance': 0.0,
            'max_flight_distance': 0.0,
            'max_task_out_speed': 0.0,
            'max_task_back_speed': 0.0
        }

    def clone(self):
        """Deep-copy the solution object."""
        new_sol = Solution()
        new_sol.truck_route = list(self.truck_route)
        new_sol.drone_assignments = dict(self.drone_assignments)
        new_sol.active_stations = set(self.active_stations)
        new_sol.total_cost = self.total_cost
        new_sol.cost_breakdown = dict(self.cost_breakdown)
        new_sol.total_time = self.total_time
        new_sol.time_breakdown = dict(self.time_breakdown)
        new_sol.flight_stats = dict(self.flight_stats)
        return new_sol


class PhysicsCalculator:
    """Energy/time feasibility and energy computation helpers."""

    @staticmethod
    def is_reachable(problem, station, cust):
        """Fast reachability check using precomputed matrix."""
        return problem.reachability[station][cust]

    @staticmethod
    def is_reachable_static(demand, dist):
        """Check if a round trip can satisfy the energy budget."""
        W_total = (Pa.W + demand) * Pa.g
        W_empty = Pa.W * Pa.g

        vertical = (
            PhysicsCalculator._calc_vertical_energy(W_total, Pa.vSpeedUp)
            + PhysicsCalculator._calc_vertical_energy(W_total, Pa.vSpeedDown)
            + PhysicsCalculator._calc_vertical_energy(W_empty, Pa.vSpeedUp)
            + PhysicsCalculator._calc_vertical_energy(W_empty, Pa.vSpeedDown)
        )

        min_v_out = dist / (
            (Pa.EnergyCap - vertical) / PhysicsCalculator._get_power(demand, Pa.MinSpeed)
        )
        min_v_back = dist / (
            (Pa.EnergyCap - vertical) / PhysicsCalculator._get_power(0, Pa.MinSpeed)
        )
        return min_v_out <= Pa.MaxSpeed and min_v_back <= Pa.MaxSpeed

    @staticmethod
    def _calc_vertical_energy(weight, speed):
        """Vertical flight energy for a single climb/descend leg."""
        term = Pa.k1 * weight * (
            speed / 2 + math.sqrt((speed / 2) ** 2 + weight / (Pa.k2 ** 2))
        ) + Pa.c2 * (weight) ** 1.5
        return term * Pa.FlightHeight / speed

    @staticmethod
    def _get_power(payload, speed):
        """Horizontal flight power consumption."""
        W = (Pa.W + payload) * Pa.g
        alpha_rad = math.radians(Pa.alpha)
        term1 = W - Pa.c5 * (speed * math.cos(alpha_rad)) ** 2
        term2 = Pa.c4 * speed ** 2
        return (Pa.c1 + Pa.c2) * (term1 ** 2 + term2 ** 2) ** 0.75 + Pa.c4 * speed ** 3

    _vertical_cache = {}

    @classmethod
    def _get_vertical(cls, weight, is_return):
        """Cached vertical energy for a given weight."""
        key = (round(weight, 3), is_return)
        if key not in cls._vertical_cache:
            speed_up = cls._calc_vertical_energy(weight, Pa.vSpeedUp)
            speed_down = cls._calc_vertical_energy(weight, Pa.vSpeedDown)
            cls._vertical_cache[key] = speed_up + speed_down
        return cls._vertical_cache[key]

    @staticmethod
    @lru_cache(maxsize=10000)
    def calc_energy(payload, speed, distance, is_return):
        """Total energy for one leg (vertical + horizontal)."""
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
    """Truck-route insertion and local improvement utilities."""

    @staticmethod
    def _calc_truck_insertion_cost(route, node, dist_matrix):
        """Incremental truck cost if inserting node at best position."""
        best_pos = RouteManager.find_best_insertion(route, node, dist_matrix)
        prev = route[best_pos - 1]
        next_ = route[best_pos]
        delta = dist_matrix[prev][node] + dist_matrix[node][next_] - dist_matrix[prev][next_]
        return delta * Pa.UnitTruckCost

    @staticmethod
    def find_best_insertion(route, node, dist_matrix):
        """Return insertion position that minimizes added distance."""
        best_pos, min_cost = 1, float('inf')
        for i in range(1, len(route)):
            prev, next_ = route[i - 1], route[i]
            delta = dist_matrix[prev][node] + dist_matrix[node][next_] - dist_matrix[prev][next_]
            if delta < min_cost:
                best_pos, min_cost = i, delta
        return best_pos

    @classmethod
    def insert_node(cls, route, node, dist_matrix):
        """Insert node into route using best insertion position."""
        if node in route:
            return route
        pos = cls.find_best_insertion(route, node, dist_matrix)
        route.insert(pos, node)
        return route

    @staticmethod
    def standardize_route(route, problem, solution):
        """Remove invalid nodes and enforce depot start/end."""
        cleaned = [problem.depot_start]
        prev_node = problem.depot_start
        for node in route:
            if node in {problem.depot_start, problem.depot_end}:
                continue
            valid = (
                (node in solution.active_stations and node in problem.valid_stations)
                or (node in range(1, problem.CustNum + 1))
            )
            if valid and node != prev_node:
                cleaned.append(node)
                prev_node = node
        cleaned.append(problem.depot_end)
        return cleaned

    @staticmethod
    def insertion_cost(route, node, dist_matrix, depot_end):
        """Insertion score with short look-ahead and balance penalty."""
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
            for j in range(1, 6):
                if i + j < route_len:
                    look_ahead += dist_matrix[node][route[i + j]] * (0.6 ** j)
            for j in range(1, 4):
                if i - j >= 0:
                    look_ahead += dist_matrix[route[i - j]][node] * (0.4 ** j)

            if next_node == depot_end:
                look_ahead += dist_matrix[node][depot_end] * 0.5

            existing_left = sum(dist_matrix[a][b] for a, b in zip(route[:i], route[1:i]))
            existing_right = total_length - existing_left - dist_matrix[prev][next_node]
            new_left = existing_left + dist_matrix[prev][node]
            new_right = existing_right + dist_matrix[node][next_node]
            balance_penalty = abs(new_left / (new_left + new_right + 1e-6) - 0.5) * 2

            total_delta = delta + look_ahead + balance_penalty * 0.3
            min_cost = min(min_cost, total_delta)

        return min_cost

    @staticmethod
    def calculate_truck_cost(route, dist_matrix):
        """Total truck travel cost along the route."""
        return sum(dist_matrix[a][b] * Pa.UnitTruckCost for a, b in zip(route[:-1], route[1:]))

    @classmethod
    def two_opt_swap(cls, route, dist_matrix, depot_end, max_trials=None):
        """Randomized 2-opt improvement for the truck route."""
        if len(route) < 4:
            return route.copy()

        best_route = route.copy()
        best_cost = cls.calculate_truck_cost(route, dist_matrix)
        max_trials = max_trials or int(math.sqrt(len(route))) * 2

        for _ in range(max_trials):
            valid_indices = [
                i for i in range(1, len(route) - 1)
                if route[i] not in {0, depot_end}
            ]
            if len(valid_indices) < 2:
                break
            i, j = sorted(random.sample(valid_indices, 2))
            new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            new_cost = cls.calculate_truck_cost(new_route, dist_matrix)
            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost

        return best_route

    @staticmethod
    def detect_subtours(route, depot_start, depot_end):
        """Detect illegal subtours caused by repeated nodes."""
        subtours = []
        visited = set()
        current_tour = []
        node_to_tour = {}

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
    """Speed search for each assigned drone task under energy capacity."""

    @staticmethod
    def optimize(sol, problem):
        """Parallel speed optimization for all current drone tasks."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for (c, s) in list(sol.drone_assignments.keys()):
                futures.append(executor.submit(
                    SpeedOptimizer._optimize_single_task, sol, c, s, problem
                ))
            for future in futures:
                future.result()
        return sol

    @staticmethod
    def _optimize_single_task(sol, c, s, problem):
        """Minimize energy by choosing (v_out, v_back) from speed levels."""
        demand = problem.CustDemand[c - 1]
        dist = problem.DistDrone[s][c]
        speed_levels = problem.speed_levels

        best_energy = float('inf')
        best_speeds = None

        for v_out in reversed(speed_levels):
            for v_back in reversed(speed_levels):
                energy = (
                    PhysicsCalculator.calc_energy(demand, v_out, dist, False)
                    + PhysicsCalculator.calc_energy(0, v_back, dist, True)
                )
                if energy <= Pa.EnergyCap and energy < best_energy:
                    best_energy = energy
                    best_speeds = (v_out, v_back)

        if best_speeds:
            sol.drone_assignments[(c, s)] = (sol.drone_assignments[(c, s)][0], *best_speeds)
        else:
            del sol.drone_assignments[(c, s)]
            RouteManager.insert_node(sol.truck_route, c, problem.DistTruck)

    @staticmethod
    def _find_best_speeds(demand, dist, problem):
        """Return best (v_out, v_back) or None if infeasible."""
        if demand < 0 or dist <= 0:
            return None

        best_energy = float('inf')
        best_speeds = None

        for v_out in reversed(problem.speed_levels):
            for v_back in reversed(problem.speed_levels):
                if not (Pa.MinSpeed <= v_out <= Pa.MaxSpeed and Pa.MinSpeed <= v_back <= Pa.MaxSpeed):
                    continue
                energy = (
                    PhysicsCalculator.calc_energy(demand, v_out, dist, False)
                    + PhysicsCalculator.calc_energy(0, v_back, dist, True)
                )
                if energy <= Pa.EnergyCap and energy < best_energy:
                    best_energy = energy
                    best_speeds = (v_out, v_back)

        return best_speeds if best_energy <= Pa.EnergyCap else None

    @staticmethod
    def get_min_energy(demand, dist, problem):
        """Return minimal energy over all speed pairs, or inf if infeasible."""
        best_speeds = SpeedOptimizer._find_best_speeds(demand, dist, problem)
        if not best_speeds:
            return float('inf')
        v_out, v_back = best_speeds
        return (
            PhysicsCalculator.calc_energy(demand, v_out, dist, False)
            + PhysicsCalculator.calc_energy(0, v_back, dist, True)
        )

    @staticmethod
    def _handle_invalid_assignment(sol, cust, station, problem):
        """Remove infeasible assignment and clean station usage."""
        del sol.drone_assignments[(cust, station)]
        if cust not in sol.truck_route:
            RouteManager.insert_node(sol.truck_route, cust, problem.DistTruck)
        if not any(k[1] == station for k in sol.drone_assignments):
            sol.active_stations.discard(station)
            if station in sol.truck_route:
                sol.truck_route.remove(station)


class SolutionEvaluator:
    """Compute solution costs, times, and summary statistics."""

    @staticmethod
    def evaluate(sol, problem):
        """Update cost/time fields and return the evaluated solution."""
        sol.truck_route = RouteManager.standardize_route(sol.truck_route, problem, sol)

        truck_cost = RouteManager.calculate_truck_cost(sol.truck_route, problem.DistTruck)
        energy_cost = SolutionEvaluator._calculate_energy_cost(sol, problem)
        depreciation_cost = len(sol.drone_assignments) * Pa.DroneDepreciationCost

        truck_time = sum(
            problem.DistTruck[a][b] / Pa.TruckSpeed
            for a, b in zip(sol.truck_route[:-1], sol.truck_route[1:])
        )
        drone_time = SolutionEvaluator._calculate_drone_time(sol, problem)

        sol.cost_breakdown = {
            'truck_cost': round(truck_cost, 6),
            'station_cost': 0.0,
            'energy_cost': round(energy_cost, 6),
            'depreciation_cost': round(depreciation_cost, 6)
        }
        sol.total_cost = round(sum(sol.cost_breakdown.values()), 6)

        sol.time_breakdown = {
            'truck_time': round(truck_time, 6),
            'drone_time': round(drone_time, 6)
        }
        sol.total_time = max(truck_time, drone_time)
        sol.flight_stats = SolutionEvaluator._calculate_flight_stats(sol, problem)
        return sol

    @staticmethod
    def _calculate_energy_cost(sol, problem):
        """Total drone energy cost across all assignments."""
        energy_cost = 0.0
        for (c, s), (_, v1, v2) in sol.drone_assignments.items():
            dist = problem.DistDrone[s][c]
            energy_cost += (
                PhysicsCalculator.calc_energy(problem.CustDemand[c - 1], v1, dist, False)
                + PhysicsCalculator.calc_energy(0, v2, dist, True)
            ) * Pa.EnergyCost
        return energy_cost

    @staticmethod
    def _calculate_drone_time(sol, problem):
        """Return makespan of drone operations (max task time)."""
        max_time = 0.0
        for (c, s), (_, v_out, v_back) in sol.drone_assignments.items():
            dist = problem.DistDrone[s][c]
            time = (
                2 * (Pa.FlightHeight / Pa.vSpeedUp + Pa.FlightHeight / Pa.vSpeedDown)
                + dist / v_out + dist / v_back
            )
            max_time = max(max_time, time)
        return max_time

    @staticmethod
    def _calculate_flight_stats(sol, problem):
        """Aggregate distance/speed stats for reporting."""
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
