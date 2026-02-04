"""
Parameters for the min-cost ALNS solver.

This file stores constants only.
"""

# Drone model constants
k1 = 0.8554
k2 = 0.3051
c1 = 2.8037
c2 = 0.3177
c4 = 0.0296
c5 = 0.0279
g = 9.8
alpha = 10
W = 1.5

# Speeds (m/s)
vSpeedUp = 10
vSpeedDown = 5
MinSpeed = 1
MaxSpeed = 30
TruckSpeed = 10

# Distance and altitude
DistFactor = 1.3       # Distance multiplier
FlightHeight = 100     # Cruise altitude

# Capacity and discretization
StaDroNum = 2          # Drones per station
SpeedLevel = 20        # Speed discretization
EnergyCap = 500000     # Energy capacity

# Costs
EnergyCost = 0.00000023773
UnitTruckCost = 0.00128
StationCost = 0
DroneDepreciationCost = 2.69

# ALNS settings
MAX_ITER = 800
DESTROY_RATE = 0.4
SA_COOLING_RATE = 0.97
SCORE_DECAY_FACTOR = 0.94
ELITE_POOL_SIZE = 5

# Search controls
NO_IMPROVE_ITER = 20
WEIGHT_UPDATE_FREQ = 35
LOCAL_SEARCH_DEPTH = 5
FINAL_SEARCH_DEPTH = 10
LOCAL_SEARCH_FREQ = 10

# Repair operator initial weights
INITIAL_REGRET_WEIGHT = 0.4
INITIAL_GREEDY_WEIGHT = 0.4
INITIAL_GLOBAL_WEIGHT = 0.2
INITIAL_OPERATOR_SCORE = 1.0

# Adaptive weight update
REWARD_SCALE = 0.5
MIN_REWARD_THRESHOLD = 0.1
MIN_SOFTMAX_TEMPERATURE = 0.1
MIN_WEIGHT_EARLY = 0.15
MAX_WEIGHT_EARLY = 0.75
MIN_WEIGHT_LATE = 0.20
MAX_WEIGHT_LATE = 0.60

# Simulated annealing
SA_TEMP_RESET_RATIO = 0.15
SA_TEMP_RESET_THRESHOLD = 0.01
SA_INITIAL_TEMP = 200

# 2-opt settings
TWO_OPT_TRIALS_FACTOR = 2
TWO_OPT_DEFAULT_TRIALS = 30

# Destroy operators
WORST_REMOVE_PROBABILITY = 0.0

# Reproducibility
RANDOM_SEED = 42
