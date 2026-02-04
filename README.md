# TSPDS-Speed-Optimization

The "MinCost_code" folder includes the code for cost minimization model.

The "MinTime_code" folder includes the code for time minimization model.

The "Instances" folder includes the test instances used for algorithm evaluation.<br><br>

Code Files Description

Both code folders contain the same five files:

main.py : Main entry point for running experiments

solver.py : ALNS algorithm implementation

operators.py : Destroy, repair, and local search operators

core.py : Data structures and energy consumption calculations

parameters.py : Algorithm and problem parameters<br><br>

Instance Data Description

File naming: {customers}_{stations}_{index}.txt

Data format:

CustNum [n] StaNum [m]

[Distance Matrix: (n+m+2) × (n+m+2)]

Customer demand

[d1] [d2] ... [dn]

Node indexing:

Node 0 : Depot

Nodes 1 to n : Customers

Nodes (n+1) to (n+m) : Candidate drone stations

Node (n+m+1) : Return depot (copy of node 0)

Customer demand: Package weight in kilograms (kg)

The distance matrix contains Euclidean distances in meters between all nodes.
