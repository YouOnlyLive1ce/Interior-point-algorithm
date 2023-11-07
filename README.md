# Interior-point-algorithm
# Optimization Assignment 2

## Team Members
- Amir Gubaidullin
- Ilnaz Magizov
- Ilya Krasheninnikov
- Dima Nekrasov

# INFO
The interior point algorithm is an iterative optimization approach for solving linear programming problems. It begins with an initial feasible solution within the interior of the feasible region, gradually moving towards the optimal solution through iterative steps. Using a barrier function to penalize points near the boundary of the feasible region, the algorithm follows a central path within the interior, reducing the impact of the barrier function over iterations to allow the solution to approach the boundary while converging to the optimal solution. It continues these iterations until reaching a solution that satisfies the optimality conditions within a specified tolerance, ultimately providing the optimal solution that maximizes or minimizes the objective function while adhering to all constraints within the feasible region.
## Usage
To use our code, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/YouOnlyLive1ce/Interior-point-algorithm.git
```
2. Navigate to the project directory:
```bash
cd Interior-point-algorithm
```
3. Install the required dependencies using pip:
```bash
pip install numpy
```

### Input format
The input is expected to be provided in a specific format, as follows:

```bash
Write coefficients of objective function in 1 string
Write a vector of right-hand side numbers in 1 string
Write a matrix of coefficients of constraint function
Write the approximation accuracy ϵ
```

Example:
```bash
3 2
20 10 0 0
2 1
-4 5
1 0
0 1
1
```

Example for "Method is not applicable":
```bash
3 2
20 -10 0 0
2 1
-4 5
1 0
0 1
1
```

Example for Problem does not have solution:
```bash
3 2
20 10 0 0
2 1
-4 -5
1 0
0 1
1
```


The script will output lines:
```bash
The vector of decision variables when α = 0.5
The vector of decision variables when α = 0.9
The vector of decision variables with Simplex method
Maximum value of the objective function when α = 0.5:
Maximum value of the objective function when α = 0.9:
```

### Important Notes
Ensure that the input data is provided in the specified format, with appropriate positive coefficients for the constraint functions. The method may not be applicable if these conditions are not met.
The accuracy parameter controls the number of decimal places in the output. Adjust it to your desired level of precision.

Now you're ready to use our code for solving IPA!
