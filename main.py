import numpy as np
import math


def add_basis(A, c):
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                A[i].append(1.0)
            else:
                A[i].append(0.0)

    while len(c) < len(A[0]):
        c.append(0.0)

    return A, c


def get_trial_solution(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]


def make_D_matrix(x):
    D = np.full((len(x), len(x)), 0.0)
    for i in range(len(x)):
        D[i, i] = x[i]

    return D


def P_calculation(A_tilda):
    return np.eye(len(A_tilda[0])) - np.dot(np.dot(A_tilda.T, np.linalg.inv(np.dot(A_tilda, A_tilda.T))), A_tilda)


def find_v(c_P):
    v = 0
    for i in c_P:
        if i < 0:
            v = min(v, i)

    return -v


def interior_point_method(c, A, b, ap, alpha):
    x = get_trial_solution(A, b)
    x_copy = x
    flag = True

    while flag:
        x = x_copy

        D = make_D_matrix(x)

        A_tilda = np.dot(A, D)
        c_tilda = np.dot(D, c)
        P = P_calculation(A_tilda)
        c_P = np.dot(P, c_tilda)

        v = find_v(c_P)

        x_tilda = np.add(np.ones(len(c_P)), np.dot(alpha / v, c_P))
        x_copy = np.dot(D, x_tilda)

        flag = False
        for i in range(len(x)):
            if x[i] < x_copy[i] - pow(0.1, ap) or x[i] > x_copy[i] + pow(0.1, ap):
                flag = True
                break

    return x


def approx(op, accuracy):
    for k in range(len(op)):
        op[k] = round(op[k], accuracy)
    return op


def calc_final_value(op, c):
    x = 0
    for k in range(len(op)):
        x += op[k] * c[k]
    return x


def solve(c, A, b, ap, alpha):
    A_new = []
    for i in A:
        A_new.append(i.copy())
    c_new = c.copy()
    A_new, c_new = add_basis(A_new, c_new)

    answer = interior_point_method(np.array(c_new), np.array(A_new), np.array(b), ap, alpha)

    output = approx(answer, ap)
    value = round(calc_final_value(answer, c_new), ap)

    return output, value


def get_user_input():
    print("Write coefficients of objective function in 1 string:")
    C = list(map(float, input().split()))

    print("Write a vector of right-hand side numbers in 1 string:")
    b = list(map(int, input().split()))

    print("Write a matrix of coefficients of constraint function:")
    A = []
    for i in range(len(b)):
        A.append(list(map(float, input().split())))

    print("Write the approximation accuracy ϵ: ", end="")
    ap = int(input())

    for i in range(len(A)):
        flag = False
        for j in range(len(A[i])):
            if A[i][j] > 0:
                flag = True
                break
        if not flag:
            print("The problem does not have solution!")
            exit(0)

    for i in b:
        if i < 0:
            print("The method is not applicable")
            exit(0)

    return C, A, b, ap


def basic_check(column):
    return column.tolist().count(1) == 1 and column.tolist().count(0) == len(column) - 1


def solution(table):
    columns = np.array(table).T
    solutions = []

    for column in range(len(columns) - 1):
        cur_solution = 0
        if basic_check(columns[column]):
            index_of_one = columns[column].tolist().index(1)
            cur_solution = columns[-1][index_of_one]
        solutions.append(cur_solution)

    return solutions


def iteration(table, pivot_position):
    new_table = []

    for i in range(len(table)):
        new_table.append([])

    i, j = pivot_position
    pivot_value = table[i][j]
    new_table[i] = np.array(table[i]) / pivot_value

    for k in range(len(table)):
        if k != i:
            new_table[k] = np.array((table[k])) - np.array(new_table[i]) * table[k][j]

    return new_table


def get_pivot_position(table):
    row = table[-1]
    column_i = 0  # be careful

    for i in range(len(row) - 1):
        if row[i] > 0:
            column_i = i
            break

    restrictions = []


    for i in range(len(table) - 1):
        if table[i][column_i] <= 0:
            restrictions.append(math.inf)
        else:
            restrictions.append(table[i][-1] / table[i][column_i])

    row_i = restrictions.index(min(restrictions))

    return row_i, column_i


def poss_to_enhance(table):
    row = table[-1]
    found = False

    for x in range(len(row) - 1):
        if row[x] > 0:
            found = True
            break

    return found


def make_table(c, A, b):
    for i in range(len(A)):
        for j in range(len(A)):
            if j == i:
                A[i].append(1)
            else:
                A[i].append(0)

    for i in range(len(A)):
        c.append(0)

    Ab = []
    for i in range(len(A)):
        Ab.append(A[i] + [b[i]])

    last_row = c + [0]
    return Ab + [last_row]


def simplex_method(c, A, b):
    table = make_table(c, A, b)

    while poss_to_enhance(table):
        pivot_position = get_pivot_position(table)
        table = iteration(table, pivot_position)

    return solution(table)


def solve_simplex_method(c, A, b, ap):
    A_new, c_new = add_basis(A.copy(), c.copy())

    answer = simplex_method(c_new, A_new, b)

    output = approx(answer, ap)
    value = round(calc_final_value(answer, c_new), ap)

    return output, value


if __name__ == "__main__":
    c, A, b, ap = get_user_input()

    output_first, value_first = solve(c, A, b, ap, 0.5)
    output_second, value_second = solve(c, A, b, ap, 0.9)
    output_third, value_third = solve_simplex_method(c, A, b, ap)

    print("The vector of decision variables when α = 0.5:")
    for i in output_first: print(i, end=" ")
    print("\n")

    print("The vector of decision variables when α = 0.9:")
    for i in output_second: print(i, end=" ")
    print("\n")

    print("The vector of decision variables with Simplex method")
    for i in output_third: print(i, end=" ")
    print("\n")

    print("Maximum value of the objective function when α = 0.5:")
    print(value_first)
    print()

    print("Maximum value of the objective function when α = 0.9:")
    print(value_second)
