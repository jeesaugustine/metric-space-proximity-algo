import itertools
from ortools.linear_solver import pywraplp
import sys
# for ortools refer to this link
# https://developers.google.com/optimization/lp/glop#python


def variable_constraints(solver, var):
    constraint = solver.Constraint(0, 1)
    constraint.SetCoefficient(var, 1)


def set_edge_constraint(solver, edge, val):
    constraint = solver.Constraint(val, val)
    constraint.SetCoefficient(edge, 1)


def tri_constraint_setter(solver, var1, var2, var3):
    constraint = solver.Constraint(-solver.infinity(), 0)
    constraint.SetCoefficient(var1, 1)
    constraint.SetCoefficient(var2, -1)
    constraint.SetCoefficient(var3, -1)

    constraint1 = solver.Constraint(-solver.infinity(), 0)
    constraint1.SetCoefficient(var1, -1)
    constraint1.SetCoefficient(var2, 1)
    constraint1.SetCoefficient(var3, -1)

    constraint2 = solver.Constraint(-solver.infinity(), 0)
    constraint2.SetCoefficient(var1, -1)
    constraint2.SetCoefficient(var2, -1)
    constraint2.SetCoefficient(var3, 1)


def LinearProgrammingExample(order_val, g, var_point1, var_point2, minFlag):
    solver = pywraplp.Solver.CreateSolver('linear_programming_examples', 'GLOP')

    l = list(itertools.combinations(range(order_val), 2))
    vars = {}
    for i, each in enumerate(l):
        vars['x_' + str(each[0]) + str(each[1])] = (solver.NumVar(0, 1, 'x_' + str(each[0]) + str(each[1])))

    for edge in g:
        set_edge_constraint(solver=solver, edge= vars['x_' + str(edge[0]) + str(edge[1])], val=g[edge])

    for tri in list(itertools.combinations(range(order_val), 3)):
        tri_constraint_setter(solver=solver, var1=vars['x_' + str(tri[0]) + str(tri[1])],
                                          var2=vars['x_' + str(tri[1]) + str(tri[2])],
                                           var3=vars['x_' + str(tri[0]) + str(tri[2])])

    # Objective function: I am confused
    objective = solver.Objective()
    objective.SetCoefficient(vars['x_' + var_point1], 1)
    objective.SetCoefficient(vars['x_' + var_point2], -1)
    objective.SetMinimization()

    # if minFlag:
    #     print("Minimization is set")
    #     objective.SetMinimization()
    # else:
    #     print("Maximization is set")
    #     objective.SetMaximization()

    # Solve the system.
    solver.Solve()
    print("Var 1:{}, Val: {}".format('x_' + var_point1, vars['x_' + var_point1].solution_value()))
    print("Var 2:{}, Val: {}".format('x_' + var_point2, vars['x_' + var_point2].solution_value()))
    print(vars['x_' + var_point1].solution_value() - vars['x_' + var_point2].solution_value())

    # for edge in vars:
    #     print("edge: {}, value (UB): {}".format(edge, vars[edge].solution_value()))
    # print(vars['x_' + var_point].solution_value())
    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())


def main():
    g = {(0, 1): 0.2, (1, 3): 0.8, (1, 5): 0.2, (2, 4): 0.2, (2, 3): 0.1, (2, 4): 0.2, (2, 5): 0.5, (3, 4): 0.1, (5, 6): 0.2}
    first_end1 = str(sys.argv[1])
    first_end2 = str(sys.argv[2])
    var_point1 = first_end1 + first_end2
    minFlag = int(sys.argv[3])
    second_end1 = str(sys.argv[4])
    second_end2 = str(sys.argv[5])
    var_point2 = second_end1 + second_end2
    print(var_point1, var_point2)

    LinearProgrammingExample(order_val=7, g=g, var_point1=var_point1, var_point2=var_point2, minFlag=minFlag)


if __name__ == "__main__":
    main()
