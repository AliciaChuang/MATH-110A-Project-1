#7.11
"""
Line search algorithm using the secant method 
The arguments to this function are the name of the M-file for the gradient, the current point, and the search direction. 
For example, the function may be called linesearch_secant and be used by the function call alpha=linesearch_secant(‘grad’,x,d), 
where grad.m is the M-file containing the gradient, 
× is the starting line search point, 
d is the search direction, 
and alpha is the value returned by the function 
Stopping criterion |d ∇f(x + αd)| ≥ ε|d ∇f(x)|, 
where ε > 0 is a prespecified number, 
∇f is the gradient, 
x is the starting line search point, 
and d is the search direction
“Note: In the solutions manual, we used the stopping criterion |d ∇f(x + αd)| ≥ ε|d ∇f(x)|, 
We used a value of ε = 10−4 and initial conditions of 0 and 0.001.”
"""

#8.25
"""
steepest descent algorithm using the secant method for the line search
For the stopping criterion, use the condition ||g(k)|| ≤ ε, where ε = 10−6. 
Test your program by comparing the output with the numbers in Example 8.1. 
Also test your program using an initial condition of [−4,5,1], 
and determine the number of iterations required to satisfy the stopping criterion. 
Evaluate the objective function at the final point to see how close it is to 0.
"""

#import numpy so we can work with matrices
import numpy as np


##Implementation of the line search algorithm using the secant method
#parameters: initial step sizes (a0, a1), function (g_prime), and the current guess for the linesearch_secant function (x)
def linesearch_secant(a0, a1, g_prime, x):
    max_iter = 1000 #setting the maximum iteration, so we don't run into infinite recursion
    alpha = [0] * max_iter #initializing the list of alphas
    alpha[0] = a0 
    alpha[1] = a1

    #recur and update minimizer alpha max_iter times 
    for k in range(1, max_iter - 1):
        #checking if we have reached convergence
        if abs(alpha[k] - alpha[k-1]) > 1e-5:
            #secant method formula for updating minimizer
            #want to drive g_prime to 0 in order to minimize g
            alpha[k+1] = alpha[k] - ((g_prime(alpha[k], x)*(alpha[k] - alpha[k-1]))/(g_prime(alpha[k], x) - g_prime(alpha[k-1], x)))
        #if convergence is reached, return the minimizer
        else:
            return alpha[k]

    #return the minimizer if maximum iteration reached without convergence
    return alpha[max_iter-1]


##Testing of the linesearch_secant function
print("Testing of the linesearch_secant function:")
f = lambda alpha, x: alpha**2 - 20
minimizer = linesearch_secant(2, 8, f, np.array([1, 1]))
print("expected value: " + str(4.4721359553))
print("function result: " + str(minimizer))

 

##Implementation of the steepest descent algorithm using the secant method
#parameters: initial search point (x), function gradient (grad), maximum iterations (max_iter), 
# tolerance (tol), function (f), function for line search (g_prime), initial step sizes (a0, a1)
def grad_desc(x, grad, max_iter, tol, f, g_prime, a0, a1):
    A = [[0, 0]] * max_iter #initialize the matrix of minimizer values
    A[0] = x 
    k = 1
    #repeat the gradient descent process until convergence of maximum iterations reached
    while abs(f(A[k-1])) > tol and k < max_iter:
        #call the linesearch_secant function of obtain the optimal step size
        alpha = linesearch_secant(a0, a1, g_prime, A[k-1])
        #update the minimizer with the step size in the direction of steepest descent
        A[k] = A[k-1] - alpha * grad(A[k-1])
        #printing out intermediate results
        if k % 20 == 0:
            print(A[k]) 
        #update x with the last guessed minimizer value
        x = np.array(A[k])
        #update k to go to the next iteration
        k += 1

    print("steps to convergence: " + str(k-1))
    return A[k-1]




#Rosenbrock's function
#derivative of g, function to minimize, used for secant method
def g_prime0(alpha, x):
    output = np.dot(-grad0(x),grad0(x-(alpha*grad0(x))))
    return output

#rosenbrock's function's function definition 
def f0(x):
    output = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    return output

#gradient of rosenbrock's function
def grad0(x):
    output = np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, 200*(x[1] - x[0]**2)])
    return output


print("\nRosenbrock's function:")
minimizer_rf = grad_desc(np.array([1.5, 2]), grad0, 10000, 1e-9, f0, g_prime0, 0.01, 0.011)
print("expected value: " + str([1,1]))
print("function result: " + str(minimizer_rf))




#Paraboloid
import math
def g_prime1(alpha, x):
    output = np.dot(-grad1(x),grad1(x-(alpha*grad1(x))))
    return output

#paraboloid's function definition
def f1(x):
    return x[0]**2 + x[1]**2

#gradient of paraboloid function
def grad1(x):
    return(np.array([2*x[0], 2*x[1]]))


print("\nParaboloid function:")
minimizer_p = grad_desc(np.array([10, 12]), grad1, 10000, 1e-9, f1, g_prime1, 1, 1.5)
print("expected value: " + str([0,0]))
print("function result: " + str(minimizer_p))