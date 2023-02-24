

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

#import
import numpy as np

##Implementation of the line search algorithm using the secant method
#arguments: initial step sizes (a0, a1), function (g_prime)
"""
def linesearch_secant(a0, a1, g_prime, grad, hessian, x):
    print("Entered linesearch_secant func")
    max_iter = 5
    alpha = [0] * max_iter
    alpha[0] = a0
    alpha[1] = a1
    for k in range(1, max_iter - 1):
        alpha[k+1] = alpha[k] - (g_prime(alpha[k], x, grad, hessian)) / ((alpha[k] - alpha[k-1]) / g_prime(alpha[k], x, grad, hessian) - g_prime(alpha[k-1], x, grad, hessian))

    print("Current alpha: ")
    print(alpha)
    return alpha
"""
def linesearch_secant(a0, a1, g_prime, grad, hessian, x):
    max_iter = 1000
    alpha = [0] * max_iter
    alpha[0] = a0
    alpha[1] = a1
    for k in range(1, max_iter - 1):
        if abs(alpha[k] - alpha[k-1]) > 1e-5 and abs(g_prime(alpha[k], x, grad)) > 1e-5:
            alpha[k+1] = alpha[k] - g_prime(alpha[k], x, grad)*(alpha[k] - alpha[k-1])/(g_prime(alpha[k], x, grad) - g_prime(alpha[k-1], x, grad))
            #print(alpha[k+1])
        else:
            print("Current alpha: ")
            print(alpha[k])
            print(g_prime(alpha[k], x, grad))
            return alpha[k]

    #print("Current alpha: ")
    #print(alpha[max_iter-1])
    #print(g_prime(alpha[max_iter-1], x, grad))
    return alpha[max_iter-1]


##Implementation of the steepest descent algorithm using the secant method
#arguments: initial search point (x), function gradient (grad), maximum iterations (max_iter), 
# tolerance (tol), function (f), function for line search (g_prime), initial step sizes (a0, a1)
def grad_desc(x, grad, hessian, max_iter, tol, f, g_prime, a0, a1):
    A = [[0, 0]] * max_iter
    A[0] = x
    k = 1
    while abs(f(A[k-1])) > tol and k < max_iter:
        alpha = linesearch_secant(a0, a1, g_prime, grad, hessian, x)
        A[k] = A[k-1] - alpha * grad(A[k-1])
        x = np.array(A[k])
        print("Iteration: " + str(k))
        print(A[k])
        k += 1
    
   
    return A[k-1]



#Rosenbrock's function

"""
def g_prime(alpha, x, grad, hessian):
    output = (grad(x))**2 * hessian(x - alpha*grad(x))
    return output
"""

def g_prime(alpha, x, grad):
    output = np.dot(-grad(x),grad(x-alpha*grad(x)))
    return output


def f(x):
    output = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    return output

def grad(x):
    output = np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, 200*(x[1] - x[0]**2)])
    return output

def hessian(x):
    output = np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]], dtype=object)
    return output


print(grad_desc(np.array([2, 2]), grad, hessian, 10, 1e-16, f, g_prime, 0.001, 0.0009))

#print(grad_desc(np.array([2, 2]), grad, hessian, 100, 1e-9, f, g_prime, 0.01, 0.011))

#alpha = linesearch_secant(0.001, 0.0009, g_prime, grad, hessian, np.array([2, 2]))
