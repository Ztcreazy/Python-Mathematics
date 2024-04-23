import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['ggplot'])

def gradient_descent(x, alpha=0.2, epsilon =0.001):
    x_history =[x]
    while len ( x_history ) == 1 or \
    abs ( x_history[-1] - x_history[-2] ) > epsilon :
        
        a = np.array([[4 *x **2]])
        b = np.array([4 *x **2 - 0.0002*x])
        alpha = float( ( np.linalg.solve(a, b) + (1-c2)/2 ) /2 )
        print("alpha: ", alpha)
        x = x - alpha *( 2 *x )
        x_history.append (x)

    return x_history

x=5
c1 = 0.0001
c2 = 0.9

x_history = gradient_descent(x)
print("x history: ", x_history)
fig , ax = plt . subplots ( figsize =(10, 5) )
ax.set_ylabel( 'y' )
ax.set_xlabel( 'x' )

x = np.linspace(-5,5,1000)
y=x**2

plt.plot(x,y,'b',linewidth = 3)
plt.plot(x_history , list( map( lambda x: x**2, x_history ) ) , \
'r' '.--' , markersize = 14)
plt.legend(['$x^2$'] , fontsize =15)
plt.show()