# Z = 4*x + 6*y
# -x + y <= 11
# x + y <= 27
# 2*x + 5*y <= 90
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 10000)
y1 = 11.0 + x
y2 = 27.0 - x
y3 = (90 - 2*x) / 5.0

plt.plot(x,y1,label = r'$-x+y\leq11$')
plt.plot(x,y2,label = r'$x+y\leq27$')
plt.plot(x,y3,label = r'$2x+5y\leq90$')
#plt.xlim((0, 20))
#plt.ylim((0, 40))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
y5 = np.minimum(y1, y2)
plt.fill_between(x, y3, y5, where=y3<y5, color='grey', alpha=0.5)
plt.legend(bbox_to_anchor=(0.7, 1.0), loc=2, borderaxespad=0.)

plt.grid(True)
plt.show()

# SymPy is a Python library for symbolic mathematics. 
# It aims to become a full-featured computer algebra system (CAS) while keeping the code 
# as simple as possible in order to be comprehensible and easily extensible.
import sympy as sp

x, y = sp.symbols('x y')

eqn = [y - x - 11, x+y-27]
print(sp.solve(eqn,[x,y]))

eqn = [y-x-11,2*x+5*y-90]
print(sp.solve(eqn,[x,y]))

eqn = [y+x-27,2*x+5*y-90]
print(sp.solve(eqn,[x,y]))

# ----> vertex