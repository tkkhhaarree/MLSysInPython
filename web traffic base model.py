import scipy as sp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
x = data[:, 0]
y = data[:, 1]
ybool = ~sp.isnan(y)
x = x[ybool]
y = y[ybool]
plt.scatter(x, y, s=10)
plt.title("web traffic stats over last month")
plt.xlabel("time")
plt.ylabel("hits per hour")
plt.xticks([w * 7 * 24 for w in range(5)], ["week %d" % w for w in range(5)])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
# plt.show()

# finding a straight line which best fits the data:
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 2, full=True)
print("model parameters (slope and intercept in this case): %s" % fp1)
print("error of approximation: ", residuals)
# creating a model (i.e line in this case) using parameters.
f1 = sp.poly1d(fp1)
print("\n\nmodel function: \n\n", f1)
fx = sp.linspace(0, x[-1], 1000)  # generate X-values for plotting
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")
#plt.show()
root = fsolve(f1 - 100000, x0=800)/(7*24)
print("weeks required to reach 100,000 hits: %f" % root)
