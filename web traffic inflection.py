# this program divides data into 2 parts, before and after inflection, and thus uses 2 models,one for each part.
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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

# divide the graph in two parts: one before and one after inflection, propose separate models for both.
inflection = int(3.5 * 7 * 24)
xa = x[:inflection]
xb = x[inflection:]
ya = y[:inflection]
yb = y[inflection:]
fpa, res1, rank1, sv1, rcond1 = sp.polyfit(xa, ya, 1, full=True)
fpb, res2, rank2, sv2, rcond2 = sp.polyfit(xb, yb, 1, full=True)
fa = sp.poly1d(fpa)
fb = sp.poly1d(fpb)
fxa = sp.linspace(0, xa[-1], 1000)
fxb = sp.linspace(xa[-1], xb[-1], 1000)
plt.plot(fxa, fa(fxa), linewidth=4)
plt.plot(fxb, fb(fxb), linewidth=4)
plt.legend(["before inflection"], loc="upper left")
plt.legend(["after inflection"], loc="upper left")
# plt.show()

print("total error: %f" % (res1 + res2))
