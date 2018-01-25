import numpy as np
from scipy import stats

N = 10
a = np.random.randn(N) + 2
b = np.random.randn(N)

a_var = a.var(ddof=1)
b_var = b.var(ddof=1)
a_mean = a.mean()
b_mean = b.mean()

sp = np.sqrt((a_var + b_var) / 2.)

t_val = (a_mean - b_mean) / (sp * np.sqrt(2. / N))
df = 2*N - 2

p = 1 - stats.t.cdf(t_val, df=df)

print(t_val)
print(p*2)

print(stats.ttest_ind(a, b))