from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pystan
plt.style.use('ggplot')

np.random.seed(1234)

mc = """
data {
    int<lower=0> n; // number of tosses
    int<lower=0> y; // number of heads
}
transformed data {}
parameters {
    real<lower=0, upper=1> p;
}
transformed parameters {}
model {
    p ~ beta(2, 2);
    y ~ binomial(n, p);
}
generated quantities {}
"""

dt = {'n': 100, 'y': 61}
sm = pystan.StanModel(model_code=mc)
fit = sm.sampling(data=dt, iter=1000, chains=1)
print(fit)

coin_dict = fit.extract()
coin_dict.keys()

fit.plot('p');
plt.tight_layout()
plt.show()