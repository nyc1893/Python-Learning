import numpy as np
import matplotlib.pyplot as plt
import seaborn
# from __future__ import division

def generate_normal_time_series(num, minl=50, maxl=100):
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn()*10
        var = np.random.randn()*1
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return data
    
data = generate_normal_time_series(2, 50, 200)


# fig, ax = plt.subplots(figsize=[16, 12])
print(data.shape)
# ax.plot(data)
# plt.show()


# all = 1000000
# a = np.zeros(20)
# b=0
# for i in range(20):
    # a[i] = np.power(0.95, i)
    # b += (a[i]*78533)
# print(b)
# c = b-all
# print(c)
def get_ind(data):
    data= np.squeeze(data)  
    import cProfile
    import bayesian_changepoint_detection.offline_changepoint_detection as offcd
    from functools import partial

    # def get_z_score
    Q, P, Pcp = offcd.offline_changepoint_detection(data, partial(offcd.const_prior, l=(len(data)+1)), offcd.gaussian_obs_log_likelihood, truncate=-40)
    k = np.exp(Pcp).sum(0)

# print(k.shape)
    ind = np.argmax(k)
    return ind
ind = get_ind(data)
# fig, ax = plt.subplots(figsize=[18, 16])
plt.subplot(2, 1, 1)
plt.plot(range(len(data)),data[:])
plt.scatter(ind,data[ind],marker='p',edgecolors='r',zorder=10) 
plt.subplot(2, 1, 2)
# plt.plot(range(len(k)),k)
# plt.scatter(ind,k[ind],marker='p',edgecolors='r',zorder=10)            
plt.show()
"""
"""