import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np

# generate signal
n_samples, dim, sigma = 1000, 1, 4
n_bkps = 4  # number of breakpoints

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
    
data = generate_normal_time_series(5, 50, 200)


# fig, ax = plt.subplots(figsize=[16, 12])
print(data.shape)
def get_ind(data):
# detection
    res =[]
    data= np.squeeze(data) 
    algo = rpt.Pelt(model="rbf").fit(data)
    ind = algo.predict(pen=10)

    for i in range(len(ind)-1):
        res.append(ind[i])
    return res    
# print((res))
res = get_ind(data)
plt.plot(range(len(data)),data[:])
plt.scatter(res,data[res],marker='p',edgecolors='r',zorder=10) 
plt.show()