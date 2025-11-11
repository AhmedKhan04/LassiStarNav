import lightkurve as lk 
import matplotlib.pyplot as plt

# plot lightkurve 

search = lk.search_lightcurve("Delta Scuti")
lc = search.download()
lc.plot()
plt.show()
