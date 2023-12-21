import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x1=[.1, .5, .9]
y1=[244.087479, 344.803261, 353.150556]
#y1=[9290, 9301, 9301]

x2=[0.027027, 0.216216, 0.432432, 0.864865]
y2=[1897.340902, 567.825627, 366.628367,   244.087479]


x3=[.1,1,10,100]
y3=[9310,9531,9432,9317]
# Create the first subplot
fig1, ax1 = plt.subplots()
ax1.scatter(x1, y1)
ax1.set_xlabel("tau")
ax1.set_ylabel("Time")
ax1.set_title('Time function of tau')  # Set a title for the first plot

# Create the second subplot
fig2, ax2 = plt.subplots()
ax2.scatter(x2, y2)
ax2.set_xlabel("theta")
ax2.set_ylabel("Time")
ax2.set_title('Time function of Theta')  # Set a title for the second plot

# Create the second subplot
fig3, ax3 = plt.subplots()
ax3.scatter(x3, y3)
ax3.set_xscale('log')
ax3.set_xlabel("Lambda")
ax3.set_ylabel("Precision")
ax3.set_title('Precision function of Lambda')  # Set a title for the second plot

fig1.savefig(f'Plot tau' + ".svg", format="svg")
fig2.savefig(f'Plot Theta' + ".svg", format="svg")
fig2.savefig(f'Plot lambda eps' + ".svg", format="svg")
plt.show()
