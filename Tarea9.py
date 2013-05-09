
import math
import sys
import string
import cmath
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy.linalg import *

obs_data = np.loadtxt("traces.dat")
x_1_obs = obs_data[:,0]
y_1_obs = obs_data[:,1]
z_1_obs = obs_data[:,2]
x_2_obs = obs_data[:,3]
y_2_obs = obs_data[:,4]
z_2_obs = obs_data[:,5]


def likelihood(x_obs, x_model, y_obs, y_model):
    chi_squared = sum ((y_obs-y_model)**2 + (x_obs-x_model)**2)
    return -chi_squared

def my_model_x(x_1_obs, z_1_obs, z_2_obs, x_rand, z_rand):
    m1 = (x_1_obs-x_rand)/(z_1_obs-z_rand)
    x = (m1*(z_2_obs-z_rand))+x_rand

    return x

def my_model_y(y_1_obs, z_1_obs, z_2_obs, y_rand, z_rand):
    m2 = (y_1_obs-y_rand)/(z_1_obs-z_rand)
    y = (m2*(z_2_obs-z_rand))+y_rand

    return y

#List to keep the steps

x_walk = np.empty((0))
y_walk = np.empty((0))
z_walk = np.empty((0))

x_walk = np.append(x_walk, np.random.random())
y_walk = np.append(y_walk, np.random.random())
z_walk = np.append(z_walk, -np.random.random())

n_iterations = 100000

for i in range(n_iterations):
    x_p = np.random.normal(x_walk[i], 1.0)
    y_p = np.random.normal(y_walk[i], 1.0)
    z_p = np.random.normal(z_walk[i], 200.0)

    while (z_p>0 or z_p<-5000):
        z_p = np.random.normal(z_walk[i], 10.0)
  
    
    x_init = my_model_x(x_1_obs, z_1_obs, z_2_obs, x_walk[i], z_walk[i])
    y_init = my_model_y(y_1_obs, z_1_obs, z_2_obs, y_walk[i], z_walk[i])
    x_prime = my_model_x(x_1_obs, z_1_obs, z_2_obs, x_p, z_p)
    y_prime = my_model_y(y_1_obs, z_1_obs, z_2_obs, y_p, z_p)

    alpha = (likelihood(x_2_obs, y_2_obs, x_prime, y_prime))/(likelihood(x_2_obs, y_2_obs, x_init, y_init))

    if(alpha>=0.0):
        x_walk = np.append(x_walk, x_p)
        y_walk = np.append(y_walk, y_p)
        z_walk = np.append(z_walk, z_p)
    else:

        beta = np.random.random()
        if(beta<=np.exp(alpha)):
            x_walk = np.append(x_walk, x_p)
            y_walk = np.append(y_walk, y_p)
            z_walk = np.append(z_walk, z_p)
        else:
            x_walk = np.append(x_walk, x_walk[i])
            y_walk = np.append(y_walk, y_walk[i])
            z_walk = np.append(z_walk, z_walk[i])

print('Creando graficas... esto tomara unos minutos')

#graficando datos para x y x vs y
count, bins, ignored = hist(x_walk, 50, normed=True)
xlabel('x')
ylabel('function')
title('Plano_x,f')
savefig('Plano_x,f.pdf')
close()

max_x=(max(count))

scatter(x_walk, y_walk)
xlabel('x')
ylabel('y')
title('Scatter_x,y')
savefig('Scatter_x,y.pdf')
close()


H, xedges, yedges = np.histogram2d(x_walk, y_walk, bins=(50, 50))
H.shape, xedges.shape, yedges.shape
extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
imshow(H, extent=extent, interpolation='nearest')
colorbar()
xlabel('x')
ylabel('y')
title('Plano_x,y')
savefig('Histogram_x,y.pdf')
close()

#graficando datos para y y y vs z
count, bins, ignored = hist(y_walk, 50, normed=True)
xlabel('y')
ylabel('function')
title('Plano_y,f')
savefig('Plano_y,f.pdf')
close()

max_y=(max(count))

scatter(y_walk, z_walk)
xlabel('y')
ylabel('z')
title('Scatter_y,z')
savefig('Scatter_y,z.pdf')
close()


H, xedges, yedges = np.histogram2d(y_walk, z_walk, bins=(50, 50))
H.shape, xedges.shape, yedges.shape
extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
imshow(H, extent=extent, interpolation='nearest')
colorbar()
xlabel('y')
ylabel('z')
title('Plano_y,z')
savefig('Histogram_y,z.pdf')
close()

#graficando datos para z y x vs z
count, bins, ignored = hist(z_walk, 50, normed=True)
xlabel('z')
ylabel('function')
title('Plano_z,f')
savefig('Plano_z,f.pdf')
close()

max_z=(max(count))

scatter(x_walk, z_walk)
xlabel('x')
ylabel('z')
title('Scatter_x,z')
savefig('Scatter_x,z.pdf')
close()


H, xedges, yedges = np.histogram2d(x_walk, z_walk, bins=(50, 50))
H.shape, xedges.shape, yedges.shape
extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
imshow(H, extent=extent, interpolation='nearest')
colorbar()
xlabel('x')
ylabel('z')
title('Histogram_x,z')
savefig('Histogram_x,z.pdf')
close()

#exportando los mejores valores iniciales
data = open('Mejor-Parametro.dat', 'w')
data.write('Iniciales: x ='+ str(max_x)+' y ='+str(max_y)+' z ='+str(max_z))

