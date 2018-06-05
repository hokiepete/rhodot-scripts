#from mpl_toolkits.basemap import Basemap
#import scipy.interpolate as interp
#import matplotlib.pyplot as plt
import numpy as np
import h5py as hp

data = hp.File('850mb_300m_10min.hdf5')
u = data['u'][:]
v = data['v'][:]
del data

ds = 300
dim = u.shape
print dim
rhodot = np.empty(dim)
for t in range(dim[0]):
    print t
    dudy,dudx = np.gradient(u[t,:,:],ds,edge_order=2)
    dvdy,dvdx = np.gradient(v[t,:,:],ds,edge_order=2)
    J = np.array([[0, 1], [-1, 0]])
    for i in range(dim[1]):
        for j in range(dim[2]):
            Utemp = np.array([u[t, i, j], v[t, i, j]])
            Grad = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
            S = 0.5*(Grad + np.transpose(Grad))
            rhodot[t, i, j] = np.dot(Utemp, np.dot(np.dot(np.transpose(J), np.dot(S, J)), Utemp))/np.dot(Utemp, Utemp)
with hp.File('850mb_300m_10min_NAM_Rhodot_t=46-62hrs_Sept2017.hdf5','w') as savefile:
        savefile.create_dataset('rhodot',shape=rhodot.shape,data=rhodot)
        savefile.close()

#np.savez('850mb_NAM_Rhodot_t=46-62hrs_Sept2017.npz',rhodot)

'''
inittime = 06 - 4 #Minus 4 to convert to EDT
initday = 16
cdict = {'red':  [(0.0, 0.0000, 0.0000),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 1.0000, 1.0000)],
        'green': [(0.0, 0.5450, 0.5450),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 0.5450, 0.5450)],
        'blue':  [(0.0, 0.5450, 0.5450),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 0.0000, 0.0000)]}
plt.register_cmap(name='CyanOrange', data=cdict)

origin = [41.3209371228,-70.53690039]
m = Basemap(width=112000,height=96000,
            rsphere=(6378137.00,6356752.3142),\
            resolution='f',area_thresh=0.,projection='lcc',\
            lat_1=35.,lat_0=origin[0],lon_0=origin[1])

ampm = ['am', 'pm']

Data = np.zeros((317, 317, 61))
x1 = np.linspace(-474, 474, 317)
x2 = np.linspace(-474, 474, 317)
X1, X2 = np.meshgrid(x1, x2)
U = np.zeros(np.shape(X1)+(61,))
V = np.zeros(np.shape(X1)+(61,))

for a in range(61):
  DataIn = np.loadtxt('Downloads/roms{:04d}.dat'.format(a), skiprows=9)
  index = 0
  for c in range(len(x2)-1, -1, -1):
    for b in range(len(x1)-1, -1, -1):
      U[c, b, a] = DataIn[index, 0]
      V[c, b, a] = DataIn[index, 1]
      index += 1
time = np.linspace(0, 60, 61)
USpline = interp.RegularGridInterpolator((x2, x1, time), U, method='linear')
VSpline = interp.RegularGridInterpolator((x2, x1, time), V, method='linear')

xlims = [-56, 56]
ylims = [-48, 48]

for t in time:#range(61):#np.array([40, 82, 88, 187, 208])/4:
  print t
  def func(y):
    return np.array([USpline((y[0], y[1], t)), VSpline((y[0], y[1], t))])
  # U = np.reshape(DataIn[:, 0], (317, 317))
  # V = np.reshape(DataIn[:, 1], (317, 317))
  # USpline = interp.RectBivariateSpline(x2, x1, U)
  # VSpline = interp.RectBivariateSpline(x2, x1, V)
  mid.goodfigure(xlims, ylims)
  x1, x2, rho_dot = mid.repulsion_rate(func, xlims, ylims, 0.3, newfig=False, plot=False, output=True)
  xplot = np.linspace(0, m.urcrnrx, len(x1))
  yplot = np.linspace(0, m.urcrnry, len(x2))
  Xplot, Yplot = np.meshgrid(xplot, yplot)
  m.drawcoastlines()
  mesh = m.pcolormesh(Xplot, Yplot, rho_dot, cmap='CyanOrange', vmin=-2, vmax=2)
  clb = plt.colorbar(mesh)
  clb.ax.set_title('$\\dot{\\rho}$', fontsize=26, y=1.02)
  h = int(t)
  minute = int(t%1*60)
  plt.title("Repulsion Rate, {0}-{1}-17 {2:02d}{3:02d} EDT".format(8,initday+(inittime+h)//24, (inittime+h)%24, minute), fontsize=24)
  plt.savefig('Downloads/frame-{:04d}.tif'.format(int(t)), transparent=False, bbox_inches='tight')
  plt.close()
'''