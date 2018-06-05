#Domain [-300 300]^2 km
#Origin (41.3209371228N, 289.46309961W)
#Projection Lambert
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import h5py as hp
import scipy.ndimage.filters as filter
initday = 04
inittime = 00-4+4 #Run Initialization(UTC) - 4hrs for EDT Conversion + 4hrs for integration time
#inittime = 12-4+4 #Run Initialization(UTC) - 4hrs for EDT Conversion + 4hrs for integration time
stepsize = 15 #minutes
'''
amaxits = 7
rmaxits = 7
astd = 2
rstd = 2
atol =-0.002
rtol =-0.002
'''
amaxits = 9
rmaxits = 9
astd = 1
rstd = 1
atol =-0.005
rtol =-0.005
parallelres = 7
meridianres = 7
dx = 300
dy = 300
star = [37.19838, -80.57834]

cdict = {'red':  [(0.0, 0.0000, 0.0000),
                  (0.125, 0.000, 0.000),
                  (0.5, 1.0000, 1.0000),
                  (0.875, 0.900, 0.900),
                  (1.0, 0.6000, 0.6000)],
        'green': [(0.0, 0.3270, 0.3270),
                  (0.125, 0.545, 0.545),
                  (0.5, 1.0000, 1.0000),
                  (0.875, 0.490, 0.490),
                  (1.0, 0.3270, 0.3270)],
        'blue':  [(0.0, 0.3270, 0.3270),
                  (0.125, 0.545, 0.545),
                  (0.5, 1.0000, 1.0000),
                  (0.875, 0.000, 0.000),
                  (1.0, 0.0000, 0.0000)]}
plt.register_cmap(name='CyOrDark', data=cdict)

#xlen = 201
#ylen = 197
xlen = 259
ylen = 257
tlen = 5
dim = [tlen,xlen,ylen]
mydata = np.genfromtxt('myfile.csv', delimiter=',')
print "Data is in"
# Organize velocity data
uvar = mydata[:,0]
vvar = mydata[:,1]
del mydata
u = np.empty(dim)
v = np.empty(dim)

index = 0
for t in range(dim[0]):
    for y in range(dim[2]):
        for x in range(dim[1]):
            u[t,x,y] = uvar[index]
            v[t,x,y] = vvar[index]
            index+=1
del uvar, vvar
#Simplectic Matrix
J = np.array([[0, 1], [-1, 0]])

#Plot set up
#plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
fig, ax = plt.subplots(figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
origin = [37.208, -80.5803]

#plt.rcParams.update({'axes.titlesize': 'large'})

print "Begin Map"
''
m = Basemap(width=77700,height=76800,\
    rsphere=(6378137.00,6356752.3142),\
    resolution='f',area_thresh=0.,projection='lcc',\
    lat_1=35.,lat_0=origin[0],lon_0=origin[1])#,ax=ax)
'''
m = Basemap(width=300000,height=294000,\
    rsphere=(6378137.00,6356752.3142),\
    resolution='f',area_thresh=0.,projection='lcc',\
    lat_1=35.,lat_0=origin[0],lon_0=origin[1])#,ax=ax)
'''
m.drawcoastlines(linewidth=2.0)
m.drawcountries()
m.drawparallels(np.linspace(m.llcrnrlat,m.urcrnrlat,parallelres),labels=[True,False,False,False])
m.drawmeridians(np.linspace(m.llcrnrlon,m.urcrnrlon,meridianres),labels=[False,False,False,True])
m.drawstates()

print "Preparing Repulsion Rate"
A = np.empty(dim)
for t in range(dim[0]):
    print t
    duy, dux = np.gradient(u[t,:,:],dx,dy,edge_order=2)
    dvy, dvx = np.gradient(v[t,:,:],dx,dy,edge_order=2)
    for i in range(dim[1]):
        for j in range(dim[2]):
            Utemp = np.array([u[t, i, j], v[t, i, j]])
            Grad = np.array([[dux[i, j], duy[i, j]], [dvx[i, j], dvy[i, j]]])
            S = 0.5*(Grad + np.transpose(Grad))
            A[t, i, j] = np.dot(Utemp, np.dot(np.dot(np.transpose(J), np.dot(S, J)), Utemp))/np.dot(Utemp, Utemp)

colormin = A.min()
colormax = A.max()
print colormin

colorlevel = 2/3.0*np.min(np.fabs([colormin,colormax]))
#colorlevel = 2
x = np.linspace(0, m.urcrnrx, dim[1])
y = np.linspace(0, m.urcrnry, dim[2])
xx, yy = np.meshgrid(x, y)
repulsionquad = m.pcolormesh(xx, yy, np.transpose(A[0,:,:]),vmin=-colorlevel,\
        vmax=colorlevel,shading='gouraud',cmap='CO')
velquiver = m.quiver(xx[::30,::30],yy[::30,::30],u[0,::30,::30],v[0,::30,::30],linewidth=0.5)
qk = plt.quiverkey(velquiver, 0.9*m.urcrnrx, 1.015*m.urcrnry, 10, r'$10 m/s$', labelpos='E', coordinates='data', fontproperties={'size': '18'})
clb = plt.colorbar(repulsionquad,fraction=0.037, pad=0.02) #shrink=0.5,pad=0.2,aspect=10)
clb.ax.set_title('$\\dot{\\rho}$',size = 18)
print "Begin Loop"
for t in range(dim[0]):
    repulsionquad.set_array(np.ravel(np.transpose(A[t,:,:])))
    velquiver.set_UVC(u[t,::30,::30],v[t,::30,::30])
    #qk = plt.quiverkey(velquiver, 0, 0, 12000, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
    #plt.quiverkey(velquiver, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='N', coordinates='figure')
    minute = stepsize * t
    h, minute = divmod(minute,60)
    x, y = m(star[1],star[0])
    m.scatter(x,y,marker='*',color='g',s=20*16)
    plt.annotate('LIDAR',xy=(x-0.05*x,y+0.03*y),size=15)
    plt.title("Repulsion Rate, 9-{0}-2017 {1:02d}{2:02d} EDT".format(initday+(inittime+h)//24, (inittime+h)%24, minute),fontsize=18)
    plt.savefig('MV_lcs_{0:04d}.tif'.format(t), transparent=False, bbox_inches='tight')