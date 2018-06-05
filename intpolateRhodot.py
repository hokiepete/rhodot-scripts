# -*- coding: utf-8 -*-
"""
Created on Wed May 09 12:51:17 2018

@author: pnola
"""

import h5py as hp
import numpy as np
import scipy.interpolate as sint
#import scipy.io as sio
import matplotlib.pyplot as plt
'''
with hp.File('850mb_300m_10min_NAM_Rhodot_t=46-62hrs_Sept2017.hdf5','r') as data:
    rhodot = data['rhodot'][:]
del data
with hp.File('850mb_NAM_gridpoints.hdf5','r') as data:
    x = data['x'][:]
    y = data['y'][:]
    t = data['t'][:]
del data
print x.shape
x0=169.4099
y0=-1043.9
points = zip(x.ravel(),y.ravel())
rhodot_origin = np.empty(t.shape)
for tt in range(len(t)):
    print tt
    rhodot_origin[tt] = sint.griddata(points,rhodot[tt,:,:].ravel(),(x0,y0),method='cubic')
    #f = sint.interp2d(x, y, rhodot[tt,:,:], kind='cubic')
    #rhodot_origin[tt] = f(x0,y0)
    #tck = sint.bisplrep(x, y, rhodot[tt,:,:], s=0)
    #rhodot_origin[tt] = sint.bisplev(x0,y0,tck)
with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=46-62hrs_Sept2017.hdf5','w') as savefile:
        savefile.create_dataset('t',shape=t.shape,data=t)
        savefile.create_dataset('rhodot',shape=rhodot_origin.shape,data=rhodot_origin)
        savefile.close()
#np.savez('850mb_NAM_Rhodot_Origin_t=46-62hrs_Sept2017.npz',t,rhodot_origin)
#f=np.load('850mb_NAM_Rhodot_Origin_t=46-62hrs_Sept2017.npz')
'''
with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=46-62hrs_Sept2017.hdf5','r') as data:
    rhodot = data['rhodot'][:]
    t = data['t'][:]
    data.close()
    
#with hp.File('simflight2000_20xhr_halfsecondres.mat','r') as data:
with hp.File('simflightdata_1000.mat','r') as data:
    htrhodot = data['rhodot'][:]
    to = data['timeout'][:]
    data.close()

with hp.File('hunterdata.mat','r') as data:
    rhodotx = data['rhodot'][:]
    tx = data['timeout'][:]

#rhodot=f['rhodot']
''
with hp.File('simflight2000_10xhr_halfsecondres.mat','r') as data:
    prhodot = data['rhodot'][:]
    po = data['timeout'][:]
    data.close()
#plt.plot(t,rhodot_origin,color='b')
#plt.plot(t,rhodot[:,125,130],color='r')
plt.close()
#ax2=plt.plot((po-46)*3600,prhodot,color='k',label="Peter's virtual flight 5x/hr")
#ax4=plt.plot((to-46)*7200,htrhodot,color='b',label="Peter's virtual flight 10x/hr")
ax4=plt.plot(to,htrhodot,color='b',label="Peter's virtual flight 10x/hr")
ax1=plt.plot(tx,rhodotx,color='y',label="Hunter's flight simulation")
ax3=plt.plot(t,rhodot,color='r',label="Rhodot")
plt.axhline(0)
plt.legend()#handles=[ax1,ax2,ax3])