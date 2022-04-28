#!/usr/bin/env python
# coding: utf-8

# In[129]:


#load packages
import numpy as np
from netCDF4 import Dataset
import pickle as pickle
import scipy.stats as stats
import cartopy.crs as ccrs
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
mpl.rc('font',size=14,weight='bold') #set default font size and weight for plots


# In[60]:


#user defined inputs
#datdir1 = '/d11/ksmith/cesmdata/b.e11.B1850C5CN.f09_g16.005/ocn/interp/'
datdir1 = '/Users/Karen/Dropbox/CESM/python/CESM/'
var = 'SST'
fileend = '.allyrs.nc'
month = ['01', '02', '03']

#open netcdf files
varnames = dict()
for i in range(len(month)):
    fname1        = datdir1 + '/' + var + '.monthly.' + month[i] + fileend
    nc1           = Dataset(fname1)
    varnames_tmp1 = np.squeeze(nc1.variables[var][:,:,:])
    varnames[i]   = np.ma.masked_greater(varnames_tmp1,1e20)
    
    lon           = nc1.variables['lon'][:]
    lat           = nc1.variables['lat'][:]


# In[61]:


#create SSW seasons (oct-apr) (also restrict latitudes to NH only lat[1:121])
var_ssws = np.stack((varnames[0][1:varnames[0].shape[0],:,:],varnames[1][1:varnames[0].shape[0],:,:],
                     varnames[2][1:varnames[0].shape[0],:,:]))


# In[62]:


var_ssws = np.ma.masked_greater(var_ssws,1e20)


# In[63]:


Nm,Ny,Nlat,Nlon = var_ssws.shape
print(var_ssws.shape)


# In[64]:


#Now, calculate standardized zonal wind anomalies
sst_bar = np.ma.mean(var_ssws,axis=1)
sst_std = np.ma.std(var_ssws,axis=1)
sst_norm = np.reshape(var_ssws,(Ny,Nm,Nlat,Nlon))/sst_std


# In[65]:


ssta = np.ma.MaskedArray.anom(var_ssws,axis=1)
ssts = np.reshape(np.ma.MaskedArray.anom(sst_norm,axis=0),(Nm,Ny,Nlat,Nlon))


# In[66]:


#seasonal averages
ssta_jfm = np.mean(ssta,axis=0) #average over JFM
#ssta_amj = np.mean(ssta[3:6,:,:,:],axis=0) #average over AMJ
#ssta_jas = np.mean(ssta[6:9,:,:,:],axis=0) #average over JAS
#ssta_ann = np.mean(ssta,axis=0)            #Annual mean


# In[67]:


sst_bar_jfm = np.ma.mean(np.ma.mean(var_ssws,axis=0),axis=0) #average over JFM


# In[68]:


ssta_jfm.shape


# In[69]:


#unpickle SSW central dates
with open(datdir1 + 'ENSO_NDJF_years_rev.pickle','rb') as fp:
    elnino_yrs,lanina_yrs = pickle.load(fp,encoding='latin1')


# In[70]:


#elnino_yrs


# In[72]:


##create composite of El Nino/La Nina years
ElNinoyrs = ssta_jfm[np.squeeze(elnino_yrs),:,:]
ElNinoyrs_composite = np.mean(ElNinoyrs,axis=0)

LaNinayrs = ssta_jfm[np.squeeze(lanina_yrs),:,:]
LaNinayrs_composite = np.mean(LaNinayrs,axis=0)


#ElNinoyrs_ann = ssta_ann[np.squeeze(elnino_yrs),:,:]
#ElNinoyrs_composite_ann = np.mean(ElNinoyrs_ann,axis=0)

#LaNinayrs_ann = ssta_ann[np.squeeze(lanina_yrs),:,:]
#LaNinayrs_composite_ann = np.mean(LaNinayrs_ann,axis=0)


# In[74]:


#test significance of ENSO composite mean anomalies
tp1 = stats.ttest_1samp(ElNinoyrs,0.0,axis=0)
tp2 = stats.ttest_1samp(LaNinayrs,0.0,axis=0)


# In[75]:


#for some reason ma.masked_outside did not work and ma.masked_where with two conditions also did not work
p1 = np.ma.masked_invalid(tp1[1])
p1 = np.ma.masked_where(p1>=0.05,p1)
p1 = np.ma.masked_where(p1==0,p1)
p1[~p1.mask]=1

p2 = np.ma.masked_invalid(tp2[1])
p2 = np.ma.masked_where(p2>=0.05,p2)
p2 = np.ma.masked_where(p2==0,p2)
p2[~p2.mask]=1


# In[142]:


# set up plot
v = np.linspace(-0.4,0.4,17)
fig = plt.figure(figsize=(16,12))

fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(16,12),
                      subplot_kw={'projection': ccrs.PlateCarree()})
#ax = plt.axes(projection=ccrs.PlateCarree())
ax[0].set_global()
ax[0].set_extent([-80,0,0,80])
ax[0].coastlines()
ax[0].gridlines(linewidth=0.5)
ax[0].set_xticks([-90,-60,-30,0],crs=ccrs.PlateCarree())
ax[0].set_yticks([0, 20, 40, 60, 80],crs=ccrs.PlateCarree())
ax[0].set_title('(a) El Niño Anomaly: SST, JFM', y=1.04, weight='bold')

# plot data
pc = ax[0].contourf(lon,lat,ElNinoyrs_composite,v,cmap="RdBu_r")

# add colorbar
cax,kw = mpl.colorbar.make_axes(ax[0],location='right',pad=0.05,shrink=0.455)
out=fig.colorbar(pc,cax=cax,extend='both',**kw)
out.set_label('K',size=14,weight='bold')
plt.tight_layout()

#ax = plt.axes(projection=ccrs.PlateCarree())
ax[1].set_global()
ax[1].set_extent([-80,0,0,80])
ax[1].coastlines()
ax[1].gridlines(linewidth=0.5)
ax[1].set_xticks([-90,-60,-30,0],crs=ccrs.PlateCarree())
ax[1].set_yticks([0, 20, 40, 60, 80],crs=ccrs.PlateCarree())
ax[1].set_title('(b) La Niña Anomaly: SST, JFM', y=1.04,weight='bold')

# plot data
pc = ax[1].contourf(lon,lat,LaNinayrs_composite,v,cmap="RdBu_r")

# add colorbar
cax,kw = mpl.colorbar.make_axes(ax[1],location='right',pad=0.05,shrink=0.455)
out=fig.colorbar(pc,cax=cax,extend='both',**kw)
out.set_label('K',size=14,weight='bold')
plt.tight_layout()

#save figure
plt.savefig(datdir1+'SST_ATL_JFM_ENSO_composites_allyrs.eps',bbox_inches='tight')


# In[71]:


#define projection
xx, yy = np.meshgrid(lon, lat)
mm = Basemap(projection='cyl', lon_0 = -180, llcrnrlon=0, llcrnrlat=-20, urcrnrlon=360, urcrnrlat=80)
x, y = mm(xx,yy) #converts rectangular meshgrid into meshgrid for the specific projection


# In[72]:


#plot in stereographic projetion
fig = plt.figure(figsize=(16,6))
v = np.linspace(-2,2,17)

ax1 = fig.add_subplot(2,1,1)
mm.drawcoastlines()
mm.drawmapboundary(fill_color='none')
pc1 = mm.contourf(x,y, ElNinoyrs_composite, v, cmap='RdBu_r')
pc1.set_clim(-2,2)
#pc2 = mm.scatter(x,y, p1,c='gray')
mm.fillcontinents(color='gray')
mm.drawparallels(range(0,90,20), labels=[1,1,0,0])
mm.drawmeridians(range(30,360,30), labels=[0,0,1,1])
plt.title('El Nino Anomaly: SST, NDJFM', y=1.15, weight='bold')

plt.tight_layout()

#add colorbar
cbar = fig.colorbar(pc1)
cbar.set_label('$^\circ$ C', fontsize=14,weight='bold') #$^\circ$ C
cbar.set_ticks(v,update_ticks=True)

ax2 = fig.add_subplot(2,1,2)
mm.drawcoastlines()
mm.drawmapboundary(fill_color='none')
pc2 = mm.contourf(x,y, LaNinayrs_composite, v, cmap='RdBu_r')
pc2.set_clim(-2,2)
#pc2 = mm.scatter(x,y, p1,c='gray')
mm.fillcontinents(color='gray')
mm.drawparallels(range(0,90,20), labels=[1,1,0,0])
mm.drawmeridians(range(30,360,30), labels=[0,0,1,1])
plt.title('La Nina Anomaly: SST, NDJFM', y=1.15, weight='bold')

plt.tight_layout()

#add colorbar
cbar = fig.colorbar(pc2)
cbar.set_label('$^\circ$ C', fontsize=14,weight='bold') #$^\circ$ C
cbar.set_ticks(v,update_ticks=True)

#save figure
#plt.savefig('SST_NDJFM_ENSO_composites_allyrs_noSSWs.eps',bbox_inches='tight')


# In[37]:


#define projection
xx, yy = np.meshgrid(lon, lat)
mm = Basemap(projection='cyl', lon_0 = -45, llcrnrlon=270, llcrnrlat=0, urcrnrlon=360, urcrnrlat=80)
x, y = mm(xx,yy) #converts rectangular meshgrid into meshgrid for the specific projection


# In[42]:


#plot in stereographic projetion
fig = plt.figure(figsize=(16,6))
v = np.linspace(-0.4,0.4,17)

ax1 = fig.add_subplot(1,2,1)
mm.drawcoastlines()
mm.drawmapboundary(fill_color='none')
pc1 = mm.contourf(x,y, ElNinoyrs_composite, v, cmap='RdBu_r')
pc1.set_clim(-0.4,0.4)
#pc2 = mm.scatter(x,y, p1,c='gray')
mm.fillcontinents(color='gray')
mm.drawparallels(range(0,90,20), labels=[1,1,0,0])
mm.drawmeridians(range(30,360,30), labels=[0,0,1,1])
plt.title('El Nino Anomaly: SST, NDJFM', y=1.15, weight='bold')

plt.tight_layout()

#add colorbar
cbar = fig.colorbar(pc1)
cbar.set_label('$^\circ$ C', fontsize=14,weight='bold') #$^\circ$ C
cbar.set_ticks(v,update_ticks=True)

ax2 = fig.add_subplot(1,2,2)
mm.drawcoastlines()
mm.drawmapboundary(fill_color='none')
pc2 = mm.contourf(x,y, LaNinayrs_composite, v, cmap='RdBu_r')
pc2.set_clim(-0.4,0.4)
#pc2 = mm.scatter(x,y, p1,c='gray')
mm.fillcontinents(color='gray')
mm.drawparallels(range(0,90,20), labels=[1,1,0,0])
mm.drawmeridians(range(30,360,30), labels=[0,0,1,1])
plt.title('La Nina Anomaly: SST, NDJFM', y=1.15, weight='bold')

plt.tight_layout()

#add colorbar
cbar = fig.colorbar(pc2)
cbar.set_label('$^\circ$ C', fontsize=14,weight='bold') #$^\circ$ C
cbar.set_ticks(v,update_ticks=True)

#save figure
#plt.savefig('SST_ATL_NDJFM_ENSO_composites_allyrs_noSSWs.eps',bbox_inches='tight')


# In[33]:


#unpickle El Nino central dates
with open('TAUX_composite_ElNino.pickle','rb') as fp:
    HMXL_ElNinovar_composite,HMXL_ElNinovar_series,p11,lon1,lat1 = pickle.load(fp)


# In[34]:


#unpickle El Nino central dates
with open('TAUX_composite_LaNina.pickle','rb') as fp:
    HMXL_LaNinavar_composite,HMXL_LaNinavar_series,p21,lon1,lat1 = pickle.load(fp)


# In[35]:


#define projection
xx, yy = np.meshgrid(lon1, lat1)
mm2 = Basemap(projection='cyl', lon_0 = -45, llcrnrlon=270, llcrnrlat=0, urcrnrlon=360, urcrnrlat=80)
x, y = mm2(xx,yy) #converts rectangular meshgrid into meshgrid for the specific projection


# In[36]:


vmin = np.around(np.min(HMXL_ElNinovar_composite),decimals=2)
vmax = np.around(np.max(HMXL_ElNinovar_composite),decimals=2)
if np.abs(vmin) > np.abs(vmax):
    vint = np.abs(np.around((vmin)/12,decimals=3))
    vmax = np.abs(vmin)
else:
    vint = np.around((vmax)/12,decimals=3)
    vmin = -vmax

v = np.linspace(float(vmin)-float(vint),float(vmax)+float(vint),13)


# In[37]:


print v


# In[39]:


#plot in stereographic projetion
fig = plt.figure(figsize=(16,6))
v = np.linspace(-120,120,21)
v2 = np.linspace(-120,120,11)

ax1 = fig.add_subplot(1,2,1)
mm2.drawcoastlines()
mm2.drawmapboundary(fill_color='none')
pc1 = mm2.contourf(x,y, HMXL_ElNinovar_composite, cmap='RdBu_r')
#pc1.set_clim(-90,90)
pc2 = mm2.contourf(x,y,p11,colors = 'none', hatches=['//'])
mm2.fillcontinents(color='gray')
mm2.drawparallels(range(0,90,20), labels=[1,0,0,0])
mm2.drawmeridians(range(30,360,30), labels=[0,0,0,1])
plt.title('(a) El Nino', y=1.03, weight='bold')

plt.tight_layout()

#add colorbar
#cbar = fig.colorbar(pc1)
#cbar.set_label('Pa', fontsize=14,weight='bold') #$^\circ$ C
#cbar.set_ticks(v,update_ticks=True)

ax2 = fig.add_subplot(1,2,2)
mm2.drawcoastlines()
mm2.drawmapboundary(fill_color='none')
pc2 = mm2.contourf(x,y, HMXL_LaNinavar_composite, cmap='RdBu_r')
#pc2.set_clim(-90,90)
pc2 = mm2.contourf(x,y,p21,colors = 'none', hatches=['//'])
mm2.fillcontinents(color='gray')
mm2.drawparallels(range(0,90,20), labels=[1,0,0,0])
mm2.drawmeridians(range(30,360,30), labels=[0,0,0,1])
plt.title('(b) La Nina', y=1.03, weight='bold')

plt.tight_layout()

#add colorbar
cbar = fig.colorbar(pc1)
cbar.set_label('m', fontsize=16,weight='bold') #$^\circ$ C
cbar.set_ticks(v2,update_ticks=True)

#save figure
#plt.savefig('HMXL_JFM_ENSO.eps',bbox_inches='tight')


# In[20]:


from area_diags import area_avg


# In[21]:


lat1[44:65]


# In[22]:


ElNino_yrs = ElNinovar_series[:,44:65,:]*0.1
LaNina_yrs = LaNinavar_series[:,44:65,:]*0.1
lat2 = lat1[44:65]


# In[23]:


#area average to great timeseries
p = area_avg.averages()
ElNino_avg = p.LatLonavg(var=np.squeeze(ElNino_yrs[:,:,:]), lat=lat2, lon=lon1)
LaNina_avg = p.LatLonavg(var=np.squeeze(LaNina_yrs[:,:,:]), lat=lat2, lon=lon1)


# In[24]:


tp11 = stats.ttest_1samp(ElNino_avg,0.0,axis=0)
p11 = np.ma.masked_invalid(tp11[1])
p11 = np.ma.masked_where(p11>=0.05,p11)
p11 = np.ma.masked_where(p11==0,p11)
p11[~p11.mask]=1

tp12 = stats.ttest_1samp(LaNina_avg,0.0,axis=0)
p12 = np.ma.masked_invalid(tp12[1])
p12 = np.ma.masked_where(p12>=0.05,p12)
p12 = np.ma.masked_where(p12==0,p12)
p12[~p12.mask]=1


# In[27]:


print np.mean(ElNino_avg), np.mean(LaNina_avg)


# In[28]:


pname = 'TAUXavg_JFM_composite_ElNino.pickle'
with open(pname,'wb') as fp:
    pickle.dump([ElNino_avg,p11],fp)
    
pname = 'TAUXavg_JFM_composite_LaNina.pickle'
with open(pname,'wb') as fp:
    pickle.dump([LaNina_avg,p12],fp)


# In[41]:


#unpickle El Nino central dates
with open('PSL_composite_ElNino.pickle','rb') as fp:
    ElNinovar_composite,ElNinovar_series,p12,lon2,lat2 = pickle.load(fp)


# In[42]:


#print p22[26,251]


# In[43]:


#unpickle El Nino central dates
with open('PSL_composite_LaNina.pickle','rb') as fp:
    LaNinavar_composite,LaNinavar_series,p22,lon2,lat2 = pickle.load(fp)


# In[44]:


len(ElNinovar_series)


# In[45]:


#add cyclic points manually 
ElNinovar_cyclic = np.zeros((len(lat2),len(lon2)+1),np.float)
ElNinovar_cyclic[:,0:len(lon2)]= ElNinovar_composite[:,:]
ElNinovar_cyclic[:,len(lon2)] = ElNinovar_composite[:,len(lon2)-1]

LaNinavar_cyclic = np.zeros((len(lat2),len(lon2)+1),np.float)
LaNinavar_cyclic[:,0:len(lon2)]= LaNinavar_composite[:,:]
LaNinavar_cyclic[:,len(lon2)] = LaNinavar_composite[:,len(lon2)-1]

p12_cyclic = np.ma.zeros((len(lat2),len(lon2)+1),np.float)
p12_cyclic[:,0:len(lon2)]= p12[:,:]
p12_cyclic[:,len(lon2)] = p12[:,len(lon2)-1]

p22_cyclic = np.ma.zeros((len(lat2),len(lon2)+1),np.float)
p22_cyclic[:,0:len(lon2)]= p22[:,:]
p22_cyclic[:,len(lon2)] = p22[:,len(lon2)-1]

lons = np.zeros([1,lon2.size+1])
lons[0,0:lon2.size] = lon2[:]
lons[0,lon2.size] = 360


# In[46]:


lon_new = lons
xx, yy = np.meshgrid(lon_new, lat2)
m = Basemap(projection='npstere', lat_0=90, lon_0=0, boundinglat=20, resolution='c',round=True)
x, y = m(xx, yy) #converts rectangular meshgrid into meshgrid for the specific projection


# In[47]:


#only plot significant values
ElNino_sig = np.ma.masked_where(p12_cyclic!=1,ElNinovar_cyclic)
LaNina_sig = np.ma.masked_where(p22_cyclic!=1,LaNinavar_cyclic)


# In[48]:


vmin = np.around(np.min(ElNinovar_cyclic),decimals=2)
vmax = np.around(np.max(ElNinovar_cyclic),decimals=2)
if np.abs(vmin) > np.abs(vmax):
    vint = np.abs(np.around((vmin)/12,decimals=3))
    vmax = np.abs(vmin)
else:
    vint = np.around((vmax)/12,decimals=3)
    vmin = -vmax

v = np.linspace(float(vmin)-float(vint),float(vmax)+float(vint),13)
print v


# In[53]:


#plot in stereographic projetion
fig = plt.figure(figsize=(14,6))
v = np.linspace(-4,4,19)
v2 = np.linspace(-4,4,9)

ax1 = fig.add_subplot(1,2,1)
m.drawcoastlines()
m.drawmapboundary(fill_color='none')
pc1 = m.contourf(x,y, ElNinovar_cyclic/100, v,cmap='RdBu_r')
#pc1.set_clim(-4,4)
pc2 = m.contourf(x,y,p12_cyclic,colors = 'none', hatches=['//'])
m.drawparallels(range(0,90,20), labels=[1,1,0,0])
m.drawmeridians(range(30,360,30), labels=[0,0,0,0])
plt.title('(a) El Nino: JFM PSL', y=1.05, weight='bold',fontsize=16)
plt.tight_layout()

#add colorbar
#cbar = fig.colorbar(pc1)
#cbar.set_label('Pa', fontsize=14,weight='bold') #$^\circ$ C
#cbar.set_ticks(v,update_ticks=True)

ax2 = fig.add_subplot(1,2,2)
m.drawcoastlines()
m.drawmapboundary(fill_color='none')
pc1 = m.contourf(x,y, LaNinavar_cyclic/100, v,cmap='RdBu_r')
#pc2.set_clim(-4,4)
pc2 = m.contourf(x,y,p22_cyclic,colors = 'none', hatches=['//'])
m.drawparallels(range(0,90,20), labels=[1,1,0,0])
m.drawmeridians(range(30,360,30), labels=[0,0,0,0])
plt.title('(b) La Nina: JFM PSL', y=1.05, weight='bold',fontsize=16)
plt.tight_layout()

#add colorbar
cbar = fig.colorbar(pc1)
cbar.set_label('hPa', fontsize=16,weight='bold') #$^\circ$ C
cbar.set_ticks(v2,update_ticks=True)

#save figure
#plt.savefig('PSL_JFM_ENSO.eps',pad_inches=2.25) #,bbox_inches='tight'


# In[ ]:




