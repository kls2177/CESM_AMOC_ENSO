#load packages
import numpy as np
from netCDF4 import Dataset
import cPickle as pickle
import scipy.stats as stats

#user defined inputs
datdir1 = '/d11/ksmith/cesmdata/b.e11.B1850C5CN.f09_g16.005/atm'
#allvars = ['SHF','SHF_QSW','EVAP_F','LWDN_F','LWUP_F','MELTH_F','SENH_F']
allvars = ['PSL']
fileend = '.allyrs.nc'
month = ['12', '01', '02', '03']

for var in range(len(allvars)):
    
    #open netcdf files
    varnames = dict()
    for i in range(len(month)):
        fname         = datdir1 + '/SURF.' + month[i] + fileend
        nc            = Dataset(fname)
        varnames_tmp1 = np.squeeze(nc.variables[allvars[var]][:,:,:])
        varnames[i]   = np.ma.masked_greater(varnames_tmp1,1e20)
        
        lon           = nc.variables['lon'][:]
        lat           = nc.variables['lat'][:]

    print 'begin', allvars[var]


    #create ENSO winters
    var_ssws = np.stack((varnames[0][1:varnames[0].shape[0]-1,:,:],varnames[0][2:varnames[1].shape[0],:,:],
                         varnames[2][2:varnames[0].shape[0],:,:],
                         varnames[3][2:varnames[0].shape[0],:,:]))

    #mask out continents
    var_ssws = np.ma.masked_greater(var_ssws,1e20)

    #calculate anomalies
    ssta = np.ma.MaskedArray.anom(var_ssws,axis=1)
    
    #seasonal averages
    ssta_jfm = np.mean(ssta[1:4,:,:,:],axis=0) #average over DJFM

    #unpickle SSW central dates
    with open('ENSO_NDJF_years.pickle','rb') as fp:
        elnino_yrs,lanina_yrs = pickle.load(fp)

    #create composite of El Nino/La Nina years
    ElNinoyrs = ssta_jfm[np.squeeze(elnino_yrs[25:])-100,:,:]
    ElNinoyrs_composite = np.mean(ElNinoyrs,axis=0)

    LaNinayrs = ssta_jfm[np.squeeze(lanina_yrs[27:])-100,:,:]
    LaNinayrs_composite = np.mean(LaNinayrs,axis=0)

    #test significance of ENSO composite mean anomalies
    tp1 = stats.ttest_1samp(ElNinoyrs,0.0,axis=0)
    tp2 = stats.ttest_1samp(LaNinayrs,0.0,axis=0)

    #mask out all p-values >=0.05 and also 0.
    #note: for some reason ma.masked_outside did not work and ma.masked_where with two conditions also did not work
    p1 = np.ma.masked_invalid(tp1[1])
    p1 = np.ma.masked_where(p1>=0.05,p1)
    p1 = np.ma.masked_where(p1==0,p1)
    p1[~p1.mask]=1

    p2 = np.ma.masked_invalid(tp2[1])
    p2 = np.ma.masked_where(p2>=0.05,p2)
    p2 = np.ma.masked_where(p2==0,p2)
    p2[~p2.mask]=1

    #pickle the dates for later use
    pname1 = allvars[var] + '_composite_ElNino.pickle'
    with open(pname1,'wb') as fp:
        pickle.dump([ElNinoyrs_composite,ElNinoyrs,p1,lon,lat],fp)

    pname2 = allvars[var] + '_composite_LaNina.pickle'
    with open(pname2,'wb') as fp:
        pickle.dump([LaNinayrs_composite,LaNinayrs,p2,lon,lat],fp)

    print 'end',allvars[var]
