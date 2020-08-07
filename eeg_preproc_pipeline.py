
#### preprocessing pipeline for EEG data

####################################### input
file = "" # file name excluding extension
####################################### input

## import libraries
import mne
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
## load data 
path = 'XXX/%s.vhdr'%(file) # input path
dat = mne.io.read_raw_brainvision(path, preload = True)
dat.set_channel_types(mapping={'EOG1':'eog',
                               'EOG2':'eog',
                               'EOG3':'eog',
                               'LMast':'misc',
                               'RMast':'misc'})
montage = mne.channels.read_montage('standard_1005')
dat.set_montage(montage)
## Re-reference to the mastoid average
dat = dat.set_eeg_reference(['LMast','RMast'])
## downsample to 200Hz 
dat = dat.resample(200, npad="auto") 
## Bandpass filter 0.05 to 50 Hz
dat = dat.filter(0.05, 50., fir_design='firwin')


####################################### input
dat.plot()
#click on bad channels
#if bads
dat.interpolate_bads()
####################################### input


## Epoch Data
events = mne.find_events(dat, stim_channel='STI 014')
event_id, tmin, tmax = 1, -0.5, 4.0
baseline = (None, 0.0) #non = minimum time, 0 equals stim onset
dat_e = mne.Epochs(dat, events, event_id, tmin, tmax,
                      baseline = baseline, reject = None, proj = False,
                      reject_by_annotation = False, preload = True)

## ICA round 1 for eye artifacts
method, n_components, random_state = 'fastica', 58, 23
ica = mne.preprocessing.ICA(n_components = n_components, method = method,
                            random_state = random_state)
ica.fit(dat_e)
icatc = ica.get_sources(dat_e).get_data()
Fp1tc = dat_e.get_data()[:,0,:]
EOG1tc = dat_e.get_data()[:,59,:]
veogtc = ((Fp1tc + EOG1tc) / 2).flatten()
EOG2tc = dat_e.get_data()[:,60,:]
EOG3tc = dat_e.get_data()[:,61,:]
heogtc = ((EOG2tc + EOG3tc) / 2).flatten()
vcorr = []
hcorr = []
for x in list(range(0,icatc.shape[1])):
   temp = icatc[:,x,:].flatten()
   vcorr = np.append(vcorr, pearsonr(temp,veogtc)[0])
   hcorr = np.append(hcorr, pearsonr(temp,heogtc)[0])
rej_vcomps = np.where(np.absolute(vcorr) > (3*np.std(vcorr)))
rej_hcomps = np.where(np.absolute(hcorr) > (3*np.std(hcorr)))
rej_eyes = sorted(np.append(rej_vcomps, rej_hcomps))

dat_e = ica.apply(dat_e, exclude = rej_eyes)
dat_e = dat_e.apply_baseline(baseline = (None, 0.))

#### ICA round 2
method, n_components, random_state = 'fastica', 58-len(rej_eyes), 23
ica2 = mne.preprocessing.ICA(n_components = n_components, method = method,
                             random_state = random_state)
ica2.fit(dat_e)


####################################### input
ica2.plot_components()
ica2.plot_properties(dat_e, picks = [],psd_args={'fmax':35.}) #fill in [] with components

exclude_bad_comps = [] # fill in [] with components to exclude
dat_e = ica2.apply(dat_e, exclude = exclude_bad_comps)
dat_e = dat_e.apply_baseline(baseline = (None, 0.))
####################################### input


####################################### input
dat_e.plot()
#click on bad epochs to drop them
####################################### input


## save file
epoch_log = dat_e.drop_log
path = 'XXX/processed/%s.fif'%(file)
dat_e.save(path)

df = pd.DataFrame(np.array(epoch_log))
df.columns = ['col']
df.col = df.col.astype(str)
location = np.where(df.col == '[]')[0]
df['bad'] = 1
df.loc[location,:'bad'] = 0
del(df['col'])
path2 = 'XXX/processed/%s_badepoch.csv'%(file)
df.to_csv(path2)







