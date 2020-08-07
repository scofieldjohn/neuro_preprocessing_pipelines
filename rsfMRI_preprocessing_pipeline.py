

############################## run first in terminal
#FSL
FSLDIR=/usr/share/fsl
. ${FSLDIR}/5.0/etc/fslconf/fsl.sh
PATH=${FSLDIR}/5.0/bin:${PATH}
export FSLDIR PATH
#FreeSurfer
#export FREESURFER_HOME=/usr/local/freesurfer
#source $FREESURFER_HOME/SetUpFreeSurfer.sh
#ANTs
#export PATH=/usr/local/ANTs-2.3.1/bin:$PATH
#export ANTSPATH=/usr/local/ANTs-2.3.1/bin/
export ANTSPATH=${HOME}/bin/ants/bin/
export PATH=${ANTSPATH}:$PATH
#AFNI
export PATH=/usr/lib/afni/bin:$PATH

ipython
############################## run first in terminal


############################## specify parameters
import os
#where does data live?
base_dir = '/home/john/Desktop'
resource_dir = '/home/john/Desktop/resources'
site = 'ABIDEII-GU_1' #enter in what the folder says
data_dir = '%s/abide2/ABIDEII-GU_1'%(base_dir) #or a site like abide1/site1
template_design_path = '%s/template_design.fsf'%(resource_dir)
#subject parameters
#subnum = '0050183'
sublist = os.listdir(data_dir)
settr_num = 2.0
highpass = .01
lowpass = .1
interleaved_val = True #true means interleaved
slice_direction_val = 3 #(x=1, y=2, z=3)
index_dir_val = False #false is F >> H direction
############################## specify parameters



#load in modules
from nipype.interfaces import fsl
from nipype.interfaces import afni
from nipype import Node
from nipype.testing import example_data
from nipype.interfaces.ants import N4BiasFieldCorrection
import os; import glob
from shutil import copyfile
import nibabel as nib
from numpy import average, savetxt
from nilearn.masking import apply_mask #might need nitime installed also
from nilearn.image import resample_to_img
from nilearn.masking import compute_epi_mask
import glob
import os
from shutil import copyfile
import time
from nilearn.image import threshold_img
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import seaborn as sns
import matplotlib.pyplot as plt



t1 = time.time()

for subnum in sublist: #loop through subjects

	#change subnum to string
	settr = str(settr_num)

	#check for/make output directory
	if not os.path.exists('%s/pipeline_output'%(base_dir)):
		os.makedirs('%s/pipeline_output'%(base_dir))
	if not os.path.exists("%s/pipeline_output/%s"%(base_dir, site)):
		os.makedirs('%s/pipeline_output/%s'%(base_dir, site))
	if not os.path.exists('%s/pipeline_output/%s/%s'%(base_dir,site,subnum)):
		os.makedirs('%s/pipeline_output/%s/%s'%(base_dir,site,subnum))

	output_dir = '%s/pipeline_output/%s/%s'%(base_dir,site,subnum)

	#create a html file for Quality Assessment
	outhtml = "%s/pipeline_output/%s/funct_motion_QA.html"%(base_dir,site)
	#path where bad vols will be stored
	out_bad_func_list = "%s/pipeline_output/%s/subs_lose_vol_scrub.txt"%(base_dir,site)
	#checking correlation matrices by subject
	out_heatmap = "%s/pipeline_output/%s/heatmap_list.html"%(base_dir,site)



	########## 1: motion scrubbing
	cur_dir = '%s/%s/session_1/rest_1/rest.nii.gz'%(data_dir,subnum)
	#strip off .nii.gz
	cur_func_no_nii = cur_dir[:-7]
	#assess motion using .2 as threshold
	if os.path.isdir("%s/motion_assess/"%(output_dir)) == False:
		os.system("mkdir %s/motion_assess"%(output_dir))
	#using fsl_motion_outliers function
	os.system("fsl_motion_outliers -i %s -o %s/motion_assess/confound.txt --fd --thresh=0.2 -p %s/motion_assess/fd_plot -v > %s/motion_assess/outlier_output.txt"%(cur_func_no_nii, output_dir, output_dir, output_dir))
	#put confound info into html file to check later
	os.system("cat %s/motion_assess/outlier_output.txt >> %s"%(output_dir, outhtml))
	os.system("echo '<p>================================================<p>FD plot %s <br><IMG BORDER=0 SRC=%s/motion_assess/fd_plot.png WIDTH=100%s></BODY></HTML>' >> %s"%(output_dir, output_dir, '%', outhtml))
	#empty confound.txt files for S's who don't need scrubbing
	#simplifies process later on
	if os.path.isfile("%s/motion_assess/confound.txt"%(output_dir))==False:
		os.system("touch %s/motion_assess/confound.txt"%(output_dir))

	dat = pd.DataFrame(np.loadtxt('%s/motion_assess/confound.txt'%(output_dir)))
	
	if len(dat)==0:
		scrub_vols = []
		print('confound file is empty')
		with open(out_bad_func_list, "a") as myfile:
			myfile.write("%s 0 percent\n"%(subnum))
	else:
		scrub_vols = []		
		for x in list(range(0,len(dat.columns))):
			scrub_vols = np.append(scrub_vols,np.where(dat.iloc[:,x]==1)[0][0])		
		howmanyvol = nib.load(cur_dir).get_fdata().shape[3]
		with open(out_bad_func_list, "a") as myfile:
			myfile.write("%s %s percent\n"%(subnum,np.round(((len(scrub_vols) / howmanyvol) * 100),2)))
	
	#manually remove bad slices
	motscrubdat = nib.load(cur_dir).get_fdata()
	if len(scrub_vols) > 0:
		motscrubdat2 = np.delete(motscrubdat,scrub_vols,axis=3)
		array_img = nib.Nifti1Image(motscrubdat2, nib.load(cur_dir).affine)
		nib.save(array_img, '%s/rest.nii.gz'%(output_dir))
	else:
		nib.save(nib.load(cur_dir), '%s/rest.nii.gz'%(output_dir))

	########## 2: Remove first 5 vols of func data
	os.system("fslroi %s/rest.nii.gz %s/rest_delvol.nii.gz 5 350"%(output_dir,output_dir))

	########## 3: Bias Field Correction
	inu_n4 = Node(N4BiasFieldCorrection(
				dimension=3,
				input_image = '%s/%s/session_1/anat_1/anat.nii.gz'%(data_dir,subnum),
				output_image = '%s/mprage_inu.nii.gz'%(output_dir))
				, name='inu_n4')
	res1 = inu_n4.run()

	########## 4: Skullstripping
	#anat
	bet_anat = Node(fsl.BET(frac=0.5,
						robust=True,
						output_type='NIFTI_GZ',
						in_file= '%s/mprage_inu.nii.gz'%(output_dir),
						out_file = '%s/mprage_inu_bet.nii.gz'%(output_dir)),
					name="bet_anat")
	res = bet_anat.run()
	#func
	bet_func = Node(fsl.BET(frac=0.5,
						output_type='NIFTI_GZ',
						in_file= '%s/rest_delvol.nii.gz'%(output_dir),
						out_file = '%s/rest_delvol_bet.nii.gz'%(output_dir),
						functional=True),
					name="bet_func")
	res2 = bet_func.run()

	########## 5: Anatomical Registration
	fltreg1 = Node(fsl.FLIRT(
			bins=640, cost_func='mutualinfo',
			in_file = '%s/mprage_inu_bet.nii.gz'%(output_dir),
			reference = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
			output_type = 'NIFTI_GZ',
			out_matrix_file = '%s/mprage_inu_bet_warpflirt.mat'%(output_dir),
			out_file = '%s/mprage_inu_bet_warpflirt.nii.gz'%(output_dir)
			),name='fltreg1')
	res3 = fltreg1.run()

	########## 6: Tissue Segmentation
	os.system("fast --channels=1 --type=1 --out=%s/fast --class=3 %s/mprage_inu_bet_warpflirt.nii.gz"%(output_dir,output_dir)) 
	#threshold to 50% CSF
	os.system("fslmaths %s/fast_pve_0.nii.gz -thr 0.50 -bin %s/fast_pve_0_thresh.nii.gz"%(output_dir,output_dir))
	#threshold to 50% WM
	os.system("fslmaths %s/fast_pve_2.nii.gz -thr 0.50 -bin %s/fast_pve_2_thresh.nii.gz"%(output_dir,output_dir))   
	   
	########## 7: Motion Correction
	mcflt = Node(fsl.MCFLIRT(
			in_file = '%s/rest_delvol_bet.nii.gz'%(output_dir),
			cost = 'mutualinfo',
			out_file = '%s/rest_delvol_bet_mc.nii.gz'%(output_dir),
			save_plots = True,
			save_rms = False,
			output_type = 'NIFTI_GZ'
			), name = 'mcflt')
	res3 = mcflt.run()

	########## 8: Slice Time Correction
	tshiftfsl = Node(fsl.SliceTimer(
			in_file = '%s/rest_delvol_bet_mc.nii.gz'%(output_dir),
			out_file = '%s/rest_delvol_bet_mc_tshift.nii.gz'%(output_dir),
			interleaved = interleaved_val,
			slice_direction = slice_direction_val,
			time_repetition = settr_num,
			index_dir = index_dir_val
			), name = 'tshiftfsl')
	res5 = tshiftfsl.run()

	########## 9: Coregistration, Normalization, Smoothing
	data4d = '%s/rest_delvol_bet_mc_tshift.nii.gz'%(output_dir)
	data3d = '%s/mprage_inu_bet'%(output_dir)
	outpath = '%s/reg'%(output_dir)

	settotvol = str(nib.load('%s/rest_delvol_bet_mc_tshift.nii.gz'%(output_dir)).get_fdata().shape[3])
	replacements = {'OUTPUTSUBNUM': outpath,
					'DATA3D': data3d,
					'DATA4D': data4d,
					'SETTR': settr,
					'SETTOTVOL': settotvol}
	with open(template_design_path) as infile:
		with open("%s/design_sub%s.fsf"%(output_dir, subnum), 'w') as outfile:
			for line in infile:
				for src, target in replacements.items():
					line = line.replace(src, target)
				outfile.write(line)

	#registration/smooth
	os.system("feat %s/design_sub%s.fsf"%(output_dir, subnum))
	#apply warp
	os.system("applywarp -i %s/reg.feat/filtered_func_data.nii.gz -o %s/func_mni.nii.gz -r %s/reg.feat/reg/standard.nii.gz -w %s/reg.feat/reg/example_func2standard_warp.nii.gz -v"%(output_dir,output_dir,output_dir,output_dir))

	########## 10: Band-pass Filter
	bandpass = Node(afni.Bandpass(
			in_file = '%s/func_mni.nii.gz'%(output_dir),
			highpass = highpass,
			lowpass = lowpass,
			out_file = '%s/func_mni_bp.nii.gz'%(output_dir),
			tr = settr_num
			),name='bandpass')
	bpres = bandpass.run()

	########## 11: Confound Estimation and Regression

	#create wm and csf and epi confound files
	func_file = '%s/func_mni_bp.nii.gz'%(output_dir)
	maskwm = '%s/white.nii'%(resource_dir)
	maskcsf = '%s/ventricle.nii'%(resource_dir)
	maskepi = '%s/EPI.nii'%(resource_dir)
	#make affines match
	resampled_csf = resample_to_img(maskcsf, func_file)
	resampled_wm = resample_to_img(maskwm, func_file)
	resampled_epi = resample_to_img(maskepi, func_file)
	#threshold
	resampled_csf_thresh = threshold_img(resampled_csf,0.5)#.5 irrelevant, just making binary
	resampled_wm_thresh = threshold_img(resampled_wm, 0.5)
	resampled_epi_thresh = threshold_img(resampled_epi,0.5)
	#apply mask
	x = apply_mask(func_file, resampled_csf_thresh)
	x_x = apply_mask(func_file, resampled_wm_thresh)
	x_x_x = apply_mask(func_file, resampled_epi_thresh)
	y = x.mean(axis=1)
	y_y = x_x.mean(axis=1)
	y_y_y = x_x_x.mean(axis=1)
	#saving for later use as confound
	savetxt('%s/CSFseries.txt'%(output_dir),y)
	savetxt('%s/WMseries.txt'%(output_dir),y_y)
	savetxt('%s/EPIseries.txt'%(output_dir),y_y_y)

    

	#calculate 36 parameter model
	ff = pd.DataFrame(np.loadtxt('%s/rest_delvol_bet_mc.nii.gz.par'%(output_dir)))
	ff['w'] = pd.DataFrame(y)
	ff['c'] = pd.DataFrame(y_y)
	ff['e'] = pd.DataFrame(y_y_y)
    
	tmpdat_deriv = pd.DataFrame()
	tmpdat_squar = pd.DataFrame()
	tmpdat_dersq = pd.DataFrame()
	for x in list(range(0,len(ff.columns))):
		tmpdat_deriv = pd.concat((tmpdat_deriv,pd.DataFrame(np.diff(ff.iloc[:,x]))),axis=1)
		tmpdat_squar = pd.concat((tmpdat_squar,pd.DataFrame(np.square(ff.iloc[:,x]))),axis=1)
		tmpdat_dersq = pd.concat((tmpdat_dersq,pd.DataFrame(np.square(np.diff(ff.iloc[:,x])))),axis=1)
	zerodat = pd.DataFrame(np.zeros(len(ff.columns))).T
	zerodat.columns = np.zeros(len(ff.columns))
	tmpdat_deriv.columns = np.zeros(len(ff.columns))
	tmpdat_squar.columns = np.zeros(len(ff.columns))
	tmpdat_dersq.columns = np.zeros(len(ff.columns))
	ff.columns = np.zeros(len(ff.columns))
    
	tmpdat_deriv = pd.concat((zerodat,tmpdat_deriv),axis=0).reset_index(drop=True)
	tmpdat_dersq = pd.concat((zerodat,tmpdat_dersq),axis=0).reset_index(drop=True)
    
	confound36p = pd.concat((ff,tmpdat_deriv,tmpdat_squar,tmpdat_dersq),axis=1).reset_index(drop=True)
	#write out confound file.
	confound36p.to_csv('%s/confounds_all.txt'%(output_dir), header=False, index=False, sep='\t', mode='a')
	#regress out confounds
	os.system("fsl_glm -i %s/func_mni_bp.nii.gz -d %s/confounds_all.txt --out_res=%s/func_mni_bp_clean.nii.gz"%(output_dir,output_dir,output_dir))


	########## 12: Time-Series Extraction and Connectivity estimation
	tseries_big = pd.DataFrame()
	for x in sorted(os.listdir('%s/PowerROIs_in_Nifti_format'%(resource_dir))):
		atpath = '%s/PowerROIs_in_Nifti_format/%s'%(resource_dir,x)
		atlas_filename = nib.load(atpath)
        
		fimg = nib.load('%s/func_mni_bp_clean.nii.gz'%(output_dir))
		masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
		tseries_big = pd.concat((tseries_big,pd.DataFrame(masker.fit_transform(fimg))),axis=1)
        
        
	tseries_big.to_csv("%s/timeseries_clean.csv"%(output_dir),index=False)
    	#create individual scan correlation matrix
	correlation_measure = ConnectivityMeasure(kind='correlation')
	cor_mat = correlation_measure.fit_transform([tseries_big.values])[0]
	np.fill_diagonal(cor_mat,0)
	cor_mat_z = np.arctanh(cor_mat)
	zma = pd.DataFrame(cor_mat_z)
	zma.to_csv("%s/cormat_z_GLM.csv"%(output_dir),index=False)
	#reshape correlation matrix
	#convert lower half to NA
	np.fill_diagonal(cor_mat_z,1)
	cor_z_long = pd.DataFrame(cor_mat_z)
	df = cor_z_long.where(np.triu(np.ones(cor_z_long.shape)).astype(np.bool))
	df = pd.DataFrame(df.stack().reset_index(drop=True))
	df.to_csv("%s/cormat_z_GLM_long.csv"%(output_dir),index=False)


	dat = pd.read_csv('%s/cormat_z_GLM.csv'%(output_dir))
	plt.figure()
	snsheatmap = sns.heatmap(dat,xticklabels=False,yticklabels=False,cmap='seismic')
	snsheatmap2 = snsheatmap.get_figure()
	snsheatmap2.savefig('%s/heatmap.png'%(output_dir))
	plt.clf()




	#put confound info into html file to check later
	os.system("echo '<p>===================== SUB %s ===========================<p>Heatmap %s <br><IMG BORDER=0 SRC=%s/heatmap.png WIDTH=100%s></BODY></HTML>' >> %s"%(subnum,output_dir, output_dir, '%', out_heatmap))



    
	print("################################################################")
	print("################################################################")
	print("################################################################")
	print("               DONE WITH SUBJECT %s       "%(subnum))

    
t2 = time.time()
tdiff = (t2-t1) / 60
print('finished script in %s minutes'%(np.round(tdiff,2)))
print(' ~ %s minutes per subject'%(np.round(tdiff/len(sublist),2)))



