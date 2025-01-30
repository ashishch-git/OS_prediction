from functools import reduce
import yaml
import os
import sys
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import regex as re
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))

def read_config():
    #print(dir_path)
    with open(dir_path + "/config.yaml","r") as file_object:
        config=yaml.load(file_object,Loader=yaml.SafeLoader)
        #print(config)
    return config

config = read_config()

def display_full(x):
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 20000,
                           "display.max_colwidth", None,
                           ):
        print(x)

def read_dicom(path):
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_series)
    vol_img = reader.Execute()
    return vol_img

def save_as_gz(vimg,path):
    writer = sitk.ImageFileWriter()
    #writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(path)
    writer.Execute(vimg)

def create_collage(path_c, path_s, save_path):
    '''

    Function to save simple collage from 0.0 and -90.0 degree png images

    '''

    image1 = Image.open(path_c)
    image2 = Image.open(path_s)
    collage = Image.new('RGB', (image1.size[0]*2, image1.size[1]))
    # Paste the first image onto the collage
    collage.paste(image1, (0, 0))
    # Paste the second image onto the collage
    collage.paste(image2, (image1.size[0], 0))

    # Save the collage
    collage.save(save_path)

def save_projections_as_png(image,img_name, invert = True):
    '''

    Function to save 2d simpleitk projection objects as uint8 png images

    '''

    writer = sitk.ImageFileWriter()
    writer.SetFileName(img_name)
    img_write= sitk.Flip(image, [False, True]) #flipping across y axis
    img_write=sitk.Cast(
        sitk.RescaleIntensity(img_write), #sitk.RescaleIntensity()
        sitk.sitkUInt8
    )
    if invert:
        img_write   = sitk.InvertIntensity(img_write,maximum=255)
    else:
        pass
    writer.Execute(img_write)  #sitk.Cast(sitk.RescaleIntensity(img,outputMinimum=0,outputMaximum=15)

def save_projections_as_nparr(image,img_name, invert = True):
    '''

    Function to save 2d simpleitk projection images as a numpy array

    '''

    img = sitk.Flip(image, [False, True])

    if invert:
        img= sitk.InvertIntensity(img,maximum=255)
    else:
        pass
    
    arr= sitk.GetArrayFromImage(img)

    # Perform min-max normalization

    minv,maxv= np.min(arr), np.max(arr)
    arr_normed = (arr - minv) / (maxv - minv)
    np.save(img_name,np.array(arr_normed))

def get_proj_after_mask(img):
    
    '''

    Function to get 3D masks for CT scans to segment out the exact tissue type.

    '''

    pix_array=sitk.GetArrayFromImage(img)
    max_i, min_i=float(pix_array.max()),float(pix_array.min())

    #multiply= sitk.MultiplyImageFilter()
    #if hu_type == 'Bone' or hu_type == 'bone' or hu_type == 'B':
    bone_mask = sitk.BinaryThreshold(
    img, lowerThreshold=200, upperThreshold=max_i,insideValue=1, outsideValue=0
    )
    
    #bone_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
        # path= img_n + r'_{0}_image.nii'.format(modality + '_' + type)
        # save_as_gz(op_img,path)
    #elif hu_type == 'Lean Tissue' or hu_type == 'lean' or hu_type == 'LT':
    lean_mask = sitk.BinaryThreshold(
    img, lowerThreshold=-29, upperThreshold=150, insideValue=1, outsideValue=0
    )
    #lean_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))

    #elif hu_type == 'Adipose' or hu_type == 'adipose' or hu_type == 'AT':
    adipose_mask = sitk.BinaryThreshold(
    img, lowerThreshold=-199, upperThreshold=-30, insideValue=1, outsideValue=0
    )
    #adipose_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
        
    #elif hu_type == 'Air' or hu_type == 'A':
    air_mask = sitk.BinaryThreshold(
    img, lowerThreshold=min_i, upperThreshold=-191, insideValue=1, outsideValue=0
    )
    #air_mask = multiply.Execute(img,sitk.Cast(seg,img.GetPixelID()))
    
    return bone_mask, lean_mask, adipose_mask, air_mask

def get_2D_projections(vol_img,modality,ptype,angle,invert_intensity = True, clip_value=15.0, t_type='N', save_img=True, save_nparr=True, img_n=''):
    """
    Input: PET/CT, Proj type, angle, min & max angle, HU Values/ mention: Bone/Adipose/Tissue

    Output: 2D sampled projections


    Notes:
    For Age,

    CT => lean sum of ,musles pixel, mean hu musle
    from musle take max PET  MIP

    CT, PT seperate

    CT(Grayscale) + PT(Color) = Overlap 2D img

    """

    projection = {'sum': sitk.SumProjection,
                'mean':  sitk.MeanProjection,
                'std': sitk.StandardDeviationProjection,
                'min': sitk.MinimumProjection,
                'max': sitk.MaximumProjection}
    
    paxis = 0
    rotation_axis = [0,0,1]
    rotation_angles = np.linspace(-np.pi/2, np.pi/2, int( (np.pi / (  ( angle / 180 ) * np.pi ) ) + 1 ) ) # angle range- [0, +180];
    rotation_center = vol_img.TransformContinuousIndexToPhysicalPoint(np.array(vol_img.GetSize())/2.0) #[(index-1)/2.0 for index in vol_img.GetSize()])

    rotation_transform = sitk.VersorRigid3DTransform()
    #rotation_transform = sitk.Euler3DTransform()
    rotation_transform.SetCenter(rotation_center)

    #Compute bounding box of rotating volume and the resampling grid structure
    image_indexes = list(zip([0,0,0], [sz-1 for sz in vol_img.GetSize()]))
    image_bounds = []
    for i in image_indexes[0]:
        for j in image_indexes[1]:
            for k in image_indexes[2]:
                image_bounds.append(vol_img.TransformIndexToPhysicalPoint([i,j,k]))

    all_points = []
    for ang in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, ang)     
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
        
    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)

    #resampling grid will be isotropic so no matter which direction we project to
    #the images we save will always be isotropic (required for vol_img formats that 
    #assume isotropy - jpg,png,tiff...)

    new_spc = [np.min(vol_img.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]
    pix_array=sitk.GetArrayFromImage(vol_img)
    maxtensity,mintensity=float(pix_array.max()),float(pix_array.min())
    
    """Setting a default pixel value based on modality (the resample function requires this argument as during rotation, 
                                      the pixel intensities for new locations are set to a default value) """
    if modality == 'CT':
        default_pix_val=-1024
    else:
        #Clipping intensities
        default_pix_val=0
        clamper = sitk.ClampImageFilter()
        clamper.SetLowerBound(0)
        clamper.SetUpperBound(clip_value)
        vol_img=clamper.Execute(vol_img)

    
    for ang in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, ang) 
        
        #Generate 3d volumes which are rotated by 'ang' angles
        resampled_image = sitk.Resample(image1=vol_img,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkLinear,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = vol_img.GetDirection(), #[1,0,0,0,1,0,0,0,1]
                                        defaultPixelValue = default_pix_val, 
                                        outputPixelType = vol_img.GetPixelID())
        
        #Generate 2d projections from the rotated volume
        if modality == 'CT_TSeg':
            arr = sitk.GetArrayFromImage(resampled_image)
            arr_new = (arr - arr.min()) / (arr.max() - arr.min())
            resampled_image1 = sitk.GetImageFromArray(arr_new)
            resampled_image1.CopyInformation(resampled_image)
            proj_image = projection[ptype](resampled_image1, paxis)
        else:
            proj_image = projection[ptype](resampled_image, paxis)

        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0 
        
        axes_shifted_pi=sitk.Extract(proj_image, extract_size) #flip axes

        if save_img:
            #Save the projections as image
            if (180 * ang/np.pi) != 90.0:
                imgname= img_n + r'{0}'.format((180 * ang/np.pi))
                save_projections_as_png(axes_shifted_pi, imgname + '.png', invert_intensity)
            #save_projections_as_nparr(axes_shifted_pi, imgname, invert_intensity)
        if save_nparr:
            #Save the projections as np array
            if (180 * ang/np.pi) != 90.0:
                imgname= img_n + r'{0}'.format((180 * ang/np.pi))
                save_projections_as_nparr(axes_shifted_pi, imgname, invert_intensity)
            
    print(f'Finished generating {int(180.0/angle)+1} - {ptype} intensity 2D projections from the {modality} volume image! ')

def compute_suv(vol_img, PatientWeight, AcquisitionTime , RadiopharmaceuticalStartTime, RadionuclideHalfLife, RadionuclideTotalDose):
    """
    Compute the SUV image from PET scans.
    """
    estimated = False

    raw = sitk.GetArrayFromImage(vol_img)    
    spacing = vol_img.GetSpacing()
    origin = vol_img.GetOrigin()
    direction = vol_img.GetDirection() 
    
    try:
        weight_grams = float(PatientWeight)*1000
    except:
        weight_grams = 76274 #average weight from master data
        estimated = True

    if weight_grams == 0:
        weight_grams = 76274 #average weight from master data
        estimated = True
        
    try:
        # Get Scan time
        scantime = datetime.strptime(AcquisitionTime,'%H%M%S')
        # Start Time for the Radiopharmaceutical Injection
        injection_time = datetime.strptime(RadiopharmaceuticalStartTime,'%H%M%S')
        # Half Life for Radionuclide # seconds
        half_life = float(RadionuclideHalfLife) 
        # Total dose injected for Radionuclide
        injected_dose = float(RadionuclideTotalDose)
        # Calculate decay
        decay = np.exp(-np.log(2)*((scantime-injection_time).seconds)/half_life)
        # Calculate the dose decayed during procedure
        injected_dose_decay = injected_dose*decay; # in Bq        
    except:
        decay = 0.61440 #average decay in metadata
        injected_dose = 265987763 #average injected dose in metadata #265 MBq
        injected_dose_decay = injected_dose * decay; 
        estimated = True
    
    # Calculate SUV # g/ml
    suv = raw*weight_grams/injected_dose_decay
    
    return suv, estimated, raw,spacing,origin,direction

""" def find_distorted_examinations(path_of_exams, path_to_save, new_pat_flag):
    
    #Used in order to find examinations that contain distorted slices.
    
    print(str(datetime.now()), ': Traversing over the exam directory...')
    directory_list = list()
    for root, dirs, files in os.walk(path_of_exams):#, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))

    directory_newlist = []
    if new_pat_flag:
        print(str(datetime.now()), ': Checking for only new patients...')
        new_pat_df = pd.read_excel(path_to_save + "new_patients_22Jan2024.xlsx")
        new_pat_ids = list(set(new_pat_df.patient_ID_new.to_list()))


        for dir in directory_list:
            for id in new_pat_ids:
                if id in dir:
                    directory_newlist.append(dir)
    else:
        directory_newlist = directory_list.copy()

    print(str(datetime.now()), ': Checking number of files in the dataset')    
    dataset = pd.DataFrame(directory_newlist, columns=['directory'])
    countfiles_selected = {"directory": [], "count":[]}

    for index, row in dataset.iterrows():
        count = 0
        for path in os.listdir(row["directory"]):
            if os.path.isfile(os.path.join(row["directory"], path)):
                count += 1
                
        countfiles_selected["directory"].append(row["directory"])
        countfiles_selected["count"].append(count)

    countfiles_selected_df = pd.DataFrame.from_dict(countfiles_selected)

    print(str(datetime.now()), ': Find the distorted exams with less thatn 179 dicom files')
    exams_with_distorted_images_file = countfiles_selected_df[countfiles_selected_df["count"] < 179].reset_index()
    exams_with_distorted_images_file[['source_directory', 'patient_directory', 'PET-CT_info']] = exams_with_distorted_images_file['directory'].str.rsplit(pat='/', n=2, expand=True)

    print(str(datetime.now()), ': Writing the distorted exams file')
    exams_with_distorted_images_file.to_excel(path_to_save + "exams_with_distorted_images_file.xlsx", index=False)

    print(str(datetime.now()), ': Writing the filtered exam file')
    dataset = dataset[~dataset.directory.isin(exams_with_distorted_images_file.directory)]
    dataset.to_excel(path_to_save + "data_ready_for_filtering.xlsx", index=False)
 """
def find_distorted_examinations(path_of_exams, path_all_exams, path_distorted_exams_dataframe):
    """
    Used in order to find examinations that contain distorted slices.
    """
    print(str(datetime.now()), ': Traversing over the exam directory...')
    directory_list = list()

    #total_files = sum([len(files) for root, dirs, files in tqdm(os.walk(path_of_exams))])
    #progress_bar = tqdm(total=total_files, desc="Processing Files")

    #for path_of_exams in tqdm(exams_folder_list):
    for root, dirs, files in os.walk(path_of_exams):#, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))
            #progress_bar.update(1)

    print(str(datetime.now()), ': Checking number of files in the dataset')    
    dataset = pd.DataFrame(directory_list, columns=['directory'])

    print(str(datetime.now()), ': Writing the all exam file')
    dataset.to_excel(path_all_exams, index=False)

    print(str(datetime.now()), ': Checking all the exams file')
    countfiles_selected = {"directory": [], "count":[]}

    total_files = len(directory_list)
    progress_bar = tqdm(total=total_files, desc="Processing Files")
    for index, row in dataset.iterrows():
        count = 0
        for path in os.listdir(row["directory"]):
            if os.path.isfile(os.path.join(row["directory"], path)):
                count += 1
        progress_bar.update(1)
                
        countfiles_selected["directory"].append(row["directory"])
        countfiles_selected["count"].append(count)

    countfiles_selected_df = pd.DataFrame.from_dict(countfiles_selected)

    print(str(datetime.now()), ': Find the distorted exams with less thatn 179 dicom files')
    exams_with_distorted_images_file = countfiles_selected_df[countfiles_selected_df["count"] < 179].reset_index()
    exams_with_distorted_images_file[['source_directory', 'patient_directory', 'PET-CT_info']] = exams_with_distorted_images_file['directory'].str.rsplit(pat='/', n=2, expand=True)

    print(str(datetime.now()), ': Writing the distorted exams file')
    exams_with_distorted_images_file.to_excel(path_distorted_exams_dataframe, index=False)

    #print(str(datetime.now()), ': Writing the filtered exam file')
    #dataset = dataset[~dataset.directory.isin(exams_with_distorted_images_file.directory)]
    

    return dataset, exams_with_distorted_images_file

def calculateAge(dob, scan_date):
    try:
        age = scan_date.year - dob.year - ((scan_date.month, scan_date.day) < (dob.month, dob.day))
    except:
        age = 0
    return age

def fn_final_date_col(diff_cd, diff_dd, diff_dd2):
    if diff_dd<diff_dd2:
        date_col = "DiaDate"
    elif diff_dd2<diff_dd:
        date_col = "DiaDate 2"
    else:
        date_col = 'consentDate'

    return date_col

def fn_final_diagnosis(x):
    d1 = x["translatedDiagnosis"]
    d2 = x["translatedDiagnosis 2"]

    if x["final_datecol"] == "DiaDate 2":
        d1 = ""
        diagnosis = d1 + d2
    else:
        d2 = ""
        diagnosis = d1 + d2

    return diagnosis

def fn_final_diff_col(final_datecol):
    if final_datecol == "DiaDate":
        diff_col = "scan_to_diagnosis1_days"
    elif final_datecol == "DiaDate 2":
        diff_col = "scan_to_diagnosis2_days"
    else:
        diff_col = "scan_to_consent_days"

    return diff_col

def linking_with_clinical_data(exams_data, clinical_data, dataframe_path, threshold_days=282):
    exams_linked_with_clinical_data_raw_path = dataframe_path + config["exams_linked_with_clinical_data_raw"]
    print(clinical_data.columns)

    exams_linked_with_clinical_data_scan_closer_to_diag = dataframe_path + config["exams_linked_with_clinical_data_scan_closer_to_diag"]
    exams_linked_with_clinical_data_remaining_pats_scan_closer_to_consent = dataframe_path + config["exams_linked_with_clinical_data_remaining_pats_scan_closer_to_consent"]
    
    #exams_linked_with_clinical_data_filtered = dataframe_path + config["exams_linked_with_clinical_data_filtered"]
    exams_linked_with_clinical_data_path = dataframe_path + config["exams_linked_with_clinical_data"]
    
    #print("exam:\n", exams_data[exams_data.patient_ID == "npr581224262306"])
    #print("clinical:\n", clinical_data[clinical_data.personReference == "npr581224262306"])

    linked_temp0 = pd.merge(exams_data, clinical_data, how="inner", left_on=["patient_ID"], right_on=["personReference"], suffixes=['','_'])
    print(str(datetime.now()), ': Shape for linked dataframe: ', linked_temp0.shape)

    #print("linked:\n", linked_temp[linked_temp.patient_ID == "npr581224262306"])
    #print(str(datetime.now()), ': Shape before dropping rows where diagnosis start date is empty: ', linked_temp.shape)
    #linked_temp.dropna(subset=["startDate"], inplace=True) 
    #print(str(datetime.now()), ': Shape after dropping rows where diagnosis start date is empty: ', linked_temp.shape)

    linked_temp0["scan_date"] = pd.to_datetime(linked_temp0["scan_date"], format="%Y%m%d")

    print(str(datetime.now()), ': Splitting the dataframe into two subsets based on availability of treatment start date')

    temp_noStartDate = linked_temp0[linked_temp0["startDate"].isna()].reset_index(drop=True).copy()
    print(str(datetime.now()), ': Shape for dataframe where treatment start date is not available: ', temp_noStartDate.shape)

    temp_withStartDate = linked_temp0[~linked_temp0["startDate"].isna()].reset_index(drop=True).copy()
    print(str(datetime.now()), ': Shape for dataframe where treatment start date is available: ', temp_withStartDate.shape)

    temp_withStartDate["startDate"] = pd.to_datetime(temp_withStartDate["startDate"], format="%Y-%m-%d")

    temp_withStartDate2 = temp_withStartDate[temp_withStartDate["scan_date"] <= temp_withStartDate["startDate"]].copy()
    print(str(datetime.now()), ': Shape after filtering where scan_date <= treatment start date: ', temp_withStartDate2.shape)

    print(str(datetime.now()), ': Combining both dataframes')
    linked_temp = pd.concat([temp_withStartDate2, temp_noStartDate], ignore_index=True).reset_index(drop=True)
    print(str(datetime.now()), ': Shape for dataframe after combining: ', linked_temp.shape)

    ##########
    #linked_temp["patientDOB"] = linked_temp.apply(lambda x: str(x["BirthYear"]) + (str(x["BirthMonth"]) if len(str(x["BirthMonth"]))>1 else "0" + str(x["BirthMonth"])) + "15", axis=1)
    #linked_temp["patientDOB"] = pd.to_datetime(linked_temp["patientDOB"], format="%Y%m%d")
    #linked_temp["age"] = linked_temp.apply(lambda x: calculateAge(x["patientDOB"], x["scan_date"]), axis=1)

    linked_temp["consentDate"] = pd.to_datetime(linked_temp["consentDate"], format="%Y-%m-%d")
    linked_temp["DiaDate"] = pd.to_datetime(linked_temp["DiaDate"], format="%Y-%m-%d")
    linked_temp["date"] = pd.to_datetime(linked_temp["date"], format="%Y-%m-%d")
    linked_temp["date 2"] = pd.to_datetime(linked_temp["date 2"], format="%Y-%m-%d")
    #linked_temp["startDate"] = pd.to_datetime(linked_temp["startDate"], format="%Y-%m-%d")
    linked_temp["deceasedDate"] = pd.to_datetime(linked_temp["deceasedDate"], format="%Y-%m-%d")
    linked_temp["survival_days"] = linked_temp.apply(lambda x: abs((x["deceasedDate"] - x["scan_date"])/np.timedelta64(1, 'D')), axis=1)
    linked_temp["survival_months"] = linked_temp.apply(lambda x: x["survival_days"]/30.0, axis=1)
    linked_temp["scan_to_consent_days"] = linked_temp.apply(lambda x: abs((x["scan_date"] - x["consentDate"])/np.timedelta64(1, 'D')), axis=1)
    linked_temp["scan_to_diagnosis1_days"] = linked_temp.apply(lambda x: abs((x["scan_date"] - x["DiaDate"])/np.timedelta64(1, 'D')), axis=1)
    linked_temp["scan_to_consent_days"] = linked_temp["scan_to_consent_days"].fillna(999999)
    linked_temp["scan_to_diagnosis1_days"] = linked_temp["scan_to_diagnosis1_days"].fillna(999999)
    linked_temp["survival_days"] = linked_temp["survival_days"].fillna(999999)
    linked_temp["survival_months"] = linked_temp["survival_months"].fillna(999999)

    #diagnosis date
    linked_temp["final_date_col"] = np.where(~linked_temp["DiaDate"].isna(), "DiaDate", np.where(~linked_temp["consentDate"].isna(), "consentDate", None))
    linked_temp["final_date"] = np.where(~linked_temp["DiaDate"].isna(), linked_temp["DiaDate"], np.where(~linked_temp["consentDate"].isna(), linked_temp["consentDate"], None))
    linked_temp["final_days"] = np.where(linked_temp["scan_to_diagnosis1_days"]!=999999, linked_temp["scan_to_diagnosis1_days"], linked_temp["scan_to_consent_days"])

    #consent date
    linked_temp["final_date_col2"] = np.where(~linked_temp["consentDate"].isna(), "consentDate", np.where(~linked_temp["DiaDate"].isna(), "DiaDate", None))
    linked_temp["final_date2"] = np.where(~linked_temp["consentDate"].isna(), linked_temp["consentDate"], np.where(~linked_temp["DiaDate"].isna(), linked_temp["DiaDate"], None))
    linked_temp["final_days2"] = np.where(linked_temp["scan_to_consent_days"]!=999999, linked_temp["scan_to_consent_days"], linked_temp["scan_to_diagnosis1_days"])

    linked_temp = linked_temp.sort_values(by=["patient_ID", "scan_date"])
    
    linked_temp.to_excel(exams_linked_with_clinical_data_raw_path, index=False)

    """
    linked_temp_noStartDate = linked_temp[linked_temp["startDate"].isna()]
    linked_temp_withStartDate = linked_temp[~linked_temp["startDate"].isna()] 
    print(str(datetime.now()), ': Shape before filtering where scan_date > diagnosis_date: ', linked_temp.shape)
    linked_temp = linked_temp[linked_temp_withStartDate["scan_date"] <= linked_temp_withStartDate["startDate"]] 
    print(str(datetime.now()), ': Shape after filtering where scan_date > diagnosis_date: ', linked_temp.shape)
    """
    

    """===========================first preference closer to diagnosis date============================="""
    print(str(datetime.now()), ': first preference closer to diagnosis date')

    linked_temp_closertodiag = linked_temp[linked_temp["final_days"]<=threshold_days].copy()
    print(str(datetime.now()), ': Shape for dataframe where scan date closer to diagnosis date is available: ', linked_temp_closertodiag.shape)

    linked_temp_closertodiag["scan_rank"] = linked_temp_closertodiag.groupby("patient_ID")["scan_date"].rank(method="first", ascending=True)
    linked_temp_closertodiag["diff_rank"] = linked_temp_closertodiag.groupby("patient_ID")["final_days"].rank(method="first", ascending=True)
    
    linked_temp_closertodiag2 = linked_temp_closertodiag[linked_temp_closertodiag["diff_rank"]==1].copy()
    #linked_temp["scan_rank"] = linked_temp.groupby("patient_ID")["scan_date"].rank(method="first", ascending=True)

    linked_temp_closertodiag2.to_excel(exams_linked_with_clinical_data_scan_closer_to_diag, index=False)

    """===========================second preference closer to consent date============================="""
    print(str(datetime.now()), ': second preference closer to consent date')

    linked_temp_closertodiag_list = list(linked_temp_closertodiag.patient_ID.unique())

    print(str(datetime.now()), ': filtering patients for whom scan date closer to diagnosis already selected')
    linked_temp_notclosertodiag = linked_temp[~linked_temp["patient_ID"].isin(linked_temp_closertodiag_list)].copy()
    print(str(datetime.now()), ': Shape for dataframe: ', linked_temp_notclosertodiag.shape)

    linked_temp_closertoconsent = linked_temp_notclosertodiag[linked_temp_notclosertodiag["final_days2"]<=threshold_days].copy()
    print(str(datetime.now()), ': Shape for dataframe where scan date closer to consent date is available: ', linked_temp_closertoconsent.shape)

    linked_temp_closertoconsent["scan_rank"] = linked_temp_closertoconsent.groupby("patient_ID")["scan_date"].rank(method="first", ascending=True)
    linked_temp_closertoconsent["diff_rank"] = linked_temp_closertoconsent.groupby("patient_ID")["final_days"].rank(method="first", ascending=True)
    
    linked_temp_closertoconsent2 = linked_temp_closertoconsent[linked_temp_closertoconsent["diff_rank"]==1].copy()

    linked_temp_closertoconsent2.to_excel(exams_linked_with_clinical_data_remaining_pats_scan_closer_to_consent, index=False)

    #linked_temp2.to_excel(exams_linked_with_clinical_data_filtered, index=False)    
    
    linked_exams = pd.concat([linked_temp_closertodiag2, linked_temp_closertoconsent2], ignore_index=True).reset_index(drop=True)

    linked_exams.to_excel(exams_linked_with_clinical_data_path, index=False)

    return linked_exams


def natural_sortkey(string):          
    tokenize = re.compile(r'(\d+)|(\D+)').findall
    return tuple(int(num) if num else alpha for num, alpha in tokenize(string.name))

def best_model_selection_from_fold(system, type, category, experiment_number, fold_number):
    """
    Fetch the best model from a selected fold and returns the path and the epoch of this model.
    """
    if type == "regression":
        path = config["Source"]["paths"][f"source_path_system_{system}"] + config["regression_path"] + f"/Experiment_{experiment_number}/CV_{fold_number}/Network_Weights/"
    else:
        path = config["Source"]["paths"][f"source_path_system_{system}"] + config["classification_path"] + f"/{category}/Experiment_{experiment_number}/CV_{fold_number}/Network_Weights/"
    path_object = Path(path)
    models = path_object.glob("*")
    models_sorted = sorted(models, key=natural_sortkey)
    best_model_path = path + [model.name for model in models_sorted][-1]
    epoch_to_continue = best_model_path.split("_")[-1].split(".")[0]
    return best_model_path, epoch_to_continue
    
def load_checkpoints(system, type, category, experiment_number, fold_number):
    """
    Function used to load the checkpoint of the model when retraining the model.
    """

    if fold_number == 0:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 1:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 2:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 3:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 4:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 5:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 6:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 7:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 8:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    elif fold_number == 9:
        checkpoint_path, epoch_to_continue = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
    
    return checkpoint_path, epoch_to_continue

def best_model_selection_from_fold_metric_based(system, type, category, experiment_number, fold_number):
    """
    Return the best model of the selected fold based on the metrics.
    """

    repls = ("Network_Weights/best_model", "Metrics/epoch"), ("pth.tar", "csv")
    
    if type == "regression":
        path = config["Source"]["paths"][f"source_path_system_{system}"] + config["regression_path"] + f"/Experiment_{experiment_number}/CV_{fold_number}/Network_Weights/"
    else:
        path = config["Source"]["paths"][f"source_path_system_{system}"] + config["classification_path"] + f"/{category}/Experiment_{experiment_number}/CV_{fold_number}/Network_Weights/"
    
    model_files = []
    for dirs, subdirs, files in os.walk(path):
        for file in files:
            file_path = str(os.path.join(dirs, file))
            file_path = file_path.replace('\\','/')
            model_files.append(file_path)

    sorted(model_files)

    model_dict = dict()

          
    if type=="regression":   
        best_metric = 100000000.0
    else:
        best_metric = -1

    for model_path in model_files:
        metric_path = reduce(lambda a, kv: a.replace(*kv), repls, model_path)
        metric_data = pd.read_csv(metric_path)

        if type=="regression":   
            metric = mean_absolute_error(metric_data["GT"], metric_data["prediction (age)"])
            if metric < best_metric:
                best_metric = metric    
        else:
            if category=="Diagnosis":
                metric = cohen_kappa_score(metric_data["prediction"], metric_data["GT"])
                if metric > best_metric:
                    best_metric = metric
            elif category=="Overall_Survival":
                auc, _ = calculate_metrics(metric_data["prediction_probability (os)"], np.array(metric_data["GT"]).astype(int))
                metric = auc
                if metric > best_metric:
                    best_metric = metric   
            else:
                auc, _ = calculate_metrics(metric_data["prediction_probability (sex)"], np.array(metric_data["GT"]).astype(int))
                metric = auc
                if metric > best_metric:
                    best_metric = metric   

        epoch_num = model_path.split("_")[-1].split(".")[0]
        #print(metric, epoch_num, model_path)
        if metric in model_dict.keys():
            if epoch_num < model_dict[metric][1]:
                model_dict[metric] = (model_path, epoch_num)
            else:
                pass
        else:
            model_dict[metric] = (model_path, epoch_num)
          
    #print(model_dict)
    best_model_path = model_dict[best_metric][0]
    epoch_to_continue = model_dict[best_metric][1] #best_model_path.split("_")[-1].split(".")[0]
    #print("best: ", best_metric, epoch_to_continue, best_model_path)
    return best_model_path, epoch_to_continue

def calculate_metrics(pred_prob, GT):
    """
    Calculate the neccessary metrics to evaluate the model.
    """

    fpr, tpr, thresholds = roc_curve(GT, pred_prob, pos_label=1)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (pred_prob >= optimal_threshold).astype(int)
    #print("prediction: ", pred_labels)
    #print("GT: ", GT)

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP = ((pred_labels == 1) & (GT == 1)).sum()
    TN = ((pred_labels == 0) & (GT == 0)).sum()
    FP = ((pred_labels == 1) & (GT == 0)).sum()
    FN = ((pred_labels == 0) & (GT == 1)).sum()
    sensitivity = TP / (TP + FN)
    # precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    auc_score = auc(fpr, tpr)

    results = [
        ["True Positives (TP)", TP],
        ["True Negatives (TN)", TN],
        ["False Positives (FP)", FP],
        ["False Negatives (FN)", FN],
        ["Sensitivity", sensitivity],
        # ["Precision", precision],
        ["Specificity", specificity],
        ["AUC", auc_score]
    ]
    # Print results in tabular form
    # print(tabulate(results, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    return auc_score, results

def evaluate_best_models_all_folds_metric_based(system, type, category, experiment_number, folds_list):
    """
    Evaluate the best models from the experiment based on their metrics.
    """

    repls = ("Network_Weights/best_model", "Metrics/epoch"), ("pth.tar", "csv")

    auc_from_all_folds = []
    metric_data_list = []
    for fold_number in folds_list:
        best_model_path, _ = best_model_selection_from_fold_metric_based(system, type, category, experiment_number, fold_number)
        best_metric_path = reduce(lambda a, kv: a.replace(*kv), repls, best_model_path)
        print(best_metric_path)
        metric_data = pd.read_csv(best_metric_path)
        metric_data_list.append(metric_data)

    all_metric_df = pd.concat(metric_data_list)

    if type=="regression":        
        metric_r_squared = r2_score(all_metric_df["GT"], all_metric_df["prediction (age)"])
        metric_mae = mean_absolute_error(all_metric_df["GT"], all_metric_df["prediction (age)"])
        metrics = {"metric_mae":metric_mae, "metric_r_squared":metric_r_squared}
        return metrics
    else:
        if category=="Diagnosis":
            c_k_score = cohen_kappa_score(all_metric_df["prediction"], all_metric_df["GT"])
            idx_classes = ["C81_GT", "C83_GT", "Others_GT"]
            col_classes = ["C81_Pred", "C83_Pred", "Others_Pred"]
            confusion_matrix_df = pd.DataFrame(confusion_matrix(all_metric_df["GT"], all_metric_df["prediction"]), columns=col_classes, index=idx_classes)
            metrics = {"c_k_score":c_k_score, "confusion_matrix":confusion_matrix_df}
            return metrics
        elif category=="Overall_Survival":
            for fold in folds_list:
                best_model_path, _ = best_model_selection_from_fold_metric_based(system, type, category, experiment_number, fold)
                best_metric_path = reduce(lambda a, kv: a.replace(*kv), repls, best_model_path)
                print(best_metric_path)
                metric_data = pd.read_csv(best_metric_path)
                #print(metric_data.columns)
                auc, _  = calculate_metrics(metric_data["prediction_probability (os)"], np.array(metric_data["GT"]).astype(int) )  #"prediction_probability_male" for exp1, prediction_probability (sex) for exp 2
                auc_from_all_folds.append(auc)
            return np.mean(auc_from_all_folds)
        else:
            for fold in folds_list:
                best_model_path, _ = best_model_selection_from_fold_metric_based(system, type, category, experiment_number, fold)
                best_metric_path = reduce(lambda a, kv: a.replace(*kv), repls, best_model_path)
                print(best_metric_path)
                metric_data = pd.read_csv(best_metric_path)
                #print(metric_data.columns)
                auc, _  = calculate_metrics(metric_data["prediction_probability (sex)"], np.array(metric_data["GT"]).astype(int) )  #"prediction_probability_male" for exp1, prediction_probability (sex) for exp 2
                auc_from_all_folds.append(auc)
            return np.mean(auc_from_all_folds)

def evaluate_best_models_all_folds(system, type, category, experiment_number, folds_list):
    """
    Evaluate the best models from all the folds.
    """
    
    repls = ("Network_Weights/best_model", "Metrics/epoch"), ("pth.tar", "csv")

    auc_from_all_folds = []
    metric_data_list = []
    for fold_number in folds_list:
        best_model_path, _ = best_model_selection_from_fold(system, type, category, experiment_number, fold_number)
        best_metric_path = reduce(lambda a, kv: a.replace(*kv), repls, best_model_path)
        print(best_metric_path)
        metric_data = pd.read_csv(best_metric_path)
        metric_data_list.append(metric_data)

    all_metric_df = pd.concat(metric_data_list)

    if type=="regression":        
        metric_r_squared = r2_score(all_metric_df["GT"], all_metric_df["prediction (age)"])
        metric_mae = mean_absolute_error(all_metric_df["GT"], all_metric_df["prediction (age)"])
        metrics = {"metric_mae":metric_mae, "metric_r_squared":metric_r_squared}
        return metrics
    else:
        if category=="Diagnosis":
            c_k_score = cohen_kappa_score(all_metric_df["prediction"], all_metric_df["GT"])
            idx_classes = ["C81_GT", "C83_GT", "Others_GT"]
            col_classes = ["C81_Pred", "C83_Pred", "Others_Pred"]
            confusion_matrix_df = pd.DataFrame(confusion_matrix(all_metric_df["GT"], all_metric_df["prediction"]), columns=col_classes, index=idx_classes)
            metrics = {"c_k_score":c_k_score, "confusion_matrix":confusion_matrix_df}
            return metrics
        elif category=="Overall_Survival":
            for fold in folds_list:
                best_model_path, _ = best_model_selection_from_fold(system, type, category, experiment_number, fold)
                best_metric_path = reduce(lambda a, kv: a.replace(*kv), repls, best_model_path)
                print(best_metric_path)
                metric_data = pd.read_csv(best_metric_path)
                #print(metric_data.columns)
                idx_classes = ["OSTrue_GT", "OSFalse_GT"]
                col_classes = ["OSTrue_Pred", "OSFalse_Pred"]
                confusion_matrix_df = pd.DataFrame(confusion_matrix(all_metric_df["GT"], all_metric_df["prediction"]), columns=col_classes, index=idx_classes)
                auc, _  = calculate_metrics(metric_data["prediction_probability (os)"], np.array(metric_data["GT"]).astype(int) )  #"prediction_probability_male" for exp1, prediction_probability (sex) for exp 2
                auc_from_all_folds.append(auc)
                metrics={'mean AuC':np.mean(auc_from_all_folds), 'confusion matrix':confusion_matrix_df }
            return metrics
        else:
            for fold in folds_list:
                best_model_path, _ = best_model_selection_from_fold(system, type, category, experiment_number, fold)
                best_metric_path = reduce(lambda a, kv: a.replace(*kv), repls, best_model_path)
                print(best_metric_path)
                metric_data = pd.read_csv(best_metric_path)
                #print(metric_data.columns)
                idx_classes = ["Male", "Female"]
                col_classes = ["Male_Pred", "Female_Pred"]
                confusion_matrix_df = pd.DataFrame(confusion_matrix(all_metric_df["GT"], all_metric_df["prediction"]), columns=col_classes, index=idx_classes)
                auc, _  = calculate_metrics(metric_data["prediction_probability (sex)"], np.array(metric_data["GT"]).astype(int) )  #"prediction_probability_male" for exp1, prediction_probability (sex) for exp 2
                auc_from_all_folds.append(auc)
                metrics={'mean AuC':np.mean(auc_from_all_folds), 'confusion matrix':confusion_matrix_df }
            return metrics

