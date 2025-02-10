import os
import glob
import wfdb
import numpy as np
from wfdb import processing
from tqdm import tqdm

# 切り出したデータを保存する関数
def save_segment(SegmentPath,PatientDatasetPath,Heart_type,Count,Segment_List,Lead):
    DirectoryOfSavefile = SegmentPath + "\\" + PatientDatasetPath + "\\" + Heart_type
    os.makedirs(DirectoryOfSavefile,exist_ok = True)
    SaveFileName = DirectoryOfSavefile + "/" + Lead + "_" + str(Count)  
    np.savetxt(SaveFileName +".csv",Segment_List,fmt = "%0.10f",delimiter=",")

path = os.getcwd() 
Dataset = "\\mitdb" 
DatasetPath = path + Dataset 
SegmentPath = ".\Segment" 
PatientDatasetPathDirectorys = glob.glob(DatasetPath+"\\"+'*')  
PatientDatasetPathDirectorys = [PatientDatasetPathDirectory.replace('\\','/') for PatientDatasetPathDirectory in PatientDatasetPathDirectorys]
DatasetPath = DatasetPath.replace("\\","/") 
StartPatientDatasetPath = input("Start Patient DatasetPath :") 
StartPatientDatasetPathDirectory = DatasetPath + "/"+ StartPatientDatasetPath
StartPatientDatasetPath_idx = PatientDatasetPathDirectorys.index(StartPatientDatasetPathDirectory)

for PatientdatasetPathDirectory in tqdm(PatientDatasetPathDirectorys[StartPatientDatasetPath_idx:]):
    PatientDatasetPath = PatientdatasetPathDirectory.replace(DatasetPath,"").replace("/","")
    DirectoryOfSavefile = SegmentPath + "\\" + PatientDatasetPath 
    os.makedirs(DirectoryOfSavefile,exist_ok = True)
    
    record_name = os.path.join(path + Dataset + "\\" +str(PatientDatasetPath), str(PatientDatasetPath))
    signals, fields = wfdb.rdsamp(record_name) # read signal and its fields
    ann = wfdb.rdann(record_name,'atr') # read annotation in the signal
    
    resamp_sig,resamp_ann = processing.resample_singlechan(x=signals[:,0],ann=ann,fs=360,fs_target=1000) # resample signal, from 360Hz to 1000Hz
    symbols = resamp_ann.symbol   # annotations' symbols
    aux_note = resamp_ann.aux_note # aux note
    positions = resamp_ann.sample # annotations' positions list
    
    # Peak segmentation

    for i in range(len(symbols)):
        if symbols[i] == "+":
            symbols[i] = aux_note[i]
    
    # symbols(annotations) list is corresponding to https://archive.physionet.org/physiobank/annotations.shtml
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    h = 0
    j = 0
     
    for i in range(2,len(symbols)-2):
        count = 0
        start = positions[i-1] 
        end = positions[i+1] 
        lead = "2"
     
        if start >= 0 and end <= len(resamp_sig):
            segment = resamp_sig[start:end]
            if symbols[i] == "N":
                heart_type = "Nsr"
                a = a + 1
                count = a
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)
                
            elif symbols[i] == "V":
                heart_type = "PVC"
                b = b + 1
                count = b
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)   
                
            elif symbols[i] == "/":
                heart_type = "PAB"
                c = c + 1
                count = c
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)
                
            elif symbols[i] == "R":
                heart_type = "RBB"
                d = d + 1
                count = d
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)
      
            elif symbols[i] == "L":
                heart_type = "LBB"
                e = e + 1
                count = e
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)
                
            elif symbols[i] == "A" or symbols[i] == "a":
                heart_type = "APC"
                f = f + 1
                count = f
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)
                
            elif symbols[i] == "!":
                heart_type = "VFW"
                g = g + 1
                count = g
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)
                
            elif symbols[i] == "E":
                heart_type = "VEB"  
                h = h + 1
                count = h
                save_segment(SegmentPath,PatientDatasetPath,heart_type,count,segment,lead)
                  