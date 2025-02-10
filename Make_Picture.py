import os 
import numpy as np
from matplotlib import pylab as plt
import glob
from tqdm import tqdm

path = os.getcwd()
DatasetPath = path + "\\Segment"
PicturePath = ".\Picture64"
PatientDatasetPathDirectorys = glob.glob(DatasetPath+"\\"+'*')
PatientDatasetPathDirectorys = [PatientDatasetPathDirectory.replace('\\','/') for PatientDatasetPathDirectory in PatientDatasetPathDirectorys]
DatasetPath = DatasetPath.replace("\\","/")
StartPatientDatasetPath = input("Start Patient DatasetPath :")
StartPatientDatasetPathDirectory = DatasetPath + "/"+ StartPatientDatasetPath
StartPatientDatasetPath_idx = PatientDatasetPathDirectorys.index(StartPatientDatasetPathDirectory)

for PatientdatasetPathDirectory in tqdm(PatientDatasetPathDirectorys[StartPatientDatasetPath_idx:]):
    DiseasePathDirectorys = glob.glob(PatientdatasetPathDirectory+"/*")
    DiseasePathDirectorys = [DiseasePathDirectory.replace("\\", "/") for DiseasePathDirectory in DiseasePathDirectorys]

    for DiseasePathDirectory in DiseasePathDirectorys:
        files = glob.glob(DiseasePathDirectory+"/*.csv")
        files = [file.replace("\\", "/") for file in files]

        #create 64x64 pixel image using the segmented peaks
        for file in tqdm(files):
            filename = file.replace(
                DiseasePathDirectory+"/", "").replace(".csv", "")
            Patinentnumber = PatientdatasetPathDirectory.replace(
                DatasetPath+"/", "")
            Diseasename = DiseasePathDirectory.replace(
                PatientdatasetPathDirectory+"/", "")
            ECG_data = np.loadtxt(file, delimiter=",")
            plt.figure(figsize=(0.64, 0.64)) 
            plt.axis('off')
            plt.plot(ECG_data, color="k")

            DirectoryOfSavefile = PicturePath + "\\" + Diseasename + "\\" + Patinentnumber
            os.makedirs(DirectoryOfSavefile, exist_ok=True)
            SaveFileName = DirectoryOfSavefile + "\\" + filename
            plt.savefig(SaveFileName+".png")
            plt.clf()
            plt.close()
            plt.cla()