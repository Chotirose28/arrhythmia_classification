import numpy as np
from PIL import Image
import os 
from tqdm import tqdm
import glob
from Parameters import Parameters
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def readGrayImg(filepath):
    img = np.array(Image.open(filepath).convert('L'))
    return list(img)

def ReadData():
    Xdata = []
    Ydata = []
    filename = []
    PeopleOfClass = []
    NumOfTry = []
    NumOfImage = []
    l = 1

    data_dir =  Parameters.Data_Directory
    local_list =  Parameters.Classes
    
    for idx,local in enumerate(local_list):
        num = 0
        num += len(glob.glob(data_dir + "/"+local + "/*"))
        for person in tqdm(glob.glob(data_dir+"/"+local+"/*")):
            NumOfTry.append(1)
            NumOfImage.append(int(len(os.listdir(person))/(2*l)))
            for N in range(int(len(os.listdir(person))/2)):
                if N%l == 0:
                #if N/l == N:
                    Xdata.append(list(readGrayImg(person+"/"+"2"+"_"+str(N+1)+".png")))
                    Ydata.append(idx)
                    filename.append(person+"/"+"2"+"_"+str(N+1)+".png") 
        PeopleOfClass.append(num)
        
    Numofsamples = []
    sumi = 0
    for i in NumOfTry:
        buf = 0
        for j in range(i):
            buf += NumOfImage[sumi]
            sumi += 1
        Numofsamples.append(buf)

    return np.array(Xdata), np.array(Ydata), PeopleOfClass,Numofsamples,np.array(filename)
    