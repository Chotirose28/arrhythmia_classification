import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import ReadData
from Parameters import Parameters
from tqdm import tqdm
import glob

def TrainData(Xdata, Ydata, n, k, file):
    index = np.arange(len(Ydata))
    TrainXdata = Xdata[index[index % k != n], :]
    TrainYdata = Ydata[index[index % k != n]]
    Trainfile = file[index[index % k != n]]
    return TrainXdata, TrainYdata, Trainfile

def TestData(Xdata, Ydata, n, k, file):
    index = np.arange(len(Ydata))
    TestXdata = Xdata[index[index % k == n], :]
    TestYdata = Ydata[index[index % k == n]]
    Testfile = file[index[index % k == n]]
    return TestXdata, TestYdata, Testfile

def TrainData_2lead(Xdata, Ydata, n, k, file):
    index = np.arange(len(Ydata))
    TrainXdata = Xdata[:, index[index % k != n], :]
    TrainXdata = np.transpose(TrainXdata, (1, 0, 2, 3, 4))
    TrainYdata = Ydata[index[index % k != n]]
    Trainfile = file[:, index[index % k != n]]
    return TrainXdata, TrainYdata, Trainfile
   
def TestData_2lead(Xdata, Ydata, n, k, file):
    index = np.arange(len(Ydata))
    TestXdata = Xdata[:, index[index % k == n], :]
    TestXdata = np.transpose(TestXdata, (1, 0, 2, 3, 4))
    TestYdata = Ydata[index[index % k == n]]
    Testfile = file[:, index[index % k == n]]
    return TestXdata, TestYdata, Testfile


def DivideData(Xdata, Ydata, file,n, k, PeopleofClass, Numofsamples):
    """Divide data corresponding to number of folds"""

    nextS = 0
    count = 0
    TraXstack = []
    TraYstack = []
    TesXstack = []
    TesYstack = []
    Trafilestack = []
    Tesfilestack = []

    for classNo in range(len(PeopleofClass)):
        buf1 = sum(Numofsamples[:sum(PeopleofClass[:classNo+1])])
        buf2 = sum(Numofsamples[:nextS])
        buf3 = sum(PeopleofClass[:classNo+1])

        dividedX = Xdata[buf2:buf1, :]
        dividedY = Ydata[buf2: buf1]
        dividedF = file[buf2: buf1]
        samplearr = Numofsamples[nextS:buf3]
        nextS += PeopleofClass[classNo]

        sampleStart = 0
        for i in range(len(samplearr)):
            NOF = sum(samplearr[:i+1])

            if count % k != n:
                TraXstack.append(dividedX[sampleStart: NOF, :])
                TraYstack.append(dividedY[sampleStart: NOF])
                Trafilestack.append(dividedF[sampleStart: NOF])

            else:
                TesXstack.append(dividedX[sampleStart: NOF, :])
                TesYstack.append(dividedY[sampleStart: NOF])
                Tesfilestack.append(dividedF[sampleStart: NOF])

            count += 1
            sampleStart += samplearr[i]

    TrainXdata = np.concatenate(TraXstack, axis=0)
    TrainYdata = np.concatenate(TraYstack, axis=0)
    TestXdata = np.concatenate(TesXstack, axis=0)
    TestYdata = np.concatenate(TesYstack, axis=0)
    TraFile = np.concatenate(Trafilestack, axis=0)
    TesFile = np.concatenate(Tesfilestack, axis=0)

    return TrainXdata, TrainYdata, TestXdata, TestYdata, TraFile, TesFile


def main():
    Xdata, Ydata, file = ReadData.readdata()
    Xdata = Xdata.astype("float32")/255.
    Xdata = Xdata[..., tf.newaxis]
    Ydata = to_categorical(Ydata, Parameters.NumofClass)

    k = 10
    for TrainTimes in range(k):
        TrainXdata, TrainYdata, Trainfile = TrainData(Xdata, Ydata, TrainTimes, k, file)
        Trainfile = [file for file in Trainfile if "VEB" in file or "VFW" in file]
        Trainfile = [file.replace('\\', '/').replace('Picture64','augmentation').replace('.png', '') for file in Trainfile]

        local_list = Parameters.Classes
        Ydata_idx = []
        for idx, local in enumerate(tqdm(local_list)):
            for files in Trainfile:
                if local in files:
                    Ydata_idx.append(idx)

        Xdata_aug = []
        Ydata_aug = []
        for idx, TrainfilePath in enumerate(Trainfile):
            for img in (glob.glob(TrainfilePath+"/*")):
                Xdata_aug.append(list(ReadData.readGrayImg(img)))
                Ydata_aug.append(Ydata_idx[idx])

        Xdata_aug = np.array(Xdata_aug)
        Xdata_aug = Xdata_aug[..., tf.newaxis]
        Ydata_aug = np.array(Ydata_aug)
        Ydata_aug = to_categorical(Ydata_aug, Parameters.NumofClass)
        print("Xdata_augのshape:", Xdata_aug.shape)
        print("Ydata_augのshape:", Ydata_aug.shape)
