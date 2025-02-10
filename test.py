import os
import numpy as np
from sklearn import metrics
import re
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from Parameters import Parameters
from focal_loss import FocalLoss
import ReadData
import DividedData_for_KCV
import Calc_EvaluationIndex

def test(TestXdata,TestYdata,TestTimes,tags,file):
    ans = TestYdata
    TestYdata = to_categorical(TestYdata, Parameters.NumofClass)
    Path = os.getcwd()
    TrainPath = Path + "/model" + "/train64_" + str(Parameters.NumofClass) + "class_" + str(Parameters.lead) +"lead_" + str(Parameters.epochs) + "epochs_" + str(Parameters.k) + "CV" + Parameters.option
    modelsave_Path = TrainPath + "/saved_model_"+str(TestTimes+1)
    new_model = load_model(modelsave_Path,custom_objects={"FocalLoss":FocalLoss})
    loss, acc = new_model.evaluate(TestXdata,TestYdata, verbose=2)
    y_pred =  tf.argmax(new_model.predict(TestXdata), axis=-1).numpy()
    
    # Extracting misclassified data
    Error_idx_list = [i for i in range(len(ans)) if ans[i] != y_pred[i]] 
    for idx in Error_idx_list:
        os.makedirs(TrainPath + "/" + "mistake", exist_ok = True)
        os.makedirs(TrainPath + "/" + "mistake" + "/"  + str(TestTimes+1), exist_ok = True)
        os.makedirs(TrainPath + "/" + "mistake" + "/"  + str(TestTimes+1) + "/" + tags[ans[idx]] + "_" + tags[y_pred[idx]], exist_ok = True)
        pil_img = Image.fromarray((np.squeeze(TestXdata, axis=3)[idx]*255).astype(np.uint8))
        pil_img.save(TrainPath + "/" + "mistake" + "/" + str(TestTimes+1) + "/" + tags[ans[idx]] + "_" + tags[y_pred[idx]] + "/" + file[idx] + ".png")
    
    ConfusionMatrix = metrics.confusion_matrix(ans, y_pred)
    
    print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
    Sensitivity = Calc_EvaluationIndex.calcSensitivity(ConfusionMatrix)
    Specificity = Calc_EvaluationIndex.calcSpecificity(ConfusionMatrix)
    Accuracy = Calc_EvaluationIndex.calcAccuracy(ConfusionMatrix)
        
    return ConfusionMatrix, Sensitivity, Specificity, Accuracy

def main():
    Xdata,Ydata,PeopleOfClass,Numofsamples,files = ReadData.ReadData()

    Xdata = Xdata.astype("float32")/255.
    Xdata = Xdata[...,tf.newaxis]

    if Parameters.NumofClass == 6 or Parameters.NumofClass == 8:
        local_list = Parameters.Classes
    elif Parameters.NumofClass == 2:
        local_list=["Nsr","Arrhythmia"]
        
    tag = [re.sub("[a-z,\-]","",str) for str in local_list]
    CV_Confusion_matrix = 0
    CV_Sensitivity = []
    CV_Specificity = []
    CV_Accuracy = []

    k = Parameters.k
    TestAll_start_time = time.time() 
    for TestTimes in range(k):
        if Parameters.NumofClass == 8 and Parameters.lead == 1 and Parameters.Sep_type == "intra":
            TestXdata, TestYdata, Testfile  = DividedData_for_KCV.TestData(Xdata, Ydata, TestTimes, k, files)
        elif Parameters.NumofClass == 6 and Parameters.lead == 1 and Parameters.Sep_type == "inter":
            TrainXdata, TrainYdata, TestXdata, TestYdata, Trainfile, Testfile = DividedData_for_KCV.DivideData(Xdata, Ydata, TestTimes, k, PeopleOfClass, Numofsamples,files)

            
        Testfile = [file.replace('\\','/').replace('Picture64/','').replace('.png','').replace('/','-') for file in Testfile]         
        ConfusionMatrix, Sensitivity, Specificity, Accuracy = test(TestXdata,TestYdata,TestTimes,tag,Testfile)
        CV_Confusion_matrix = ConfusionMatrix + CV_Confusion_matrix
        CV_Sensitivity.append(Sensitivity)
        CV_Specificity.append(Specificity)
        CV_Accuracy.append(Accuracy)
        
        print(str(TestTimes+1)+"CV:")
        print("ConfusionMatrix_multiclass:")
        Calc_EvaluationIndex.show_conf(ConfusionMatrix,tag)
        
        local_list_mini=["Arrhythmia","Nsr"]    
        tag_2class = [re.sub("[a-z,\-]","",str) for str in local_list_mini]
        cm = Calc_EvaluationIndex.multi2binary(ConfusionMatrix,tag) 
        
        print("ConfusionMatrix_2class:")
        Calc_EvaluationIndex.show_conf(cm,tag_2class)
        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]
        print("Sen:",TP/(TP+FN))
        print("Spe:",TN/(FP+TN))
        print("Acc:",(TP+TN)/(TP+TN+FP+FN))
        print("")
     
    #-------------------multi-------------------------        
    
    print("Confusion_matrix_all:")
    Calc_EvaluationIndex.show_conf(CV_Confusion_matrix,tag)
    print("Sensitivity:")
    Calc_EvaluationIndex.show_Sen(CV_Sensitivity,tag)
    print("")
    print("Specificity:")
    Calc_EvaluationIndex.show_Spe(CV_Specificity,tag)
    print("")
    print("Accuracy:")
    Calc_EvaluationIndex.show_Acc(CV_Accuracy)
    print("")
    
    #------------------binary---------------------------
    
    cm = Calc_EvaluationIndex.multi2binary(CV_Confusion_matrix,tag)
    local_list=["Arrhythmia","Nsr"]    
    tag = [re.sub("[a-z,\-]","",str) for str in local_list]
    print("Confusion_matrix_2class_all:")
    Calc_EvaluationIndex.show_conf(cm,tag)
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    print("Sen:",TP/(TP+FN))
    print("Spe:",TN/(FP+TN))
    print("Acc:",(TP+TN)/(TP+TN+FP+FN))
    print("")
    
    TestAll_elapsed_time = time.time() - TestAll_start_time # time consumption
    print ("TestAll_elapsed_time:{0}".format(TestAll_elapsed_time) + "[sec]")

if __name__ == "__main__":
    main()