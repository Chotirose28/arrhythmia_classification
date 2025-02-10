import numpy as np

def calcSensitivity(confusion_matrix):
    NumOfClass = len(confusion_matrix[0])
    Sensitivity = np.zeros(NumOfClass)
    for i in range(NumOfClass):
        Sensitivity[i] = confusion_matrix[i,i] / confusion_matrix[i,:].sum()
    Sensitivity[np.isnan(Sensitivity)] = 0

    return Sensitivity

def calcSpecificity(confusion_matrix):
    NumOfClass = len(confusion_matrix[0])
    Specificity = np.zeros(NumOfClass)
    Diagonal = np.diag(confusion_matrix)
    for i in range(NumOfClass):
        Specificity[i] = (Diagonal[:i].sum() + Diagonal[i+1:].sum())/(confusion_matrix[:i,:].sum() + confusion_matrix[i+1:,:].sum())
    Specificity[np.isnan(Specificity)] = 0

    return Specificity
            
def calcAccuracy(confusion_matrix):
    Accuracy = np.diag(confusion_matrix).sum()/confusion_matrix.sum()
    return Accuracy
    
def show_conf(confusion_matrix,local_list):
    # Create confusion matrix in Table format ***local_list = list of class names
    print("Class",end="\t")
    for i,local in enumerate(local_list):
        print(local,end="\t")
    print()
        
    for i,local in enumerate(local_list):
        print(local,end="\t")
        for j in range(len(confusion_matrix[i])):
            print(confusion_matrix[i][j],end="\t")
        print()
    print()

def show_Sen(CV_Sensitivity, local_list):
    # Adjust sensitivity results appearance
    print("Fold",end="\t")
    for i in range(len(CV_Sensitivity)):
        print(str(i+1),end="\t\t")
    print("Ave")
    
    total = 0
    for i,local in enumerate(local_list):
        mean = 0
        print(local,end="\t")
        for j in range(len(CV_Sensitivity)):
            print("{:.6f}".format(CV_Sensitivity[j][i]),end="\t")
            mean += CV_Sensitivity[j][i]
            total += CV_Sensitivity[j][i]
        mean /= len(CV_Sensitivity)

        print("{:.6f}".format(mean))
        
def show_Spe(CV_Specificity, local_list):
    # Adjust specificity results appearance
    print("Fold",end="\t")
    for i in range(len(CV_Specificity)):
        print(str(i+1),end="\t\t")
    print("Ave")
    
    total = 0
    for i,local in enumerate(local_list):
        mean = 0
        print(local,end="\t")
        for j in range(len(CV_Specificity)):
            print("{:.6f}".format(CV_Specificity[j][i]),end="\t")
            mean += CV_Specificity[j][i]
            total += CV_Specificity[j][i]
        mean /= len(CV_Specificity)

        print("{:.6f}".format(mean))
            
def multi2binary(Confusion_matrix,local_list):
    # Compare between Normal (class Nsr) and Abnormal beat (other classes)
    TP = 0
    Diagonal = np.diag(Confusion_matrix)
    for i,local in enumerate(local_list):
        if local == "N":
            TN = Diagonal[i]
            FP = Confusion_matrix[i,:i].sum()+Confusion_matrix[i,i+1:].sum()
        else:
            TP += Diagonal[i] 
            
    FN = Confusion_matrix.sum()-(TN+FP+TP)
    cm = [[TP,FN],[FP,TN]]
    return cm
            
def show_Acc(CV_Accuracy):
    # Adjust accuracy results appearance

    print("Fold",end="\t")
    for i in range(len(CV_Accuracy)):
        print(str(i+1),end="\t\t")
    print("Ave")

    print("Acc",end="\t")
    mean = 0
    for Acc in CV_Accuracy:
        print("{:.6f}".format(Acc),end="\t")
        mean += Acc

    mean /= len(CV_Accuracy)

    print("{:.6f}".format(mean))