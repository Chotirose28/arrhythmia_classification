import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from Parameters import Parameters
import ReadData
import DividedData_for_KCV
import time
from CNN_Architecture import CNN
from focal_loss import FocalLoss

def train(TrainXdata, TrainYdata, TestXdata, TestYdata, TrainTimes):
    Path = os.getcwd()
    TrainPath = Path + "/model"+"/train64_" + str(Parameters.NumofClass) + "class_" + str(
        Parameters.lead) + "lead_" + str(Parameters.epochs) + "epochs_" + str(Parameters.k) + "CV" + Parameters.option
    checkpoint_Path = TrainPath + "/training_" + \
        str(TrainTimes+1)+"/cp-{epoch:04d}.ckpt"
    modelsave_Path = TrainPath + "/saved_model_"+str(TrainTimes+1)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_Path,
        verbose=1,
        save_weights_only=True,
        save_freq="epoch"
    )

    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min')

    adam = Adam(lr=0.001)

    model = CNN()
    model.compile(optimizer=adam,
                  loss=FocalLoss(gamma=1, alpha=2.5),
                  # loss = "categorical_crossentropy",
                  metrics=['accuracy'])

    # model.save_weights(checkpoint_Path.format(epoch=10))

    history = model.fit(TrainXdata,
                        TrainYdata,
                        batch_size=Parameters.batch_size,
                        epochs=Parameters.epochs,
                        # callbacks=es_callback,
                        # callbacks=[cp_callback,es_callback],
                        validation_data=(TestXdata, TestYdata),
                        verbose=1)

    model.save(modelsave_Path)

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()

    DirectoryOfSavefile = TrainPath
    os.makedirs(DirectoryOfSavefile, exist_ok=True)
    SaveFileName = DirectoryOfSavefile + "/" + "CV"+str(TrainTimes+1)
    plt.savefig(SaveFileName+".png")
    plt.close()


def main():
    Xdata, Ydata, PeopleOfClass, Numofsamples, files = ReadData.ReadData()

    Xdata = Xdata.astype("float32")/255.
    Xdata = Xdata[..., tf.newaxis]
    Ydata = to_categorical(Ydata, Parameters.NumofClass)

    k = Parameters.k
    TrainAll_start_time = time.time()
    for TrainTimes in range(k):
        print("")
        print("TrainTimes:", TrainTimes+1)
        
        if Parameters.NumofClass == 8 and Parameters.lead == 1 and Parameters.Sep_type == "intra":
            TrainXdata, TrainYdata, Trainfile = DividedData_for_KCV.TrainData(Xdata, Ydata, TrainTimes, k, files)
            TestXdata, TestYdata, Testfile = DividedData_for_KCV.TestData(Xdata, Ydata, TrainTimes, k, files)
            
        elif Parameters.NumofClass == 6 and Parameters.lead == 1 and Parameters.Sep_type == "inter":
            TrainXdata, TrainYdata, TestXdata, TestYdata, Trainfile,Testfile = DividedData_for_KCV.DivideData(Xdata, Ydata, files, TrainTimes, k, PeopleOfClass, Numofsamples)
            
        train(TrainXdata, TrainYdata, TestXdata, TestYdata, TrainTimes)

    TrainAll_elapsed_time = time.time() - TrainAll_start_time  # time consumption
    print("TrainAll_elapsed_time:{0}".format(TrainAll_elapsed_time) + "[sec]")


if __name__ == "__main__":
    main()
