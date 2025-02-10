class Parameters:
    """8 classes (10-fold cv)"""
    Classes = ["Nsr","PVC","PAB","RBB","LBB","APC","VFW","VEB"]
    """6 classes (leave-one-patient-out)"""
    # Classes = ["Nsr","PVC","PAB","RBB","LBB","APC"]
    """Number of signal leads"""
    lead = 1
    """Number of classes"""
    NumofClass = len(Classes)
    """Data directory"""
    Data_Directory = "Picture64" # Picture64 or Picture64_detrend
    """Experiment setting selection"""        
    Sep_type = "intra" # intra (10-fold cv) or inter (leave-one-patient-out)
    """Number of folds"""
    k = 10 # 10 or 2
    batch_size = 128
    epochs = 40 # 40 or 100
    """Loss function selection"""
    option = "" # "" or "_focal"
