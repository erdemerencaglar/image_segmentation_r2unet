import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):

    out_val = 0.0078
    SR = SR > threshold
    GT = GT != out_val
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)
    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    out_val = 0.0078
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT != out_val
    
    # TP : True Positive
    # FN : False Negative
    print("-----SR: ", SR)
    TP = ((SR==1)*(GT==1))==1
    FN = ((SR==0)*(GT==1))==1
    print("SR==1: ", SR==1)
    print("(GT==1): ", (GT==1))
    print("TP: ", torch.sum(TP))
    print("FN: ", torch.sum(FN))
    print("TP + FN: ", torch.sum(TP) + torch.sum(FN))
    SE = float(torch.sum(TP))/(float(torch.sum(TP) + torch.sum(FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    out_val = 0.0078
    SR = SR > threshold
    GT = GT != out_val

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)*(GT==0))==1
    FP = ((SR==1)*(GT==0))==1

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    out_val = 0.0078
    SR = SR > threshold
    GT = GT != out_val

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)*(GT==1))==1
    FP = ((SR==1)*(GT==0))==1

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    out_val = 0.0078
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT != out_val
    
    Inter = torch.sum((SR*GT)==1)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    out_val = 0.0078
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT != out_val

    Inter = torch.sum((SR*GT)==1)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC