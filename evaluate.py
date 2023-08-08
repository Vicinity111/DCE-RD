# From https://github.com/majingCUHK/Rumor_RvNN/blob/master/model/evaluate.py

def evaluation_2class(prediction, y): # 4 dim
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    e = 0.000001
    for i in range(len(y)):
        ## Pre, Recall, F    

        Act = str(int(y[i][0].item()))  
        Pre = str(int(prediction[i][0].item()))

        ## for class 1
        if Act == '0' and Pre == '0': TP1 += 1
        if Act == '0' and Pre != '0': FN1 += 1    
        if Act != '0' and Pre == '0': FP1 += 1
        if Act != '0' and Pre != '0': TN1 += 1
        ## for class 2
        if Act == '1' and Pre == '1': TP2 += 1
        if Act == '1' and Pre != '1': FN2 += 1    
        if Act != '1' and Pre == '1': FP2 += 1
        if Act != '1' and Pre != '1': TN2 += 1    

    ## print result
    Acc_all = round( float(TP1+TP2)/float(len(y)+e), 4 )

    Prec1 = round( float(TP1)/float(TP1+FP1+e), 4 )
    Recll1 = round( float(TP1)/float(TP1+FN1+e), 4 )
    F1 = round( 2*Prec1*Recll1/(Prec1+Recll1+e), 4 )
    
    Prec2 = round( float(TP2)/float(TP2+FP2+e), 4 )
    Recll2 = round( float(TP2)/float(TP2+FN2+e), 4 )
    F2 = round( 2*Prec2*Recll2/(Prec2+Recll2+e), 4 )

    return Acc_all, Prec1, Recll1, F1,Prec2, Recll2, F2