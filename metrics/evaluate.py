import math

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras import backend as K
#Error Function for Competitive Data
def score(y_true,y_pred,a1=10,a2=13):
    score = 0
    d = y_pred - y_true
    for i in d:
        if i >= 0 :
            score += math.exp(i/a2) - 1
        else:
            score += math.exp(- i/a1) - 1
    return score


def score_func(y_true, y_pred):
    lst = [round(score(y_true, y_pred), 2),
           round(mean_absolute_error(y_true, y_pred), 2),
           round(mean_squared_error(y_true, y_pred), 2) ** 0.5,
           round(r2_score(y_true, y_pred), 2)]

    print(f' compatitive score {lst[0]}')
    print(f' mean absolute error {lst[1]}')
    print(f' root mean squared error {lst[2]}')
    print(f' R2 score {lst[3]}')
    return [lst[1], round(lst[2], 2), lst[3] * 100]

def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def tensorflow_models(model,dataset_dict,model_index,hp_config):
    df_result=pd.DataFrame([])

    for k,v in dataset_dict.items():
        if k in ['df_train','df_val','df_truth']:
            continue

        X_test,y_test=v[0],v[1]
        scores_test=model.evaluate(X_test,y_test)
        loss,MAE,R2=scores_test[0],scores_test[1],scores_test[2]
        print(f"\nloss: {scores_test[0]}")
        print('\nMAE: {}'.format(scores_test[1]))
        print('\nR^2: {}'.format(scores_test[2]))

        y_pred_test=model.predict(X_test)

        df_result['model_no']=model_index
        df_result.loc[:,]=[]


    return scores_test,y_pred_test,loss,MAE,R2
