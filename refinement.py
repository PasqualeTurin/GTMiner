import numpy as np
import config


def extend(y_pred, ids):

    for i in range(len(y_pred)):

        if y_pred[i] == config.relationships['INSIDE']:

            for j in range(len(y_pred)):

                if ids[i][0] in ids[j] and y_pred[j] == config.relationships['SAME_AS']:

                    if ids[i][0] == ids[j][0]:
                        to_check = [ids[j][1], ids[i][1]]
                    else:
                        to_check = [ids[j][0], ids[i][1]]

                    for k in range(i + 1, len(y_pred)):

                        if ids[k] == to_check:
                            y_pred[k] = 2.0

                    break
                    
                    
        if y_pred[i] == 1:
            
            for j in range(len(y_pred)):
                
                if ids[i][0] in ids[j] and i != j and y_pred[j] == 1:
                
                    if ids[i][0] == ids[j][0]:
                        to_check = ids[j][1]
                        
                    else:
                        to_check = ids[j][0]
                    
                    for k in range(len(y_pred)):
                        
                        if ids[i][1] in ids[k] and to_check in ids[k]:
                            
                            y_pred[k] = 1.0
                            break
                            
    return


def repair(y_pred, y_probs, ids):

    for i in range(len(y_pred)):

        if y_pred[i] == config.relationships['INSIDE']:

            for j in range(i + 1, len(y_pred)):

                if ids[i][0] == ids[j][0] and y_pred[j] == config.relationships['INSIDE'] and ids[i][1] != ids[j][1]:

                    if abs(y_probs[i] - y_probs[j]) > config.rep_diff and min(y_probs[i], y_probs[j]) < config.rep_thr:

                        if y_probs[i] < y_probs[j]:
                            y_pred[i] = 0.0

                        else:
                            y_pred[j] = 0.0

    return
