import numpy as np
import os

def var_detection(top_logps:list) ->float:
    variance = np.var(top_logps)
    return float(variance)

def confidence_detection(logpr:float) ->float:
    confidence = np.round(np.exp(logpr)*100,2)
    return float(confidence)  