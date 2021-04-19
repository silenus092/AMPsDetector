import pickle
import numpy as np
import glob
import os 
import pandas as pd
import matplotlib.pyplot as plt

char_code_list = [char for char in "ACDEFGHIKLMNPQRSTVWY"] 

def scale_by(x_ar):
    # sigmod function 
    return 1 / (1 + np.exp(x_ar * -1)) 

def convert_to20X20_(pssm , index_pattern):
    """
    
    pssm: dcit
        get pssm as dict 
    """
    pssm_df = pd.DataFrame.from_dict(pssm)
    pssm_df.index = index_pattern
    result_df = pd.DataFrame(0, columns=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'],
    index = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
    for  char in char_code_list: # column
        for index, row in pssm_df.iterrows():
            result_df.loc[index, char] += round(row[char],2)
    return result_df

def get_pssm(m):
    pwm = m.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()
    return pssm

def create_image(df,file):
    plt.axis("off")
    plt.imshow(df)
    plt.savefig(file, bbox_inches='tight', pad_inches=0)