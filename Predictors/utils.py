import os 
import glob
import numpy as np
import pandas as pd

def remove_amb_char(df):
    NON_CODE = "B|Z|J|U|O|X"
    df = df[~df["Sequence"].str.contains(NON_CODE, regex=True)]
    return df

def mergeDF(path,to_dir,file_name):
    appended_data = []
    for infile in glob.glob(path):
        #print(infile)
        data = pd.read_pickle(infile)
        # store DataFrame in list
        appended_data.append(data)
    result_path=to_dir+"/"+file_name
    print("Save:",result_path)
    appended_data = pd.concat(appended_data)
    try:
        appended_data['ID'] = appended_data['ID'].astype(int)
    except:
        pass
    appended_data.sort_values(by=['ID'], inplace=True)
    appended_data.to_pickle(result_path)
    return appended_data


def predict_withModel(path,to_dir,existing_result,model):
    appended_pkl = []
    for infile in glob.glob(path):  
     #print("Read:",infile)
        file_name = os.path.basename(infile)
        result_path=to_dir+"/"+file_name.replace("pkl", "ML.pkl")
        if result_path in existing_result :
        # print("found then skip : " , result)
            continue
        else:
            df = pd.read_pickle(infile)
            ready_df =df[[ "reps"]]
            
            X= np.array(df['reps'].to_list())
            _y = model.predict(X)
            df.drop(columns=['reps'],inplace =True)
            df['class'] = _y
            #print("Save:",result_path)
            df.to_pickle(result_path) 

    print("Complete")


def predictprob_withModel(path,to_dir,existing_result,model,threshold=0.5):
    appended_pkl = []
    for infile in glob.glob(path):  
     #print("Read:",infile)
        file_name = os.path.basename(infile)
        result_path=to_dir+"/"+file_name.replace("pkl", "ML.pkl")
        if result_path in existing_result :
        # print("found then skip : " , result)
            continue
        else:
            df = pd.read_pickle(infile)
            ready_df =df[[ "reps"]]
            
            x= np.array(df['reps'].to_list())
            predicted_proba = model.predict_proba(x)
            _y = (predicted_proba [:,1] >= threshold).astype('int')
            #_y = model.predict(X)
            df.drop(columns=['reps'],inplace =True)
            df['class'] = _y
            #print("Save:",result_path)
            df.to_pickle(result_path) 

    print("Complete")    
    

def read_exists_result(path):
    existing_result = []
    for infile in glob.glob(path+"/*.pkl"):
        # print(infile)
        existing_result.append(infile)
    print(len(existing_result))
    return existing_result

def predict_CNN(path,to_dir,existing_result,learner):
    appended_pkl = []
    for infile in glob.glob(path):  
     #print("Read:",infile)
        file_name = os.path.basename(infile)
        result_path=to_dir+"/"+file_name.replace("pkl", "ML.pkl")
        if result_path in existing_result :
        # print("found then skip : " , result)
            continue
        else:
            df = pd.read_pickle(infile)
            
            X= np.array(df['reps'].to_list())
            X_test = np.reshape(X,(X.shape[0],X.shape[1],1))
            y_probas = learner.predict(X_test)
            threshold = 0.5
            _y = np.where(y_probas > threshold, 1, 0)
            df.drop(columns=['reps'],inplace =True)
            df['class'] = _y
            #print("Save:",result_path)
            df.to_pickle(result_path) 

    print("Complete")

def predict_CNN_PWM(path,to_dir,existing_result,learner):
    appended_pkl = []
    for infile in glob.glob(path):  
     #print("Read:",infile)
        file_name = os.path.basename(infile)
        result_path=to_dir+"/"+file_name.replace("pkl", "ML.pkl")
        if result_path in existing_result :
        # print("found then skip : " , result)
            continue
        else:
            df = pd.read_pickle(infile)

            X= np.array(df['reps'].to_list())
            X_test = X.reshape(len(X),1,20,20)
            y_probas = learner.predict(X_test)
            threshold = 0.5
            _y = np.where(y_probas > threshold, 1, 0)
            df.drop(columns=['reps'],inplace =True)
            df['class'] = _y
            #print("Save:",result_path)
            df.to_pickle(result_path) 

    print("Complete")