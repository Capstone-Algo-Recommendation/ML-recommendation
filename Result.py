import tensorflow as tf
import pandas as pd
import numpy as np
import os

class Result:
    def __init__(self, MODEL_DIR, DATASET_DIR, data_no, batch_size):
        self.model = tf.keras.models.load_model(MODEL_DIR)
        self.problems =  pd.read_csv(os.path.join(DATASET_DIR , 'd'+data_no+'_problems.csv'))
        self.batch_size = batch_size
  
    def get_result(self, id):
        id = np.array(id).reshape(-1, 1)
        entry = []
        N = len(self.problems['problemId'])
        for i in range(0, N, self.batch_size):
            idxlist = range(i, min(i+self.batch_size, N))
            probs = np.array(idxlist).reshape(-1, 1)
            ids = np.array([id]*len(idxlist)).reshape(-1, 1)
            entry.append(self.model.predict([ids, probs]))

        entry = np.concatenate(entry)
        entry = np.array(entry).reshape(1, -1)
        ## 이미 푼 문제는 제외하고 리턴해야함--> 수정필요
        top_idx = np.argpartition(-entry, 50, axis=1)
        return self.problems.iloc[top_idx[0][:10],0].values
