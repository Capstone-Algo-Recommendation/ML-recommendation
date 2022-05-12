import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ast

class Result:
    def __init__(self, MODEL_DIR, DATASET_DIR, data_no, batch_size):
        self.model = tf.keras.models.load_model(MODEL_DIR)
        
        users = pd.read_csv(os.path.join(DATASET_DIR+'/preprocessed/data1', 'd'+data_no+'_users.csv'))
        problems =  pd.read_csv(os.path.join(DATASET_DIR+'/preprocessed/data1', 'd'+data_no+'_problems.csv'))
        solvedProblem = pd.read_csv(os.path.join(DATASET_DIR+'/raw_data', 'solvedProblem.csv'))

        self.N = len(problems['problemId'])
        self.id2idx_usr = {u[0]:i for i, u in enumerate(users.values)}
        self.id2prblist = {u[1]:u[3] for u in solvedProblem.values}
        self.idx2id_prb = {i:p[0] for i, p in enumerate(problems.values)}
        
        self.batch_size = batch_size
  
    def get_result(self, id):
        entry = []
        probs = list(set(range(0, self.N))-set(ast.literal_eval(self.id2prblist[id])))
        for i in range(0, self.N, self.batch_size):
            idxlist = probs[i:min(i+self.batch_size, self.N)]

            input_p = np.array(idxlist).reshape(-1, 1)
            input_u = np.array([self.id2idx_usr[id]]*len(idxlist)).reshape(-1, 1)

            entry.append(self.model.predict([input_u, input_p]))

        entry = np.concatenate(entry)
        entry = np.array(entry).reshape(1, -1)
        top_idx = np.argpartition(-entry, 30, axis=1)
        return [self.idx2id_prb[i] for i in top_idx[0]]