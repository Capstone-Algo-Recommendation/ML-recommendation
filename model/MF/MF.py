import tensorflow as tf
import numpy as np
import pandas as pd

class MF(tf.keras.Model):
    def __init__(self, user_no, prob_no, K=4):
        super(MF, self).__init__()
        # 변수
        self.user_no = user_no
        self.prob_no = prob_no
        self.K = K

        # 레이어
        input_user = tf.keras.layers.Input(shape=(1,), dtype='int32')
        input_prob =  tf.keras.layers.Input(shape=(1,), dtype='int32')
        embedding_user = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(user_no, K)(input_user))
        embedding_prob =  tf.keras.layers.Flatten()(tf.keras.layers.Embedding(prob_no, K)(input_prob))
        matmul =  tf.keras.layers.Multiply()([embedding_user, embedding_prob])
        output =  tf.keras.layers.Dense(1)(matmul)

        # 모델
        self.model = tf.keras.Model(inputs=[input_user, input_prob], outputs=output)

    def get_model(self):
        return self.model

    def save_model(self, DIR):
        self.model.save(DIR)
        
    def level_filtering(self, dataframe, userlevel_map, problevel_map, k):
  
        user = dataframe['handle'].to_numpy()
        prob = dataframe['problemId'].to_numpy()
        pred = dataframe['pred'].to_numpy()
        
        
        limit = min(k*10, len(pred))
        idx = np.argpartition(-pred, limit)[:limit]
        
        candidates = dataframe.iloc[idx]
        problevel = candidates['problemId'].apply(lambda x: problevel_map[x]).to_numpy()
        maxlevel = candidates['handle'].apply(lambda x: userlevel_map[x]).to_numpy()
        lam = np.mean(candidates['pred'].to_numpy())/10
        dist = np.abs(problevel-maxlevel)*lam
        candidates['pred']+=dist
          
        #top_idx = np.argpartition(dist, k)[:limit]
        top_idx = np.argpartition(candidates['pred'].to_numpy(), k)[:limit]
        top_k_data = dataframe.iloc[top_idx]
        
        return top_k_data