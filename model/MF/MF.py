import tensorflow as tf
import numpy as np
import pandas as pd

class MF:
    def __init__(self, user_no, prob_no, useridx2level, probidx2level, K=4):
        # 변수
        self.user_no = user_no+5
        self.prob_no = prob_no
        self.useridx2level = useridx2level
        self.probidx2level = probidx2level
        self.K = K

        # 레이어
        input_user = tf.keras.layers.Input(shape=(1,), dtype='int32') # 사용자 index 
        input_prob =  tf.keras.layers.Input(shape=(1,), dtype='int32') # 문제 index 
        embedding_user = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(user_no, K)(input_user)) # [사용자 index, 잠재요인]
        embedding_prob =  tf.keras.layers.Flatten()(tf.keras.layers.Embedding(prob_no, K)(input_prob)) # [문제 index, 잠재요인]
        matmul =  tf.keras.layers.Multiply()([embedding_user, embedding_prob]) # new [사용자 index, 문제 index]
        output =  tf.keras.layers.Dense(1)(matmul)

        # 모델
        self.model = tf.keras.Model(inputs=[input_user, input_prob], outputs=output)

    def train(self, model, train_usr, train_prb, train_entry, valid_tr_usr, valid_tr_prb, valid_tr_entry, valid_te_X, valid_te_y, cluster, epochs=100, batch_size=1024, k=30):
      train_N = len(train_usr)
      valid_tr_N = len(valid_tr_usr)
      valid_te_N = len(valid_te_X) 
      rc_vad = [] 
      hr_vad = []
      best_eval = -1
      best_epoch = -1
      BESTMODEL_DIR = '/content/drive/MyDrive/(22-1)캡스톤/recomm/Recommendation/model/MF/best_model/cluster' + str(cluster)

      for epoch in range(0, epochs):
        ct, tr_loss, vad_loss = 0, 0, 0
        for i in range(0, train_N, batch_size):
          idxlist = range(i, min(i+batch_size, train_N-1))
          hist = self.model.fit([train_usr[idxlist], train_prb[idxlist]], train_entry[idxlist], verbose=0)
          ct+=1
          tr_loss += hist.history['loss'][-1]
        
        print("epoch[", epoch, "] train loss: ", tr_loss / ct)
        weights = self.model.get_weights()

        ct = 0
        for i in range(0, valid_tr_N, batch_size):
          idxlist = range(i, min(i+batch_size, valid_tr_N-1))
          hist = self.model.fit([valid_tr_usr[idxlist], valid_tr_prb[idxlist]], valid_tr_entry[idxlist],verbose=0)
          ct +=1
          vad_loss += hist.history['loss'][-1]
        
        print("epoch[", epoch, "] validation loss: ", vad_loss / ct)
        
        hit_rate = 0
        recall = []
        for i in range(0, valid_te_N):
          usr = valid_te_X.iloc[i,0]
          prbs = np.array(valid_te_X.iloc[i,1])
          usrs = np.array([usr]*len(prbs))
          valid_te_usr = usrs.reshape(-1,1)
          valid_te_prb = prbs.reshape(-1,1)

          pred = self.model.predict([valid_te_usr, valid_te_prb])
          pred = np.concatenate(pred).reshape(-1,1)
          filtered = self.level_filtering(valid_te_usr, valid_te_prb, pred, self.useridx2level, self.probidx2level, k)

          # valid_y
          heldout = np.array(valid_te_y.iloc[i,1])
          recall.append(Metric.recall_at_k(filtered[1], heldout, k))
          hit_rate += Metric.hit_rate_at_k(filtered[1], heldout, k)
          
        recall_ = np.mean(recall)
        hit_rate_ = hit_rate/valid_te_N
        
        self.model.set_weights(weights)
        
        if hit_rate_ > 0.30:
          break

        rc_vad.append(recall_)
        print("epoch[", epoch, "] recall: ", recall_)
        hr_vad.append(hit_rate_)
        print("epoch[", epoch, "] hit rate: ", hit_rate_)

        if hit_rate_ > best_eval:
          best_eval = hit_rate_
          best_epoch = epoch
          self.model.save(BESTMODEL_DIR)

      return rc_vad, hr_vad, best_eval, best_epoch
      

    def test(self, test_tr_usr, test_tr_prb, test_tr_entry, test_te_X, test_te_y, cluster, batch_size=1024, k=30):
      BESTMODEL_DIR = '/content/drive/MyDrive/(22-1)캡스톤/recomm/Recommendation/model/MF/best_model/cluster' + str(cluster)
      best_model = self.load_model(BESTMODEL_DIR)
      test_tr_N = len(test_tr_usr)
      test_te_N = len(test_te_X)
      test_hit_rate = 0
      test_recall = []
      
      for i in range(0, test_tr_N, batch_size):
          idxlist = range(i, min(i+batch_size, test_tr_N-1))
          self.model.fit([test_tr_usr[idxlist], test_tr_prb[idxlist]], test_tr_entry[idxlist], verbose=0)

      for i in range(0, test_te_N):
        usr = test_te_X.iloc[i,0]
        prbs = np.array(test_te_X.iloc[i,1])
        usrs = np.array([usr]*len(prbs))
        test_te_usr = usrs.reshape(-1,1)
        test_te_prb = prbs.reshape(-1,1)

        pred = self.model.predict([test_te_usr, test_te_prb])
        pred = np.concatenate(pred).reshape(-1,1)
        filtered = self.level_filtering(test_te_usr, test_te_prb, pred, self.useridx2level, self.probidx2level, k)

        heldout = np.array(test_te_y.iloc[i,1])
        test_recall.append(Metric.recall_at_k(filtered[1], heldout, k))
        test_hit_rate += Metric.hit_rate_at_k(filtered[1], heldout, k)

      recall = np.mean(test_recall)
      test_hit_rate = test_hit_rate/test_te_N

      return recall, test_hit_rate

    def get_model(self):
      return self.model

    def load_model(self, DIR):
      return tf.keras.models.load_model(DIR)

    def level_filtering(self, usr, prb, pred, userlevel_map, problevel_map, k):
      limit = min(k*10, len(pred)-1)
      usr = np.squeeze(usr)
      prb = np.squeeze(prb)
      pred = np.squeeze(pred)
      idx = np.argpartition(-pred, limit)[:limit]

      cdUser = usr[idx]
      cdProb = prb[idx]
      cdPred = pred[idx]

      problevel = np.array([problevel_map[p] for p in cdProb])
      maxlevel = np.array([userlevel_map[u] for u in cdUser]).astype('int64')
      lam = np.mean(cdPred)/100

      dist = np.abs(np.subtract(problevel,maxlevel))*lam
      cdPred = np.subtract(cdPred, dist)
              
      top_idx = np.argsort(-cdPred)[:k]
      top_k_data = (cdUser[top_idx], cdProb[top_idx], cdPred[top_idx])
      
      return top_k_data