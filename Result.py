import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/content/drive/MyDrive/(22-1)캡스톤/recomm/Recommendation/')
sys.path.append('/content/drive/MyDrive/(22-1)캡스톤/recomm/Recommendation/model/MF')
import Loader
import MF

class Result:
  def __init__(self, batch_size):
    DIR = '/content/drive/MyDrive/(22-1)캡스톤/recomm/data/preprocessed/'
    loader1 = Loader.Loader(DIR, 1, 4) 
    loader2 = Loader.Loader(DIR, 2, 4)
    loader3 = Loader.Loader(DIR, 3, 4)
    loader4 = Loader.Loader(DIR, 4, 4)
    loader5 = Loader.Loader(DIR, 5, 4)
    loader1.load_dataset()
    loader2.load_dataset()
    loader3.load_dataset()
    loader4.load_dataset()
    loader5.load_dataset()
    self.loaders = [loader1, loader2, loader3, loader4, loader5]


    DIR2 = '/content/drive/MyDrive/(22-1)캡스톤/recomm/data/raw_data/'
    problemMeta = pd.read_csv(os.path.join(DIR2, "problemMeta.csv"))
    self.probid2level = {row[0]:row[5] for row in problemMeta.values}

    self.mf = MF.MF(loader1.users_no, loader1.prob_no, loader1.useridx2level, loader1.probidx2level)
    BESTMODEL_DIR = '/content/drive/MyDrive/(22-1)캡스톤/recomm/Recommendation/model/MF/best_model/cluster'
    model1 = tf.keras.models.load_model(BESTMODEL_DIR+'1')
    model2 = tf.keras.models.load_model(BESTMODEL_DIR+'2')
    model3 = tf.keras.models.load_model(BESTMODEL_DIR+'3')
    model4 = tf.keras.models.load_model(BESTMODEL_DIR+'4')
    model5 = tf.keras.models.load_model(BESTMODEL_DIR+'5')
    self.models = [model1, model2, model3, model4, model5]
    #self.models = [model1, model2, model3]

    self.batch_size = batch_size


  def get_output(self, id, problemIds):
    cluster = self.get_cluster(problemIds)
    output = self.goto_model(id, problemIds, self.models[cluster-1], cluster)
    return output

  def get_cluster(self, problemIds):
   maxlevel = max([self.probid2level[prob] for prob in problemIds])
   if maxlevel <= 4 and maxlevel >= 1:
     return 1
   elif maxlevel <= 10:
     return 2
   elif maxlevel <= 13:
     return 3
   elif maxlevel <= 16:
     return 4
   else:
     return 5

  def goto_model(self, id, problemIds, model, cluster):
    usridx = self.get_usr_index(id, cluster)
    probidx = self.get_prb_index(problemIds, cluster)
    neg_probidx = self.get_negative_prob(probidx, cluster)
    
    train_usr = np.array([usridx] * len(probidx)).reshape(-1,1)
    train_prb = np.array(probidx).reshape(-1,1)
    train_entry = np.ones_like(train_usr)

    test_usr = np.array([usridx] * len(neg_probidx)).reshape(-1,1)
    test_prb = np.array(neg_probidx).reshape(-1,1)

    weights = model.get_weights()
    for i in range(0, len(train_usr), self.batch_size):
      idxlist = range(i, min(i+self.batch_size, len(train_usr)-1))
      model.fit([train_usr[idxlist], train_prb[idxlist]], train_entry[idxlist], verbose=0)
          
    for i in range(0, len(test_usr), self.batch_size):
      idxlist = range(i, min(i+self.batch_size, len(test_usr)-1))
      pred = model.predict([test_usr[idxlist], test_prb[idxlist]])
      pred = np.concatenate(pred).reshape(-1,1)
    
    filtered = self.mf.level_filtering(test_usr, test_prb, pred, self.loaders[cluster-1].useridx2level, self.loaders[cluster-1].probidx2level, k=30)
    model.set_weights(weights)

    output = self.get_id(filtered[1], cluster)
    return output

  def get_usr_index(self, id, cluster):
    try:
      usridx = self.loaders[cluster-1].userid2idx[id]
    except:
      usridx = self.loaders[cluster-1].users_no + self.loaders[cluster-1].users_free
    return usridx
  
  def get_prb_index(self, problemIds, cluster):
    prbidx = [self.loaders[cluster-1].probid2idx[prob] for prob in problemIds]
    return prbidx
  
  def get_negative_prob(self, problems, cluster):
    return list(set(range(0, self.loaders[cluster-1].prob_no)) - set(problems))

  def get_id(self,problems, cluster):
    prbidx = [self.loaders[cluster-1].probidx2id[prob] for prob in problems]
    return prbidx