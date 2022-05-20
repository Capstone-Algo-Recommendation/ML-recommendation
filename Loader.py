import os
import pandas as pd
import numpy as np
from ast import literal_eval

class Loader:
    def __init__(self, DIR, data_no, neg_no):
        self.DIR = DIR
        self.neg_no = neg_no
        self.data_no = data_no
  
    def load_dataset(self):
        data_no = str(self.data_no)
        self.DATASET_DIR = self.DIR + 'data' + data_no

        # 데이터 로드
        self.train = pd.read_csv(os.path.join(self.DATASET_DIR, 'd'+data_no+'_train.csv'))
        self.valid_X = pd.read_csv(os.path.join(self.DATASET_DIR , 'd'+data_no+'_validation_X.csv'))
        self.valid_y = pd.read_csv(os.path.join(self.DATASET_DIR , 'd'+data_no+'_validation_y.csv'))
        self.test_X = pd.read_csv(os.path.join(self.DATASET_DIR , 'd'+data_no+'_test_X.csv'))
        self.test_y = pd.read_csv(os.path.join(self.DATASET_DIR , 'd'+data_no+'_test_y.csv'))
        self.users = pd.read_csv(os.path.join(self.DATASET_DIR , 'd'+data_no+'_users.csv'))
        self.problems = pd.read_csv(os.path.join(self.DATASET_DIR , 'd'+data_no+'_problems.csv'))
        
        # 변수
        self.users_no = len(self.users)
        self.prob_no = len(self.problems)
        self.userid2idx = {row[1]:row[0] for row in self.users.values}
        self.useridx2id = {row[0]:row[1] for row in self.users.values}
        self.probid2idx = {row[1]:row[0] for row in self.problems.values}
        self.probidx2id = {row[0]:row[1] for row in self.problems.values}
        self.useridx2level = {row[0]:row[2] for row in self.users.values}
        self.probidx2level = {row[0]:row[2] for row in self.problems.values}

        self.user_free = 1
        return self.users_no, self.prob_no
    
    def formatting(self, case):
        # 모델 input 형식에 맞게 바꾸기
        if case == 0:
          train = self.train_formating(self.train)
          return train
        elif case == 1: # valid
          train = self.train_formating(self.valid_X)
          test, neg = self.test_formating(self.valid_y, self.valid_X)
          return train, (test, neg)
        else: # test
          train = self.train_formating(self.test_X)
          test, neg = self.test_formating(self.test_y, self.test_X)
          return train , (test, neg)
  
    # train foramting
    def train_formating(self, dataframe):
        userId, probId, entry = [], [], []
        checked = set([(self.userid2idx[x[0]], self.probid2idx[x[1]]) for x in dataframe.values])

        neg_checked = set()

        for up in checked:
            u, p = up[0], up[1]
            # nonzero
            userId.append(u)
            probId.append(p)
            entry.append(1)

            # zero: negative sampling
            userId, probId, entry = self.negative_sampling(u, checked, neg_checked, userId, probId, entry)
        return userId, probId, entry
        
    def test_formating(self, y, pos):
      y_handle = y['handle'].apply(lambda x: self.userid2idx[x])
      y_problem = y['problemId'].apply(lambda x: literal_eval(x))
      y_problem = y_problem.apply(lambda x: [self.probid2idx[prob] for prob in x])
      df_new = pd.concat([y_handle, y_problem], axis=1)
      
      pos['handle'] = pos['handle'].apply(lambda x: self.userid2idx[x])
      pos['problemId'] = pos['problemId'].apply(lambda x: self.probid2idx[x])
      df_neg = self.get_negative_sampling(pos)
      return df_new, df_neg

    def get_negative_sampling(self, dataframe):
        
        totalProb = self.problems['problemId'].apply(lambda x: self.probid2idx[x])
        totalProb = set(totalProb.tolist())
        pos = dataframe.groupby(["handle"]).agg({'problemId': lambda x: x.tolist()}).reset_index()
        pos['problemId'] = pos['problemId'].apply(lambda x: list(totalProb-set(x)))
        df_neg = pos
        return df_neg

    # negative sampling
    def negative_sampling(self, u, checked, neg_checked, user, prob, entry):
        for n in range(self.neg_no):
            flag = False
            for t in range(20):
                p = np.random.randint(self.prob_no)
                if (u,p) not in checked and (u,p) not in neg_checked:
                    neg_checked.add((u,p))
                    flag = True
                    break
            if flag:
                user.append(u)
                prob.append(p)
                entry.append(0)
                
        return user, prob, entry