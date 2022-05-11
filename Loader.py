import os
import pandas as pd
import numpy as np

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
        self.users_no = len(self.users)
        self.prob_no = len(self.problems)
        return self.users_no, self.prob_no
    
    
    def formatting(self, case):
        # 모델 input 형식에 맞게 바꾸기
        # case: 0 train
        if case == 0:
          train = self.train_formating(self.train)
          return train
        elif case == 1: # valid
          #valid_X = self.test_formating(self.valid_X)
          neg = self.get_negative_sampling(self.valid_y)
          return self.valid_X, (self.valid_y, neg)
        else: # test
          #test_X = self.test_formating(self.test_X)
          neg = self.get_negative_sampling(self.test_y)
          return self.test_X, (self.test_y, neg)
  
    def idx_to_id(self, idx, dataframe):
        return dataframe.iloc[idx, 0]

    # train foramting
    def train_formating(self, dataframe):
        userId, probId, entry = [], [], []
        checked = set([tuple(x) for x in dataframe.values])
        print(len(checked))
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

    def get_negative_sampling(self, dataframe):
        userId, probId, entry = [], [], []
        checked = set([tuple(x) for x in dataframe.values])
        neg_checked = set()
        for up in checked:
            u = up[0]
            # zero: negative sampling
            userId, probId, entry = self.negative_sampling(u, checked, neg_checked, userId, probId, entry)
        df_neg = pd.DataFrame(list(zip(userId, probId)))
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

    def get_idx(self, dataframe, idxlist):
        idx = []
        for id in idxlist:
            for d in dataframe.index[dataframe['handle']==id].tolist():
                idx.append(d)
        return idx