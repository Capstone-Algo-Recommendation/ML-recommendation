import tensorflow as tf

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