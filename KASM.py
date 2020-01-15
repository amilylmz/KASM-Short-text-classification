import tensorflow as tf
from Parameters import Parameters as pm
from data_processing import *
from tensorflow.contrib import layers
class KASM(object):

    def __init__(self):
        """---输入-----------"""
        # word level
        self.input_word_x1 = tf.placeholder(tf.int32, shape=[None, pm.word_seq_length], name='input_word_x1')
        self.input_y1 = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y1')
        # 2 self entity level
        self.input_selfentity_x1 = tf.placeholder(tf.int32, shape=[None, pm.self_entity_seq_length], name='input_selfentity_x1')
        self.input_selfentity_y1 = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_selfentity_y1')
        # 3 father entity level
        self.input_fatherentity_x1 = tf.placeholder(tf.int32, shape=[None, pm.father_entity_seq_length],name='input_fatherentity_x1')
        self.input_fatherentity_y1 = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_fatherentity_y1')

        # 4 father entity level
        self.input_cnn_x1 = tf.placeholder(tf.int32, shape=[None, pm.cnn_inter_seq_length],name='input_cnn_x1')
        self.input_cnn_y1 = tf.placeholder(tf.float32, shape=[None, pm.num_classes],name='input_cnn_y1')


        """------句子长度-----------"""
        self.word_seq_length = tf.placeholder(tf.int32, shape=[None], name='word_seq_length')
        self.self_entity_seq_length = tf.placeholder(tf.int32, shape=[None], name='self_entity_seq_length')
        self.father_entity_seq_length = tf.placeholder(tf.int32, shape=[None], name='father_entity_seq_length')
        self.cnn_inter_seq_length = tf.placeholder(tf.int32, shape=[None], name='cnn_inter_seq_length')


        # 2 type
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.Rnn_attention()




    def Rnn_attention(self):

        """----------定义不同水平的细胞-----------"""
        #word 水平的细胞
        with tf.variable_scope('word_Cell'): #word 细胞
            word_cell_fw = tf.contrib.rnn.GRUCell(pm.word_hidden_dim)
            word_Cell_fw = tf.contrib.rnn.DropoutWrapper(word_cell_fw, self.keep_pro)
            word_cell_bw = tf.contrib.rnn.GRUCell(pm.word_hidden_dim)
            word_Cell_bw = tf.contrib.rnn.DropoutWrapper(word_cell_bw, self.keep_pro)

        # self-entity 水平的细胞
        with tf.variable_scope('self_entity_Cell'):  #(self-entity细胞)声明不同的cell 用 tf.variable_scope
            selfentity_cell_fw = tf.contrib.rnn.GRUCell(pm.selfentity_hidden_dim)
            selfentity_Cell_fw = tf.contrib.rnn.DropoutWrapper(selfentity_cell_fw, self.keep_pro)
            selfentity_cell_bw = tf.contrib.rnn.GRUCell(pm.selfentity_hidden_dim)
            selfentity_Cell_bw = tf.contrib.rnn.DropoutWrapper(selfentity_cell_bw, self.keep_pro)
        # father-entity 水平的细胞
            # self-entity 水平的细胞
        with tf.variable_scope('father_entity_Cell'):  # (self-entity细胞)声明不同的cell 用 tf.variable_scope
            fatherentity_cell_fw = tf.contrib.rnn.GRUCell(pm.fatherentity_hidden_dim)
            fatherentity_Cell_fw = tf.contrib.rnn.DropoutWrapper(fatherentity_cell_fw, self.keep_pro)
            fatherentity_cell_bw = tf.contrib.rnn.GRUCell(pm.fatherentity_hidden_dim)
            fatherentity_Cell_bw = tf.contrib.rnn.DropoutWrapper(fatherentity_cell_bw, self.keep_pro)



        """----------输入嵌入矩阵-----------"""
        #word 水平
        with tf.variable_scope('word_embedding'):
            self.embedding = tf.get_variable('word_embedding', shape=[pm.word_vocab_size, pm.word_embedding_dim],
                                             initializer=tf.constant_initializer(pm.word_pre_training))

            self.embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_word_x1)
            # print(self.embedding)


        #self-entity 水平
        with tf.variable_scope('self_entity_embedding'):
            self.selfentity_embedding= tf.get_variable('selfentity_embedding', shape=[pm.selfentity_vocab_size, pm.self_entity_embedding_dim],
                                             initializer=tf.constant_initializer(pm.selfentity_pre_training))
            self.selfentity_embedding_input = tf.nn.embedding_lookup(self.selfentity_embedding, self.input_selfentity_x1)

        # father-entity 水平
        with tf.variable_scope('father_entity_embedding'):
            self.fatherentity_embedding = tf.get_variable('fatherentity_embedding', shape=[pm.fatherentity_vocab_size,
                                                                                    pm.father_entity_embedding_dim],
                                                                                    initializer=tf.constant_initializer(
                                                                                    pm.fatherentity_pre_training))

            self.fatherentity_embedding_input = tf.nn.embedding_lookup(self.fatherentity_embedding,
                                                                     self.input_fatherentity_x1)

        #cnn-interaction  水平

        with tf.variable_scope('cnninteration_embedding'):
            self.cnninter_embedding = tf.get_variable('cnninter_embedding', shape=[pm.cnn_vocab_size,
                                                                                    pm.cnn_inter_embedding_dim],
                                                                                    initializer=tf.constant_initializer(
                                                                                    pm.cnninter_pre_training))

            self.fatherentity_embedding_input = tf.nn.embedding_lookup(self.cnninter_embedding,
                                                                     self.input_cnn_x1)

            self.embedded_chars_expanded = tf.expand_dims(self.fatherentity_embedding_input, -1)
            # print("1",self.fatherentity_embedding_input )
            # print("2",self.embedded_chars_expanded )
        """-------CNN 架构---------"""
        pooled_outputs = []
        for i, filter_size in enumerate(pm.filter_sizes):
            filter_size = int(filter_size)
            #    Convolution Layer
            filter_shape = [filter_size, pm.cnn_inter_embedding_dim, 1, pm.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="cnn_W")

            b = tf.Variable(tf.constant(0.1, shape=[pm.num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            #
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, pm.cnn_inter_seq_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = pm.num_filters * len(pm.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        #

        """----------细胞的连接，双向连接-----------"""
        with tf.variable_scope('biRNN'):
            wordoutput, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_Cell_fw,
                                                            cell_bw=word_Cell_bw,
                                                            inputs=self.embedding_input,
                                                            sequence_length=self.word_seq_length,
                                                            dtype=tf.float32)
            wordoutput = tf.concat(wordoutput, 2) #[batch_size, seq_length, 2*hidden_dim] shape=(?, 34, 100),

        with tf.variable_scope('selfentitybiRNN'):
            selfentity_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=selfentity_Cell_fw,
                                                                   cell_bw=selfentity_Cell_bw,
                                                                   inputs=self.selfentity_embedding_input,
                                                                   sequence_length=self.self_entity_seq_length,
                                                                   dtype=tf.float32)
            selfentity_output = tf.concat(selfentity_output, 2) #[batch_size, seq_length, 2*hidden_dim]


        with tf.variable_scope('fatherentitybiRNN'):
            fatherentity_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fatherentity_Cell_fw,
                                                                   cell_bw=fatherentity_Cell_bw,
                                                                   inputs=self.fatherentity_embedding_input,
                                                                   sequence_length=self.father_entity_seq_length,
                                                                   dtype=tf.float32)
            fatherentity_output = tf.concat(fatherentity_output, 2)  # [batch_size, seq_length, 2*hidden_dim]



        with tf.variable_scope('word_attention'):
            u_list = []
            seq_size = wordoutput.shape[1].value
            hidden_size = wordoutput.shape[2].value #[2*hidden_dim]
            # print("hidden_size:",hidden_size)
            word_attention_w = tf.Variable(tf.truncated_normal([hidden_size, pm.word_attention_size], stddev=0.1), name='attention_w')
            word_attention_u = tf.Variable(tf.truncated_normal([pm.word_attention_size, 1], stddev=0.1), name='attention_u')
            word_attention_b = tf.Variable(tf.constant(0.1, shape=[pm.word_attention_size]), name='attention_b')
            word_attention_V=tf.Variable(tf.constant(0.1, shape=[pm.word_attention_size,pm.word_attention_size]), name='attention_b')

            for t in range(seq_size):
                #u_t:[1,attention]

                # print(wordoutput)#shape=(?, 34, 100)
                # print(wordoutput[:, t, :])#shape=(?, 100)
                u_t = tf.tanh(tf.matmul(wordoutput[:, t, :], word_attention_w) + tf.reshape(word_attention_b, [1, -1]))#tf.reshape(tensor,[-1,1])将张量变为一维列向量tf.reshape(tensor,[1,-1])将张量变为一维行向量

                u_t=tf.matmul(u_t,tf.transpose(word_attention_V))
                u = tf.matmul(u_t, word_attention_u)
                u_list.append(u)
            logit = tf.concat(u_list, axis=1)

            #u[seq_size:attention_z]
            weights = tf.nn.softmax(logit, name='attention_weights')
            #weight:[seq_size:1]
            # print("weight",weights)#shape=(?, 34)
            # print((tf.reshape(weights, [-1, seq_size, 1]), 1))#shape=(?, 34, 1)
            word_out_final = tf.reduce_sum(wordoutput * tf.reshape(weights, [-1, seq_size, 1]), 1)

        #
        #

        with tf.variable_scope('selfentity_attention'):
            selfentity_u_list = []
            seq_selfentity_size = selfentity_output.shape[1].value
            hidden_selfentity_size = selfentity_output.shape[2].value  # [2*hidden_dim]
            selfentity_w = tf.Variable(tf.truncated_normal([hidden_selfentity_size, pm.selfentity_attention_size], stddev=0.1),name='selfentity_w')
            selfentity_u = tf.Variable(tf.truncated_normal([pm.selfentity_attention_size, 1], stddev=0.1), name='selfentity_u')
            selfentity_b = tf.Variable(tf.constant(0.1, shape=[pm.selfentity_attention_size]), name='selfentity_b')

            selfentity_U=tf.Variable(tf.truncated_normal([1, 1], stddev=0.1), name='selfentity_u')

            for t in range(seq_selfentity_size):
                # u_t:[1,attention]
                u_t = tf.tanh(tf.matmul(selfentity_output[:, t, :], selfentity_w) + tf.reshape(selfentity_b, [1, -1]))
                u = tf.matmul(u_t, selfentity_u)
                u=tf.matmul(u,tf.transpose(selfentity_U))

                selfentity_u_list.append(u)
            selfentity_logit = tf.concat(selfentity_u_list, axis=1)
            # u[seq_size:attention_z]
            selfentity_weights = tf.nn.softmax(selfentity_logit, name='self_attention_weights')
            # weight:[seq_size:1]
            selfentity_out_final = tf.reduce_sum(selfentity_output * tf.reshape(selfentity_weights, [-1, seq_selfentity_size, 1]), 1)

        #
        #等待配置
        with tf.variable_scope('fatherentity_attention'):
            fatherentity_u_list = []
            seq_fatherentity_size = fatherentity_output.shape[1].value
            hidden_fatherentity_size = fatherentity_output.shape[2].value  # [2*hidden_dim]


            selffather_w = tf.Variable(tf.truncated_normal([hidden_fatherentity_size, pm.fatherentity_attention_size], stddev=0.1),name='fatherentity_w')
            selffather_u = tf.Variable(tf.truncated_normal([pm.fatherentity_attention_size, 1], stddev=0.1), name='fatherentity_u')
            selffather_b = tf.Variable(tf.constant(0.1,shape=None), name='fatherentity_b')
            # print(selffather_b)
            for t in range(seq_fatherentity_size):
                # u_t:[1,attention]
                u_t = tf.tanh(tf.matmul(fatherentity_output[:, t, :], selffather_w) )
                u = tf.matmul(u_t, selffather_u)+selffather_b
                fatherentity_u_list.append(u)
            # print(fatherentity_u_list)
            fatherentity_logit = tf.concat(fatherentity_u_list, axis=1)
            # u[seq_size:attention_z]
            fatherentity_weights = tf.nn.softmax(fatherentity_logit, name='father_attention_weights')
            # weight:[seq_size:1]
            fatherentity_out_final = tf.reduce_sum(fatherentity_output * tf.reshape(fatherentity_weights, [-1, seq_fatherentity_size, 1]), 1)
            # print(fatherentity_out_final)



        """-------张量拼接---------"""
        end_final=tf.concat([word_out_final,selfentity_out_final,fatherentity_out_final,self.h_pool_flat],1)
        # print("end_final",end_final)

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(end_final, keep_prob=self.keep_pro)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([2*(pm.word_hidden_dim+pm.selfentity_hidden_dim+pm.fatherentity_hidden_dim)+pm.num_filters * len(pm.filter_sizes), pm.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.zeros([pm.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict') #shape=(?,),


        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y1)
            self.loss = tf.reduce_mean(cross_entropy)+layers.l1_regularizer(0.3)(w)+layers.l2_regularizer(0.3)(w)
    #
        with tf.name_scope('optimizer'):
            # 退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            # learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
            # decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            # no.2
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度倿变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯庿(clip/l2_g),得到新梯庿
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1
    #
        with tf.name_scope('accuracy'):
            correct = tf.equal(self.predict, tf.argmax(self.input_y1, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
            # print(self.accuracy)





    def feed_data(self, word_x_batch, word_y_batch, word_seq_length,
                  selfentity_x_batch,selfentity_y_batch,selfentity_length,
                  fatherentity_x_batch, fatherentity_y_batch, fatherentity_length,
                  cnn_x_batch, cnn_y_batch, cnn_length,

                  keep_pro):
        feed_dict = {self.input_word_x1: word_x_batch,
                    self.input_y1: word_y_batch,


                    self.input_selfentity_x1: selfentity_x_batch,
                    self.input_selfentity_y1:selfentity_y_batch,



                    self.input_fatherentity_x1: fatherentity_x_batch,
                    self.input_fatherentity_y1: fatherentity_y_batch,

                    self.input_cnn_x1: cnn_x_batch,
                    self.input_cnn_y1: cnn_y_batch,



                    self.word_seq_length: word_seq_length,
                    self.self_entity_seq_length:selfentity_length,
                    self.father_entity_seq_length:fatherentity_length,
                    self.cnn_inter_seq_length:cnn_length,
                    self.keep_pro: keep_pro}

        return feed_dict





    def evaluate(self, sess, x1, y1,x2,y2,z1,z2,s1,s2,):
        batch_test = batch_iter(x1, y1,x2,y2,z1,z2,s1,s2,batch_size=64)
        for x_batch, y_batch,x_selfentity_train,y_selfentity_train,x_fatherentity_train,y_fatherentity_train,x_cnn,y_cnn in batch_test:


            seq_len = sequence(x_batch)
            selfentity_len = sequence(x_selfentity_train)
            fatherentity_len = sequence(x_fatherentity_train)
            cnninter_len = sequence(x_cnn)

            feed_dict = self.feed_data(x_batch, y_batch, seq_len,
                                       x_selfentity_train,y_selfentity_train,selfentity_len,
                                       x_fatherentity_train,y_fatherentity_train, fatherentity_len,
                                       x_cnn, y_cnn,cnninter_len,1.0)

            test_loss, test_accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)


        return test_loss, test_accuracy


    #
    # #




