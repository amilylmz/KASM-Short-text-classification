#encoding=utf-8
import os
import tensorflow as tf
from Parameters import Parameters as pm
from data_processing import batch_iter, sequence
from KASM import KASM

from model_pre_vector  import make_vector_matrix
from model_pre_vector  import make_selfentity_vector_matrix
from model_pre_vector import make_fatherentity_vector_matrix
from model_pre_vector import make_cnninter_vector_matrix

import  data_helpers
from tensorflow.contrib import learn
import numpy as np
import datetime
import matplotlib.pyplot as plt

def make_source_data(all_file,train_file,test_file):
    """
    各种level (word, self-entity, father-entity)  的训练集 测试集
    :return:
    """
    print("Loading data...")
    x_all,y_all=data_helpers.load_data_and_labels(all_file)
    x_train,y_train=data_helpers.load_data_and_labels(train_file)
    x_test, y_test = data_helpers.load_data_and_labels(test_file)

    word_level_max_document_length = max([len(x.split(" ")) for x in x_all])  # 最大长度34  经计算 写在下面
    # word_level_max_document_length=34

    vocab_processor = learn.preprocessing.VocabularyProcessor(word_level_max_document_length)
    #
    vocab=vocab_processor.vocabulary_
    #
    # # train  所有的
    x = np.array(list(vocab_processor.fit_transform(x_all)))
    #
    vocab_processor_n = learn.preprocessing.VocabularyProcessor(word_level_max_document_length,vocabulary=vocab)
    # # train
    x_train = np.array(list(vocab_processor_n.fit_transform(x_train)))
    y_train=np.array(y_train)
    # #test
    x_test = np.array(list(vocab_processor_n.fit_transform(x_test)))
    y_test=np.array(y_test)
    return x_train,y_train,vocab_processor,x_test,y_test




def shuffle_data(all_word_file,train_word_file,test_word_file,
                 all_selfentity_file, train_selfentity_file, test_selfentity_file,
                 all_fatherentity_file, train_fatherentity_file, test_fatherentity_file,
                 all_cnn_file, train_cnn_file, test_cnn_file):
    """
    对所有水平的数据进行洗牌 word, self entity ,father entity
    :return:
    """
    x_train_word, y_train_word, vocab_word_processor,x_test_word, y_test_word=make_source_data(all_word_file,train_word_file,test_word_file)

    x_train_selfentity, y_train_selfentity, vocab_selfentity_processor,x_test_selfentity, y_test_selfentity = make_source_data(
                                                                                                            all_selfentity_file,
                                                                                                            train_selfentity_file,
                                                                                                            test_selfentity_file)

    x_train_fatherentity, y_train_fatherentity, vocab_fatherentity_processor, x_test_fatherentity, y_test_fatherentity = make_source_data(
                                                                                                                        all_fatherentity_file,
                                                                                                                        train_fatherentity_file,
                                                                                                                        test_fatherentity_file)

    x_cnn_train, y_cnn_train, vocab_cnn_processor, x_cnn_test, y_cnn_test=make_source_data(
                                                                                            all_cnn_file,
                                                                                            train_cnn_file,
                                                                                            test_cnn_file)





    shuffle_indices = np.random.permutation(np.arange(len(y_train_word)))
    #word
    x_word_shuffled_train = np.array(x_train_word)[shuffle_indices]
    y_word_shuffled_train = np.array(y_train_word)[shuffle_indices]

    #self-entity
    x_selfentity_shuffled_train = np.array(x_train_selfentity)[shuffle_indices]
    y_selfentity_shuffled_train = np.array(y_train_selfentity)[shuffle_indices]
    # print(len(x_selfentity_shuffled_train))

    # father-entity
    x_fatherentity_shuffled_train = np.array(x_train_fatherentity)[shuffle_indices]
    y_fatherentity_shuffled_train = np.array(y_train_fatherentity)[shuffle_indices]
    # print(len(x_selfentity_shuffled_train))

    #cnn-interaction
    x_cnnshuffled_train = np.array(x_cnn_train)[shuffle_indices]
    y_cnnshuffled_train = np.array(y_cnn_train)[shuffle_indices]


    return x_word_shuffled_train,y_word_shuffled_train, vocab_word_processor,x_test_word,y_test_word,\
    x_selfentity_shuffled_train,y_selfentity_shuffled_train,vocab_selfentity_processor,x_test_selfentity, y_test_selfentity,\
    x_fatherentity_shuffled_train, y_fatherentity_shuffled_train, vocab_fatherentity_processor, x_test_fatherentity, y_test_fatherentity, \
           x_cnnshuffled_train, y_cnnshuffled_train, vocab_cnn_processor, x_cnn_test, y_cnn_test




def train(x_word_train, y_word_train,
          x_word_test, y_word_test,

          x_selfentity_train, y_selfentity_train,
          x_selfentity_test, y_selfentity_test,

          x_fatherentity_train, y_fatherentity_train,
          x_fatherentity_test, y_fatherentity_test,

          x_cnn_train, y_cnn_train,
          x_cnn_test, y_cnn_test,


        ):
    tensorboard_dir = './tensorboard/Rnn_Attention'
    save_dir = './checkpoints/Rnn_Attention'
    if not os.path.exists(tensorboard_dir):  #只是创建目录而已
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')


    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)


    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    T_loss = []
    L_loss = []

    for epoch in range(pm.num_epochs):
    # # #     print('Epoch:', epoch+1)
        num_batchs = int((len(x_word_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_word_train, y_word_train,
                                 x_selfentity_train,y_selfentity_train,
                                 x_fatherentity_train, y_fatherentity_train,
                                 x_cnn_train, y_cnn_train,
                                 batch_size=pm.batch_size)

        for x1,y1,x2,y2,x3,y3,x4,y4 in batch_train: #坑就是在这里  前面4个值，不能和上面batch_iter的值重了


            word_seq_len = sequence(x1)
            selfentity_seq_len=sequence(x2)
            fatherentity_seq_len = sequence(x3)
            cnn_inter_seq_len = sequence(x4)

            feed_dict = model.feed_data(x1, y1, word_seq_len,
                                        x2,y2,selfentity_seq_len,
                                        x3, y3,fatherentity_seq_len,
                                        x4, y4, cnn_inter_seq_len,
                                        pm.keep_prob)

            _, global_step, _summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step, merged_summary,
                                                                                model.loss, model.accuracy],feed_dict=feed_dict)

            # print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy)
            if global_step % 50== 0:
                test_loss, test_accuracy = model.evaluate(session,
                                                          x_word_test, y_word_test,
                                                          x_selfentity_test,y_selfentity_test,
                                                          x_fatherentity_test, y_fatherentity_test,
                                                          x_cnn_test, y_cnn_test)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      'test_loss:', test_loss, 'test_accuracy:', test_accuracy)
                T_loss.append(train_loss)
                L_loss.append(test_loss)

            if global_step % num_batchs == 0:
                print('Saving Model...')
                saver.save(session, save_path, global_step=global_step)
    #
    pm.learning_rate *= pm.lr_decay

    x = range(len(T_loss))
    plt.plot(x, T_loss)  # blue is the train loss
    plt.plot(x, L_loss, "r--")  # red is the test
    plt.show()



if __name__ == '__main__':
    # preprocess()
    begin=datetime.datetime.now()
    """----word level----"""
    all_word_file="./word/all_clean.txt"
    train_word_file="./word/train_clean.txt"
    test_word_file="./word/test_clean.txt"
    """----self entity level----"""
    all_selfentity_file="./self_entity/all_good_selfentity_last.txt"
    train_selfentity_file="./self_entity/train_good_selfentity_last.txt"
    test_selfentity_file="./self_entity/test_good_selfentity_last.txt"

    """----father entity level----"""
    all_fatherentity_file = "./father_entity/all_good_fatherentity_last.txt"
    train_fatherentity_file = "./father_entity/train_good_fatherentity_last.txt"
    test_fatherentity_file = "./father_entity/test_good_fatherentity_last.txt"

    """---------CNN--interaction----------"""

    all_cnn_file = "./interaction_data/all_label.txt"
    train_cnn_file = "./interaction_data/train_label.txt"
    test_cnn_file = "./interaction_data/test_label.txt"



    x_word_train, y_word_train, vocab_word_processor, x_word_test, y_word_test, \
    x_selfentity_train, y_selfentity_train, vocab_selfentity_processor, x_selfentity_test, y_selfentity_test, \
    x_fatherentity_train, y_fatherentity_train, vocab_fatherentity_processor, x_fatherentity_test, y_fatherentity_test, \
    x_cnn_train, y_cnn_train, vocab_cnn_processor, x_cnn_test, y_cnn_test\
                                                        =shuffle_data(  all_word_file,train_word_file,test_word_file,
                                                                        all_selfentity_file,train_selfentity_file,test_selfentity_file,
                                                                        all_fatherentity_file, train_fatherentity_file,test_fatherentity_file,
                                                                        all_cnn_file, train_cnn_file,test_cnn_file
                                                                        )

    pm = pm


    #
    pm.word_vocab_size =len(vocab_word_processor.vocabulary_)
    pm.selfentity_vocab_size=len(vocab_selfentity_processor.vocabulary_)
    pm.fatherentity_vocab_size = len(vocab_fatherentity_processor.vocabulary_)
    pm.cnn_vocab_size = len(vocab_cnn_processor.vocabulary_)


    pm.word_pre_training =make_vector_matrix() #词典向量  已更换   word's prevector
    pm.selfentity_pre_training=make_selfentity_vector_matrix()   #self-entity prevector
    pm.fatherentity_pre_training = make_fatherentity_vector_matrix()  # father-entity  prevector
    pm.cnninter_pre_training=make_cnninter_vector_matrix()


    pm.word_seq_length=x_word_train.shape[1]
    pm.self_entity_seq_length=x_selfentity_train.shape[1]
    pm.father_entity_seq_length = x_fatherentity_train.shape[1]
    pm.cnn_inter_seq_length = x_cnn_train.shape[1]

    pm.num_classes = y_word_train.shape[1]

    print("哥们 开始加载模型了奥！！")

    model = Rnn_Attention()

    print("哥们 模型加载结束 开始进数了。操作起来！")
    train(x_word_train, y_word_train, x_word_test, y_word_test,
    x_selfentity_train, y_selfentity_train, x_selfentity_test, y_selfentity_test,
    x_fatherentity_train, y_fatherentity_train, x_fatherentity_test, y_fatherentity_test,
    x_cnn_train, y_cnn_train, x_cnn_test, y_cnn_test)

    print("卧槽， 累死我了，终于跑完了。喝杯奶。relax!!")


    end = datetime.datetime.now()
    print("CPU为您花费的计算时间为:", (end - begin).seconds, "秒")
