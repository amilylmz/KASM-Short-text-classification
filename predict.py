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







def val():
    """----word level----"""
    all_word_file = "./word/all_clean.txt"
    train_word_file = "./word/train_clean.txt"
    test_word_file = "./word/test_clean.txt"
    """----self entity level----"""
    all_selfentity_file = "./self_entity/all_good_selfentity_last.txt"
    train_selfentity_file = "./self_entity/train_good_selfentity_last.txt"
    test_selfentity_file = "./self_entity/test_good_selfentity_last.txt"

    """----father entity level----"""
    all_fatherentity_file = "./father_entity/all_good_fatherentity_last.txt"
    train_fatherentity_file = "./father_entity/train_good_fatherentity_last.txt"
    test_fatherentity_file = "./father_entity/test_good_fatherentity_last.txt"

    """---------CNN--interaction----------"""

    all_cnn_file = "./interaction_data/all_label.txt"
    train_cnn_file = "./interaction_data/train_label.txt"
    test_cnn_file = "./interaction_data/test_label.txt"

    pre_label = []
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/Rnn_Attention')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    x_word_train, y_word_train, vocab_word_processor, x_word_test, y_word_test, \
    x_selfentity_train, y_selfentity_train, vocab_selfentity_processor, x_selfentity_test, y_selfentity_test, \
    x_fatherentity_train, y_fatherentity_train, vocab_fatherentity_processor, x_fatherentity_test, y_fatherentity_test, \
    x_cnn_train, y_cnn_train, vocab_cnn_processor, x_cnn_test, y_cnn_test \
        = shuffle_data(all_word_file, train_word_file, test_word_file,
                       all_selfentity_file, train_selfentity_file, test_selfentity_file,
                       all_fatherentity_file, train_fatherentity_file, test_fatherentity_file,
                       all_cnn_file, train_cnn_file, test_cnn_file
                       )

    batch_val = batch_iter(x_word_test, y_word_test,x_selfentity_test, y_selfentity_test,x_fatherentity_test,
                           y_selfentity_test,x_cnn_test,y_cnn_test,
                           batch_size=64)

    for x_batch, y_batch,x_selfentity_test, y_selfentity_test,x_fatherentity_test, \
        y_selfentity_test, x_cnn_test,y_cnn_test in batch_val:

        seq_len = sequence(x_batch)
        selfentity_len = sequence(x_selfentity_test)
        fatherentity_len = sequence(x_fatherentity_test)
        cnninter_len = sequence(x_cnn_test)

        pre_lab = session.run(model.predict, feed_dict = {model.input_word_x1: x_batch,
                                                          model.input_y1: y_batch,

                                                          model.input_selfentity_x1: x_selfentity_test,
                                                          model.input_selfentity_y1:y_selfentity_test,

                                                          model.input_fatherentity_x1: x_fatherentity_test,
                                                          model.input_fatherentity_y1: y_selfentity_test,

                                                          model.input_cnn_x1: x_cnn_test,
                                                          model.input_cnn_y1: y_cnn_test,

                                                          model.word_seq_length: seq_len,
                                                          model.self_entity_seq_length:selfentity_len,
                                                          model.father_entity_seq_length:fatherentity_len,
                                                          model.cnn_inter_seq_length:cnninter_len,
                                                          model.keep_pro: 1.0})
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label


if __name__ == '__main__':

    pm = pm

    """----word level----"""
    all_word_file = "./word/all_clean.txt"
    train_word_file = "./word/train_clean.txt"
    test_word_file = "./word/test_clean.txt"
    """----self entity level----"""
    all_selfentity_file = "./self_entity/all_good_selfentity_last.txt"
    train_selfentity_file = "./self_entity/train_good_selfentity_last.txt"
    test_selfentity_file = "./self_entity/test_good_selfentity_last.txt"

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
    x_cnn_train, y_cnn_train, vocab_cnn_processor, x_cnn_test, y_cnn_test \
        = shuffle_data(all_word_file, train_word_file, test_word_file,
                       all_selfentity_file, train_selfentity_file, test_selfentity_file,
                       all_fatherentity_file, train_fatherentity_file, test_fatherentity_file,
                       all_cnn_file, train_cnn_file, test_cnn_file
                       )

    pm = pm



    pm.word_vocab_size = len(vocab_word_processor.vocabulary_)
    pm.selfentity_vocab_size = len(vocab_selfentity_processor.vocabulary_)
    pm.fatherentity_vocab_size = len(vocab_fatherentity_processor.vocabulary_)
    pm.cnn_vocab_size = len(vocab_cnn_processor.vocabulary_)

    pm.word_pre_training = make_vector_matrix()  # 词典向量  已更换   word's prevector
    pm.selfentity_pre_training = make_selfentity_vector_matrix()  # self-entity prevector
    pm.fatherentity_pre_training = make_fatherentity_vector_matrix()  # father-entity  prevector
    pm.cnninter_pre_training = make_cnninter_vector_matrix()

    pm.word_seq_length = x_word_train.shape[1]
    pm.self_entity_seq_length = x_selfentity_train.shape[1]
    pm.father_entity_seq_length = x_fatherentity_train.shape[1]
    pm.cnn_inter_seq_length = x_cnn_train.shape[1]
    pm.num_classes = y_word_train.shape[1]



    model = KASM()

    pre_label, label=val()

    correct = np.equal(pre_label, np.argmax(label, 1))
    accuracy = np.mean(np.cast['float32'](correct))
    print('accuracy:', accuracy)
    # print("预测前10项：", ' '.join(str(pre_label[:10])))
    # print("正确前10项：", ' '.join(str(np.argmax(label[:10], 1))))

