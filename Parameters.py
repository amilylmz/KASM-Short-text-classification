# -*- coding: utf-8 -*-
class Parameters(object):

    """-------------不同水平嵌入的维度-------"""
    word_embedding_dim = 768     #dimension of word embedding
    self_entity_embedding_dim=100
    father_entity_embedding_dim = 100
    cnn_inter_embedding_dim=10 #你的向量的维度      交互信息的vector


    """---所有预训练向量--------"""
    word_pre_training = None       #use vector_char trained by word2vec
    selfentity_pre_training=None
    fatherentity_pre_training = None
    cnninter_pre_training=None

    num_classes = None     #number of labels


    """---所有水平的长度--------"""
    word_seq_length = None  # max length of sentence
    self_entity_seq_length=None
    father_entity_seq_length=None
    cnn_inter_seq_length=None



    """---所有水平细胞的hidden layer--------"""
    word_hidden_dim =300        #the number of hidden units
    selfentity_hidden_dim=300

    fatherentity_hidden_dim = 600

    """---所有水平细胞的注意力机制的隐含size--------"""
    word_attention_size = 300    #the size of attention layer
    selfentity_attention_size = 300

    fatherentity_attention_size = 600

    keep_prob = 0.5         #droppout
    learning_rate = 1e-4    #learning rate
    lr_decay = 0.9          #learning rate decay
    clip = 5.0              #gradient clipping threshold

    num_epochs = 9 #epochs
    batch_size = 64         #batch_size


    """---------词典长度------------"""
    word_vocab_size=None
    selfentity_vocab_size=None
    fatherentity_vocab_size=None
    cnn_vocab_size=None


    """---------CNN配置---------"""
    filter_sizes=["3","4","5"]
    num_filters=10  #应该和cnn_inter_embedding_dim 相同