#encoding=utf-8
import tensorflow as tf
import numpy as np
def random_vector():
    vocab_size=2
    embedding_size=3
    a=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        s=sess.run(a)
        print(s)

def make_vector_matrix():
    with open("./word/MR_word_BERT_vec.txt","r",encoding="utf-8") as fr:
        figure_matrix=[]
        for line in fr.readlines():
            line=line.strip().split("\t")
            figures=[float(x) for x in line[1].split(" ")]
            figure_matrix.append(figures)

        dic_vector_matrix=np.array(figure_matrix)
        return dic_vector_matrix


def make_selfentity_vector_matrix():
    with open("./self_entity/good_selfentity_dic_vector.txt", "r", encoding="utf-8") as fr:
        figure_matrix = []
        for line in fr.readlines():
            line = line.strip().split("\t")
            figures = [float(x) for x in line[1].split(" ")]


            figure_matrix.append((figures))

        dic_concept_vector_matrix = (np.array(figure_matrix))
        return dic_concept_vector_matrix


def make_fatherentity_vector_matrix():
    with open("./father_entity/good_fatherentity_dic_vector.txt", "r", encoding="utf-8") as fr:
        figure_matrix = []
        for line in fr.readlines():
            line = line.strip().split("\t")
            figures = [float(x) for x in line[1].split(" ")]


            figure_matrix.append((figures))

        dic_concept_vector_matrix = (np.array(figure_matrix))
        return dic_concept_vector_matrix


def make_cnninter_vector_matrix():
    with open("./interaction_data/MR_labelword_vector.txt","r",encoding="utf-8") as fr:
        figure_matrix=[]
        for line in fr.readlines():
            line=line.strip().split("\t")
            figures=[float(x) for x in line[1].split(" ")]
            figure_matrix.append(figures)

        dic_vector_matrix=np.array(figure_matrix)
        return dic_vector_matrix






if __name__=="__main__":
    # print(type(make_vector_matrix()))
    # random_vector()
    # a=(make_vector_matrix())
    # print(a)

    print("您经过了初始化向量阶段")