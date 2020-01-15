#encoding:utf-8

import numpy as np




def batch_iter(x1,y1,
               x2,y2,
               x3,y3,
               x4,y4,
               batch_size = 64):



    data_len = len(x1)
    num_batch = int((data_len - 1)/batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    '''
    np.arange(4) = [0,1,2,3]
    np.random.permutation([1, 4, 9, 12, 15]) = [15,  1,  9,  4, 12]
    
    '''

    x1_shuff = x1[indices]
    y1_shuff = y1[indices]


    x2_shuff=  x2[indices]
    y2_shuff = y2[indices]


    x3_shuff = x3[indices]
    y3_shuff = y3[indices]

    x4_shuff = x4[indices]
    y4_shuff = y4[indices]


    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)

        #
        x1_end=x1_shuff[start_id:end_id]
        y1_end=y1_shuff[start_id:end_id]


        x2_end=x2_shuff[start_id:end_id]
        y2_end=y2_shuff[start_id:end_id]


        x3_end = x3_shuff[start_id:end_id]
        y3_end = y3_shuff[start_id:end_id]

        x4_end = x4_shuff[start_id:end_id]
        y4_end = y4_shuff[start_id:end_id]



        yield x1_end,y1_end,x2_end,y2_end,x3_end,y3_end,x4_end,y4_end








def sequence(x_batch):
    seq_len = []
    for line in x_batch:
        length = np.sum(np.sign(line)) #sign 输入：[-0.2, -1.1, 0, 2.3]   输出：[-1, -1,0,1]
        seq_len.append(length)

    return seq_len


