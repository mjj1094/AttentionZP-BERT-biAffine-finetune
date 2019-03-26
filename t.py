# -*- coding:UTF-8 -*-

# def generate_vec(data_path):
#     zp_candi_target = []
#     zp_vec_index = 0
#     candi_vec_index = 0
#
#     zp_sent_bert = []
#     zp_np_sent_bert = []
#     zp_np_sent_bert_mask = []
#     zp_sent_mask_bert = []
#     zp_orig_to_tok_bert = []
#     zp_np_orig_to_tok_bert = []
#
#     # zp_prefixs = []
#     # zp_prefixs_mask = []
#     # zp_postfixs = []
#     # zp_postfixs_mask = []
#     # candi_vecs = []
#     # candi_vecs_mask = []
#     ifl_vecs = []
#
#     read_f = open(data_path + "zp_info", "rb")
#     zp_info_test = pickle.load(read_f)
#     read_f.close()
#
#     vectorized_sentences = numpy.load(data_path + "sen.npy")
#     vectorized_sentences_bert = numpy.load(data_path + "sen_bert.npy")
#     vectorized_sentences_bert_nomark = numpy.load(data_path + "sen_bert_nomark.npy")
#     vectorized_sentences_mask_bert = numpy.load(data_path + "sen_mask_bert.npy")
#     orig_to_tok_bert = numpy.load(data_path + "orig_to_tok_bert.npy")
#     for zp, candi_info in zp_info_test:
#         index_in_file, sentence_index, zp_index, ana = zp
#         if ana == 1:
#
#             i = max(0, index_in_file - 2)
#             sequence_bert_nomark = vectorized_sentences_bert_nomark[i]
#             max_index_bert = len(sequence_bert_nomark)
#             sequence_bert_nomark = sequence_bert_nomark[:min(args.max_sent_len, max_index_bert)]
#             sequence_bert_mask = (
#                         len(sequence_bert_nomark) * [1] + (args.max_sent_len - len(sequence_bert_nomark)) * [0])
#             sequence_bert_nomark = (sequence_bert_nomark + (args.max_sent_len - len(sequence_bert_nomark)) * [0])
#
#             sequence_orig_to_tok_bert = (orig_to_tok_bert[i] + (args.max_sent_len - len(sequence_bert_nomark)) * [0])
#             i += 1
#             while i <= index_in_file:
#                 sent_bert_nomark = vectorized_sentences_bert_nomark[i]
#                 max_index_bert = len(sent_bert_nomark)
#                 sent_bert_nomark = sent_bert_nomark[:min(args.max_sent_len, max_index_bert)]
#                 sent_bert_mask = (len(sent_bert_nomark) * [1] + (args.max_sent_len - len(sent_bert_nomark)) * [0])
#                 sent_bert_nomark = (sent_bert_nomark + (args.max_sent_len - len(sent_bert_nomark)) * [0])
#                 assert sent_bert_nomark[0] == 101
#                 sent_bert_nomark[0] = 102
#                 sequence_orig_to_tok_bert.extend(
#                     (orig_to_tok_bert[i] + (args.max_sent_len - len(sent_bert_nomark)) * [0]))  # 是句子内的相对位置!!!!!!!!
#                 sequence_bert_nomark.extend(sent_bert_nomark)
#                 sequence_bert_mask.extend(sent_bert_mask)
#                 i += 1
#             # max_index_sequence = len(sequence_bert_nomark)
#             # sequence_bert_nomark = sequence_bert_nomark[max(-args.max_sent_len, -max_index_sequence):]
#             # sent_bert_mask = ((args.max_sent_len - len(sequence_bert_nomark)) * [0]+len(sequence_bert_nomark) * [1])
#             # sequence_bert_nomark = ((args.max_sent_len - len(sequence_bert_nomark)) * [0]+sequence_bert_nomark )
#
#             zp_sent_bert.append(vectorized_sentences_bert[index_in_file])
#             zp_np_sent_bert.append(sequence_bert_nomark)
#             zp_np_sent_bert_mask.append(sent_bert_mask)
#             zp_np_orig_to_tok_bert.append(sequence_orig_to_tok_bert)
#
#             zp_sent_mask_bert.append(vectorized_sentences_mask_bert[index_in_file])  # no best,直接计算比较好
#             zp_orig_to_tok_bert.append(orig_to_tok_bert[index_in_file])
#
#             candi_vec_index_inside = []
#             for candi_index_in_file, candi_sentence_index, candi_begin, candi_end, res, target, ifl in candi_info:
#                 # candi_word_embedding_indexs = vectorized_sentences[candi_index_in_file]
#                 # candi_vec = candi_word_embedding_indexs[candi_begin:candi_end+1]
#                 # if len(candi_vec) >= 8:#限制candi长度max为8
#                 #     candi_vec = candi_vec[-8:]
#                 # candi_mask = (8-len(candi_vec))*[0] + len(candi_vec)*[1]
#                 # candi_vec = (8-len(candi_vec))*[0] + candi_vec
#                 #
#                 # candi_vecs.append(candi_vec)
#                 # candi_vecs_mask.append(candi_mask)
#
#                 ifl_vecs.append(ifl)
#
#                 # candi_vec_index_inside.append((candi_vec_index,res,target))
#                 candi_vec_index_inside.append((candi_vec_index, candi_begin, candi_end, res, target))
#                 candi_vec_index += 1
#
#             # zp_candi_target.append((zp_vec_index,candi_vec_index_inside)) #(zpi,candis(candij,res,target))
#             zp_candi_target.append(((zp_vec_index, zp_index), candi_vec_index_inside))
#             zp_vec_index += 1
#     save_f = open(data_path + "zp_candi_pair_info", 'wb')
#     pickle.dump(zp_candi_target, save_f, protocol=pickle.HIGHEST_PROTOCOL)
#     save_f.close()
#
#     # zp_prefixs = numpy.array(zp_prefixs,dtype='int32')
#     # numpy.save(data_path+"zp_pre.npy",zp_prefixs)
#     # zp_prefixs_mask = numpy.array(zp_prefixs_mask,dtype='int32')
#     # numpy.save(data_path+"zp_pre_mask.npy",zp_prefixs_mask)
#     # zp_postfixs = numpy.array(zp_postfixs,dtype='int32')
#     # numpy.save(data_path+"zp_post.npy",zp_postfixs)
#     # zp_postfixs_mask = numpy.array(zp_postfixs_mask,dtype='int32')
#     # numpy.save(data_path+"zp_post_mask.npy",zp_postfixs_mask)
#
#     zp_sent_bert = numpy.array(zp_sent_bert, dtype='int32')
#     numpy.save(data_path + "zp_sent_bert.npy", zp_sent_bert)
#
#     zp_np_sent_bert = numpy.array(zp_np_sent_bert, dtype='int32')
#     numpy.save(data_path + "zp_np_sent_bert.npy", zp_np_sent_bert)
#
#     zp_np_sent_bert_mask = numpy.array(zp_np_sent_bert_mask, dtype='int32')
#     numpy.save(data_path + "zp_np_sent_bert_mask.npy", zp_np_sent_bert_mask)
#
#     zp_sent_mask_bert = numpy.array(zp_sent_mask_bert, dtype='int32')
#     numpy.save(data_path + "zp_sent_mask_bert.npy", zp_sent_mask_bert)
#
#     zp_np_orig_to_tok_bert = numpy.array(zp_np_orig_to_tok_bert, dtype='int32')
#     numpy.save(data_path + "zp_np_orig_to_tok_bert.npy", zp_np_orig_to_tok_bert)
#
#     # zp_orig_to_tok_bert = numpy.array(zp_orig_to_tok_bert, dtype='int32')
#     zp_orig_to_tok_bert = numpy.array(zp_orig_to_tok_bert)
#     numpy.save(data_path + "zp_orig_to_tok_bert.npy", zp_orig_to_tok_bert)
#
#     # candi_vecs = numpy.array(candi_vecs,dtype='int32')
#     # numpy.save(data_path+"candi_vec.npy",candi_vecs)
#     # candi_vecs_mask = numpy.array(candi_vecs_mask,dtype='int32')
#     # numpy.save(data_path+"candi_vec_mask.npy",candi_vecs_mask)
#
#     # assert len(ifl_vecs) == len(candi_vecs)
#
#     ifl_vecs = numpy.array(ifl_vecs, dtype='float')
#     numpy.save(data_path + "ifl_vec.npy", ifl_vecs)

#不要删！！！！

import torch
import random
import math
import os
import shutil
def youlishu(x):
    x=str(x)
    if len(x)<17: #Python的浮点型最多有17位
        return True
    else:
        x=x[x.find('.')+1:] #截取小数点后面的数字
        for i1 in x: #判断是否循环
            weizhi1=x.find(i1)
            weizhi2=x.find(i1,weizhi1+1)
            if weizhi2+3<=len(x):
                if x[weizhi1+1]== x[weizhi2+1] and x[weizhi1+2]== x[weizhi2+2]:
                    return True
                else:
                    return False
            else:
                return False
def main():
    f=open("./out.txt",'w')
    a=range(0,100)
    a = [i * 0.01 for i in a]
    b=range(0,100)
    b = [i * 0.01 for i in b]

    for aa in a:
        for bb in b:
            x=-320*aa**3*bb**3+460*aa**4*bb**2+80*aa**2*bb**4-260*aa**5*bb+50*aa**6
            if x<0:
                continue
            m=-216*aa*bb**2+268*aa*aa*bb-88*aa**3+64*bb**3+math.sqrt(x)*10
            x = 10*aa**6-20*aa**5*bb-20*aa**4*bb**2+80*aa**2*bb**4
            if x < 0:
                continue
            n=-28*aa**3+28*aa**2*bb+24*aa*bb**2+64*bb**3+10*math.sqrt(x)
            m=pow(m, 1.0/3)
            n = pow(n, 1.0 / 3)
            if youlishu(m) and youlishu(n):
                f.write(str(aa)+","+str(bb)+'\n')

    f.close()
if __name__ == "__main__":
    # zp_vec_index,candi_index=getNumZpNp("./data/train/")
    # print(zp_vec_index*4*512/1024/1024/1024)#12090，0.023059844970703125
    # print(candi_index*40*4*256/1024/1024/1024)#1811960，6.912078857421875
    # zp_vec_index1, candi_index1 = getNumZpNp("./data/test/")
    # print(zp_vec_index1*4*512/1024/1024/1024)#1711，0.0032634735107421875
    # print(candi_index1*40*4*256/1024/1024/1024)#22545，0.8600234985351562
    # zp_neicun=(zp_vec_index+zp_vec_index1)*4*512/1024/1024/1024
    # candi_neicun=(candi_index+candi_index1)*40*4*256/1024/1024/1024
    # print(zp_neicun)#0.026323318481445312
    # print(candi_neicun)#7.772102355957031

    # input_ids = random.sample(range(50),20)
    # # masked_lm_labels:[-1,-1,..,mask_idx,-1,..]
    # mask_lm_labels = random.sample(range(len(input_ids)), int(len(input_ids) * 0.15))
    # mask_lm_labels.sort()
    # random_input_ids = input_ids[:mask_lm_labels[0]] + [random.randint(0,21128)]  # vocab_size=21128
    # label_id = [-1] * mask_lm_labels[0] + [input_ids[mask_lm_labels[0]]]
    # i = 1
    # while i < len(mask_lm_labels):
    #     random_input_ids.extend(input_ids[mask_lm_labels[i - 1] + 1:mask_lm_labels[i]] + [random.randint(0,21128)])
    #     label_id.extend([-1] * (mask_lm_labels[i] - mask_lm_labels[i - 1] - 1) + [input_ids[mask_lm_labels[i]]])
    #     i += 1
    # label_id.extend([-1] * (len(input_ids) - mask_lm_labels[- 1] - 1))
    # random_input_ids.extend(input_ids[mask_lm_labels[i - 1] + 1:(len(input_ids))])
    # assert len(input_ids) == len(random_input_ids)
    #
    # label_ids=label_id

    # main()

    output_dir = "./out/"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
