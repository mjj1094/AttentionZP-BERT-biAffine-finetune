#coding=utf8
import os
import sys
import re
import math
import timeit
from subprocess import *
from conf import *
import pickle
import collections
import codecs
sys.setrecursionlimit(1000000)

class DataGnerater():
    def __init__(self,file_type,max_pair):
        data_path = args.data+file_type+"/" 
        if args.reduced == 1:
            data_path = args.data+file_type + "_reduced/"
        # self.candi_vec = numpy.load(data_path+"candi_vec.npy")
        # self.candi_vec_mask = numpy.load(data_path+"candi_vec_mask.npy")
        # self.candi_vec_bert = numpy.load(data_path + "candi_vec_bert.npy")
        # self.candi_vec_mask_bert = numpy.load(data_path + "candi_vec_mask_bert.npy")
        self.ifl_vec = numpy.load(data_path+"ifl_vec.npy")
        # self.zp_sent_cls_output_bert = numpy.load(data_path + "zp_sent_cls_output_bert.npy")
        # self.zp_np_sent_bert = numpy.load(data_path + "zp_np_sent_bert.npy")#[[CLS],...,[SEQ]]--->idx_bert
        read_f = codecs.open(data_path + "zp_np_sent_bert", "rb")
        self.zp_np_sent_bert = pickle.load(read_f)
        read_f.close()
        read_f = codecs.open(data_path + "zp_np_sent_bert_mask", "rb")
        self.zp_np_sent_bert_mask = pickle.load(read_f)
        read_f.close()

        # self.zp_np_orig_to_tok_bert = numpy.load(data_path + "zp_np_orig_to_tok_bert.npy")  #word piece-->token
        # self.zp_np_sent_bert_mask = numpy.load(data_path + "zp_np_sent_bert_mask.npy")

        read_f = codecs.open(data_path + "zp_candi_pair_info","rb")
        zp_candis_pair = pickle.load(read_f)
        read_f.close()
        self.data_batch = []
        zp_rein = []
        candi_rein = []
        candi_idx = []
        this_target = []
        this_result = []

        # zps_idx=[]
        # zp_sent_idx=[]
        # candis_begin=[]
        # candis_end=[]

        s2e = []
        for i in range(len(zp_candis_pair)):
            (zpi,zp_idx),candis = zp_candis_pair[i]
            if len(candis)+len(candi_rein) > max_pair and len(candi_rein) > 0:
                ci_s = candi_rein[0]
                ci_e = candi_rein[-1]+1
                zpi_s = zp_rein[0]
                zpi_e = zp_rein[-1]+1
                this_batch = {}

                # this_batch["zp_sent_idx"] = numpy.array(zp_sent_idx, dtype="int32")
                # this_batch["zp_idx"] = numpy.array(zps_idx, dtype="int32")
                # this_batch["candis_begin"] = numpy.array(candis_begin, dtype="int32")
                # this_batch["candis_end"] = numpy.array(candis_end, dtype="int32")
                # this_batch["target"] = numpy.array(this_target,dtype="int32")
                # this_batch["result"] = numpy.array(this_result,dtype="int32")
                # this_batch["zp_np_sent_bert_mask"] = self.zp_np_sent_bert_mask[zpi_s:zpi_e]
                # this_batch["zp_np_orig_to_tok_bert"] = self.zp_np_orig_to_tok_bert[zpi_s:zpi_e]

                # this_batch["zp_sent_cls_output_bert"] = self.zp_sent_cls_output_bert[zpi_s:zpi_e]
                # this_batch["candi_bert"] = self.candi_vec_bert[ci_s:ci_e]
                # this_batch["candi_mask_bert"] = self.candi_vec_mask_bert[ci_s:ci_e]
                this_batch["zp_reindex"] = numpy.array(zp_rein, dtype="int32") - zp_rein[0]
                this_batch["candi_reindex"] = numpy.array(candi_rein, dtype="int32") - candi_rein[0]
                this_batch["candi_sent_idx"] = numpy.array(candi_idx, dtype="int32")
                this_batch["zp_np_sent_bert"] = self.zp_np_sent_bert[zpi_s:zpi_e]
                this_batch["zp_np_sent_bert_mask"] = self.zp_np_sent_bert_mask[zpi_s:zpi_e]
                this_batch["fl"] = self.ifl_vec[ci_s:ci_e]
                this_batch["start2end"] = s2e
                self.data_batch.append(this_batch)
                zp_rein = []
                candi_rein = []
                candi_idx = []

                # zps_idx = []
                # zp_sent_idx=[]
                # candis_begin = []
                # candis_end = []

                this_target = []
                this_result = []
                s2e = []
            start = len(this_result)
            end = start
            for candii,candi_sent_idx,candi_begin,candi_end, res, tar in candis:
                zp_rein.append(zpi)
                candi_rein.append(candii)
                candi_idx.append((candi_sent_idx,candi_begin,candi_end))

                # zps_idx.append(zp_idx)
                # zp_sent_idx.append(zp_sent_index)
                # candis_begin.append(candi_begin)
                # candis_end.append(candi_end)

                this_target.append(tar)
                this_result.append(res)
                end += 1
            s2e.append((start,end))
        if len(candi_rein) > 0:
            ci_s = candi_rein[0]
            ci_e = candi_rein[-1]+1
            zpi_s = zp_rein[0]
            zpi_e = zp_rein[-1]+1
            this_batch = {}

            # this_batch["zp_sent_idx"] = numpy.array(zp_sent_idx, dtype="int32")
            # this_batch["zp_idx"] = numpy.array(zps_idx, dtype="int32")
            # this_batch["candis_begin"] = numpy.array(candis_begin, dtype="int32")
            # this_batch["candis_end"] = numpy.array(candis_end, dtype="int32")
            # this_batch["target"] = numpy.array(this_target,dtype="int32")
            # this_batch["result"] = numpy.array(this_result,dtype="int32")
            # this_batch["zp_np_sent_bert_mask"] = self.zp_np_sent_bert_mask[zpi_s:zpi_e]
            # this_batch["zp_np_orig_to_tok_bert"] = self.zp_np_orig_to_tok_bert[zpi_s:zpi_e]

            # this_batch["zp_sent_cls_output_bert"] = self.zp_sent_cls_output_bert[zpi_s:zpi_e]
            # this_batch["candi_bert"] = self.candi_vec_bert[ci_s:ci_e]
            # this_batch["candi_mask_bert"] = self.candi_vec_mask_bert[ci_s:ci_e]
            this_batch["zp_reindex"] = numpy.array(zp_rein, dtype="int32") - zp_rein[0]
            this_batch["candi_reindex"] = numpy.array(candi_rein, dtype="int32") - candi_rein[0]
            this_batch["candi_sent_idx"] = numpy.array(candi_idx, dtype="int32")
            this_batch["zp_np_sent_bert"] = self.zp_np_sent_bert[zpi_s:zpi_e]
            this_batch["zp_np_sent_bert_mask"] = self.zp_np_sent_bert_mask[zpi_s:zpi_e]
            this_batch["fl"] = self.ifl_vec[ci_s:ci_e]
            this_batch["start2end"] = s2e
            self.data_batch.append(this_batch)
    def devide(self,k=0.2,shuffle=False):
        # random.shuffle(self.data_batch)
        if shuffle:
            random.shuffle(self.data_batch)
        length = int(len(self.data_batch)*k)
        self.dev = self.data_batch[:length]
        self.train = self.data_batch[length:]
        self.data_batch = self.train
    def generate_data(self,shuffle=False):
        if shuffle:
            random.shuffle(self.data_batch) 
        estimate_time = 0.0 
        done_num = 0 
        total_num = len(self.data_batch)

        for data in self.data_batch:
            start_time = timeit.default_timer()
            done_num += 1
            yield data
            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            info = "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)
            sys.stderr.write(info+"\r")
        print(file=sys.stderr)
    def generate_dev_data(self,shuffle=False):
        if shuffle:
            random.shuffle(self.dev) 
        estimate_time = 0.0 
        done_num = 0
        total_num = len(self.dev)
        for data in self.dev:
            start_time = timeit.default_timer()
            done_num += 1
            yield data
            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            info = "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)
            sys.stderr.write(info+"\r")
        print(file=sys.stderr)