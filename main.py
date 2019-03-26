#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.setrecursionlimit(1000000)

import logging
import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from data_generater import *
from net_with_bert import *
# from net_with_pretrained_bert import *

print("PID", os.getpid(), file=sys.stderr)
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)
import datetime

TIME = datetime.datetime.now()

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]

def get_predict_max(data):
    predict = []
    for result,output in data:
        max_index = -1
        max_pro = 0.0
        for i in range(len(output)):
            if output[i][1] > max_pro:
                max_index = i
                max_pro = output[i][1]
        predict.append(result[max_index])
    return predict
 
 
def get_evaluate(data,overall=1713.0):
    best_result = {}
    best_result["hits"] = 0
    predict = get_predict_max(data)
    result = evaluate(predict,overall)
    if result["hits"] > best_result["hits"]:
        best_result = result
    return best_result

def evaluate(predict,overall):
    result = {}
    result["hits"] = sum(predict)
    result["performance"] = sum(predict)/overall
    return result

MAX = 2

def main():
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    batch_size = int( nnargs["batch_size"]/ args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Prepare model
    read_f = codecs.open(args.data+"train_data", "rb")
    train_generater = pickle.load(read_f, encoding='latin1')
    read_f.close()
    test_generater = DataGnerater("test", nnargs["batch_size"])#256->1

    print("Building torch model")
    # model = Network(nnargs["hidden_dimention"], 2)
    model = Network.from_pretrained(args.bert_dir,
                                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                    hidden_dimention=nnargs["hidden_dimention"],
                                    output_dimention=2)

    best_result = {}
    best_result["hits"] = 0
    # best_model = Network(nnargs["hidden_dimention"], 2)
    best_model = Network.from_pretrained(args.bert_dir,
                                         cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                             args.local_rank),
                                         hidden_dimention=nnargs["hidden_dimention"],
                                         output_dimention=2)

    if args.fp16:
        model.half()
    model.to(device)
    best_model.to(device)

    this_lr = 0.003
    optimizer = optim.Adagrad(model.parameters(), lr=this_lr)  # -------------------
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
#-----------------------------------------------------------------------------------------------------------------------

    for echo in range(args.num_train_epochs):
        cost = 0.0
        print("Begin epoch",echo, file=sys.stderr)
        for data in train_generater.generate_data(shuffle=False):
            # output,output_softmax = model.forward(data,dropout=nnargs["dropout"])#no .forward(),for 3
            output, output_softmax = model.forward(data, dropout=nnargs["dropout"])  # no .forward()
            if len(output.size()) == 1 and output.size()[0] == 2:
                output = torch.unsqueeze(output, 0)
                # print(input.size())
            loss = F.cross_entropy(output,torch.tensor(data["result"]).type(torch.cuda.LongTensor))
            optimizer.zero_grad()
            cost += loss.item()
            loss.backward()
            optimizer.step()
        print("End epoch",echo,"Cost:", cost, file=sys.stderr)
        predict = []
        for data in train_generater.generate_dev_data():
            output,output_softmax = model.forward(data)
            for s,e in data["start2end"]:
                if s == e:
                    continue
                predict.append((data["result"][s:e],output_softmax[s:e]))

        result = get_evaluate(predict,float(len(predict)))
        if result["hits"] > best_result["hits"]:
            print("best echo:",echo)
            best_result = result
            best_result["epoch"] = echo
            print("dev:", best_result["performance"])
            net_copy(best_model,model)
        sys.stdout.flush()
    torch.save(best_model,"./model/best_model"+str(TIME.month)+"-"+str(TIME.day))
    predict = []
    for data in test_generater.generate_data():
        output,output_softmax = best_model.forward(data)
        for s,e in data["start2end"]:
            if s == e:
                continue
            predict.append((data["result"][s:e],output_softmax[s:e]))
    result = get_evaluate(predict)
    # print("dev:",best_result["performance"])
    print("test:",result["performance"])
    print("total echo:",echo)
if __name__ == "__main__":
    main()
