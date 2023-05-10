#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import math
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, BertForMultipleChoice, BertTokenizer
from transformers import XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer
from transformers import RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer
from transformers import GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer

from common.DiscrimLoss import (DiscrimEA_TANHLoss, DiscrimEA_EMAK_TANHLoss, DiscrimEA_GAK_TANHLoss)
from dataset.utils_multiple_choice import convert_examples_to_features, SwagProcessor

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMultipleChoice, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
    "gpt": (GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer)
}


def get_WIKIHOW_train_val_test_loader(args):
    """"
    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader for WIKIHOW train data.
        val_loader (torch.utils.data.DataLoader): data loader for WIKIHOW val data.
    """
    print('==> Preparing data for WIKIHOW..')
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        args.model_path,
        do_lower_case=True,
    )

    trainset, valset, = load_and_cache_examples(args.train_data_path, tokenizer, args.max_seq_length, "train",
                                                model_name_or_path=args.model_type,split_rate=(0.9, 0.1), seed=args.seed)
    testset, _ = load_and_cache_examples(args.train_data_path, tokenizer, args.max_seq_length, "val",
                                         model_name_or_path=args.model_type,split_rate=(1, 0), seed=args.seed)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # args.per_gpu_train_batch_size
    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              shuffle=True, )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # args.per_gpu_eval_batch_size
    val_loader = DataLoader(valset,
                            batch_size=args.eval_batch_size,
                            shuffle=False, )  # num_workers=args.workers)

    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)  # args.per_gpu_eval_batch_size
    test_loader = DataLoader(testset,
                             batch_size=args.test_batch_size,
                             shuffle=False, )  # num_workers=args.workers)
    return train_loader, val_loader, test_loader


def get_WIKIHOW_model_and_loss_criterion(args, params=None, ITERATION=None):
    """

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module):
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    print('Building WiKiHow')

    args.arch = 'WikiHow_RoBerta'  # 20epochs with SGD
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.model_path,
        num_labels=args.nr_classes,
        finetuning_task="swag",
    )

    model = model_class.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config,
    )

    if ITERATION is None:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    else:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '_hy_iter_%d.csv' % ITERATION

    model.to(args.device)
    # todo 选择 discrimloss
    a = args.a if params is None else params['tanh_a']
    p = args.p if params is None else params['tanh_p']
    q = -args.newq * args.p if params is None else -params['tanh_q'] * params['tanh_p']
    sup_eps = args.sup_eps if params is None else params['sup_eps']

    if args.wikihow_loss_type == "ea_gak_tanh_newq":
        criterion = DiscrimEA_GAK_TANHLoss(a=a, p=p,
                                           q=q, sup_eps=sup_eps).to(args.device)
    elif args.wikihow_loss_type == "ea_emak_tanh_newq":
        criterion = DiscrimEA_EMAK_TANHLoss(a=a, p=p,
                                            q=q, sup_eps=sup_eps).to(args.device)
    elif args.wikihow_loss_type == "ea_tanh_newq":
        criterion = DiscrimEA_TANHLoss(k1=0.5, a=a, p=p,
                                       q=q, sup_eps=sup_eps).to(args.device)
    elif args.wikihow_loss_type == "no_cl":
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    else:
        assert False
    # TODO-jlgao 根据不同的数据类型，需要使用不同到的criterion_val
    criterion_val = nn.CrossEntropyLoss(reduction='none').to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    # Distributed training (should be after apex fp16 initialization)
    assert args.local_rank == -1
    # if args.local_rank != -1:
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank,
    #                                                   find_unused_parameters=True)

    return model, criterion, criterion_val, logname


def load_and_cache_examples(data_dir, tokenizer, max_seq_length, cached_mode, task="swag",
                            model_name_or_path="roberta", overwrite_cache=False, split_rate=(1, 0), seed=0):
    # print("start load_and_cache_examples: ",cached_mode)
    processor = SwagProcessor()
    # Load data features from cache or dataset file

    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not overwrite_cache:
        print("Loading features from cached file %s" % cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s" % data_dir)
        label_list = processor.get_labels()
        if cached_mode == "train":
            examples = processor.get_train_examples(data_dir)
        elif cached_mode == "val":
            examples = processor.get_dev_examples(data_dir)
        else:
            assert False
        print("number: %s" % str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            max_seq_length,
            tokenizer,
            pad_on_left=bool(model_name_or_path in ["xlnet"]),
            pad_token_segment_id=4 if model_name_or_path in ["xlnet"] else 0
        )
        print("Saving features into cached file %s" % cached_features_file)
        torch.save(features, cached_features_file)
    # print("start split datatensor")
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    tmp_index = list(range(len(features)))
    random.seed(seed)
    random.shuffle(tmp_index)
    tmp_size = math.floor(len(features) * split_rate[0] / sum(split_rate))

    dataset = []
    dataset.append(TensorDataset(all_input_ids[tmp_index[:tmp_size]], all_input_mask[tmp_index[:tmp_size]],
                                 all_segment_ids[tmp_index[:tmp_size]], all_label_ids[tmp_index[:tmp_size]],
                                 torch.tensor(range(tmp_size), dtype=torch.long)) if tmp_size > 0 else None)
    dataset.append(TensorDataset(all_input_ids[tmp_index[tmp_size:]], all_input_mask[tmp_index[tmp_size:]],
                                 all_segment_ids[tmp_index[tmp_size:]], all_label_ids[tmp_index[tmp_size:]],
                                 torch.tensor(range(len(features) - tmp_size), dtype=torch.long)) if len(
        features) - tmp_size > 0 else None)
    return dataset


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]
