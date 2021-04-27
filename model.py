#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        self.evaluator = evaluator
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        logging.info('Sample type {}'.format(type(sample)))
        logging.info('Sample[0]aaa: {}'.format(sample[0]))   # sample[0]是
        logging.info('Sample[1]bbb: {}'.format(sample[1]))
        logging.info('Sample shape {}'.format(len(sample)))
        logging.info("Start construct entity dic")

        logging.info("sample[0].size:{}".format(sample[0].size()))
        logging.info("sample[1].size:{}".format(sample[1].size()))

        # logging.info("Start construct entity dic")
        # dic_entity = {}
        # for i, v in tqdm(enumerate(np.unique(sample[0][:, [0, 2]]))):
        #     dic_entity[v] = i
        # logging.info("dic_entity.len:".format(len(dic_entity)))
        # logging.info("Start construct relation dic")
        # dic_relation = {}
        # for i, v in tqdm(enumerate(np.unique(sample[0][:, 1]))):
        #     dic_relation[v] = i
        logging.info("embedding is {}".format(self.entity_embedding))
        logging.info("mode is {}".format(mode))
        if mode == 'single':
            logging.info("mode={}".format(mode))
            logging.info("single mode {}".format(sample[0]))
            batch_size, negative_sample_size = sample[0].size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[0][0] #torch.tensor([i for i in sample[0][:, 0]])  #[dic_entity[int(i[0])] for i in sample[0][:, 0]]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[0][1] #torch.tensor([i for i in sample[0][:, 1]]) #sample[:,1]    [dic_relation[int(i[0])] for i in sample[0][:, 1]]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[0][2] #torch.tensor([i for i in sample[0][:, 2]])  #[dic_entity[int(i[0])] for i in sample[0][:, 2]]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            logging.info("mode={}".format(mode))  #mode=head-batch
            logging.info("sample: {}".format(sample))
            tail_part, head_part = sample[0][:, 0], sample[0][:, 2] #sample
            batch_size, negative_sample_size = sample[0].size(0), 1 #head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            # 这里sample[0]是triplets，sample[1]是刚才抽出来的3：6的错误的test node
            logging.info("mode=tail-batch:{}".format(sample[0]))
            logging.info("sample: {}".format(sample))
            logging.info("embedding's dimention:{}".format(self.entity_embedding.shape))
            head_part, tail_part = sample[0][:, 0], sample[0][:, 2]
            batch_size, negative_sample_size = sample[0].size(0), 1#tail_part.size(0), tail_part.size(1)
            logging.info("max node of test:{}".format(torch.max(head_part)))
            logging.info("tail_part: {}".format(tail_part))
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index= head_part #torch.Tensor([i for i in head_part]) #[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[0][:,1] #torch.Tensor([i for i in head_part]) #head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part #torch.Tensor([i for i in tail_part]) #tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        logging.info("这里这里！训练的positive score为：{}".format(positive_score))

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        logging.info("损失为：{}".format(loss))
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        
        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, args, random_sampling=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()

        #Prepare dataloader for evaluation
        # test_dataloader_head = DataLoader(
        #     TestDataset(
        #         test_triples,
        #         args,
        #         'head-batch',
        #         random_sampling
        #     ),
        #     batch_size=args.test_batch_size,
        #     num_workers=0, #max(1, args.cpu_num//2),
        #     collate_fn=TestDataset.collate_fn
        # )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                args, 
                'tail-batch',
                random_sampling
            ), 
            batch_size=args.test_batch_size,
            num_workers=0, #max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_tail] #test_dataloader_head,
        
        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)  # 这里返回的是定义的kegmodel里forward部分的返回值score

                    input_dict = {}
                    t_pred_top10 = np.array([1, 2, 3])
                    t_correct_index = np.array(positive_sample[:,2])
                    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}
                    batch_results = model.evaluator.eval(input_dict)  #{'y_pred_pos': score[:, 0],
                                                #'y_pred_neg': score[:, 1:]}
                    logging.info("MRR is:{}".format(batch_results['mrr']))

                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics
