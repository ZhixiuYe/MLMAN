import random
import torch
import numpy as np
np.set_printoptions(threshold=np.nan)
torch.set_printoptions(precision=8)

from models.data_loader import JSONFileDataLoader
from models.framework import FewShotREFramework
from models.MLMAN import MLMAN as MLMAN

seed = int(np.random.uniform(0,1)*10000000)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
print('seed: ', seed)
import argparse

parser = argparse.ArgumentParser(description='Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification')
parser.add_argument('--model_name', type=str, default='MLMAN', help='Model name')
parser.add_argument('--N_for_train', type=int, default=20, help='Num of classes for each batch for training')
parser.add_argument('--N_for_test', type=int, default=5, help='Num of classes for each batch for test')
parser.add_argument('--K', type=int, default=1, help='Num of instances for each class in the support set')
parser.add_argument('--Q', type=int, default=5, help='Num of instances for each class in the query set')
parser.add_argument('--batch', type=int, default=1, help='batch size')
parser.add_argument('--max_length', type=int, default=40, help='max length of sentence')
parser.add_argument('--learning_rate', type=float, default=1e-1, help='initial learning rate')

args = parser.parse_args()
print('setting:')
print(args)

print("{}-way(train)-{}-way(test)-{}-shot with batch {} Few-Shot Relation Classification".format(args.N_for_train, args.N_for_test, args.K, args.Q))
print("Model: {}".format(args.model_name))

max_length = args.max_length

train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length, reprocess=False)
val_data_loader = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length, reprocess=False)

framework = FewShotREFramework(train_data_loader, val_data_loader, val_data_loader)

model = MLMAN(train_data_loader.word_vec_mat, max_length, hidden_size=100, args=args)
model_name = args.model_name + str(seed)
framework.train(model, model_name, args.batch, N_for_train=args.N_for_train,  N_for_eval=args.N_for_test,
                K=args.K, Q=args.Q,  learning_rate=args.learning_rate,
                train_iter=50000, val_iter=1000, val_step=2000, test_iter=2000)
