import os
import torch
from torch import optim


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    

    def train(self, model, model_name, B=4, N_for_train=20, N_for_eval=5, K=5, Q=100,
              ckpt_dir='./checkpoint', learning_rate=1e-1, lr_step_size=20000,
              weight_decay=1e-5, train_iter=30000, val_iter=1000, val_step=2000,
              test_iter=3000, pretrain_model=None, optimizer=optim.SGD):
        '''
        model: model
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        # Init
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        model = model.cuda()
        model.train()

        # Training
        best_acc = 0
        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            support, query, label = self.train_data_loader.next_batch(B, N_for_train, K, Q)
            logits, pred, dist = model(support, query, N_for_train, K, Q)
            loss = model.loss(logits, label)
            allloss = loss + dist
            optimizer.zero_grad()
            allloss.backward()
            optimizer.step()

            if (it + 1) % val_step == 0:
                with torch.no_grad():
                    acc = self.eval(model, 5, N_for_eval, K, 5, val_iter)
                    print("{0:}---{1:}-way-{2:}-shot test   Test accuracy: {3:3.2f}".format(it, N_for_eval, K, acc*100))

                    if acc > best_acc:
                        print('Best checkpoint')
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                        torch.save({'state_dict': model.state_dict()}, save_path)
                        best_acc = acc
                model.train()

        print("\n####################\n")
        print("Finish training " + model_name)
        with torch.no_grad():
            test_acc = self.eval(model, 5, N_for_eval, K, 5, test_iter, ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'))
            print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(N_for_eval, K, test_acc*100))

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            eval_dataset = self.test_data_loader
        model.eval()

        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            support, query, label = eval_dataset.next_batch(B, N, K, Q)
            _, pred, _ = model(support, query, N, K, Q)
            right = model.accuracy(pred, label)
            iter_right += right.item()
            iter_sample += 1
        return iter_right / iter_sample
