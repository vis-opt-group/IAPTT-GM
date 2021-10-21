import math
import argparse
import time
import itertools
import os,sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torchmeta.datasets.helpers import omniglot, miniimagenet, tieredimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from few_shot.ResNet12 import ResNet12,MetaLinearLayer
from few_shot.ConvNet import ConvolutionalNeuralNetwork,TaskLinearLayer
import higher
import csv

parser = argparse.ArgumentParser(description='Few shot Classification')

parser.add_argument('--dataset', type=str, default='miniimagenet'
                                                   '', metavar='N', help='omniglot or miniimagenet or tieredImagenet')
parser.add_argument('--network', type=str, default='convnet'
                                                   '', metavar='N', help='convnet for ConvNet-4 or resnet for ResNet-12')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--reg_param', type=float, default=0.25, help="coefficient of regularization part")
parser.add_argument('--exp_param', type=float, default=0.5, help="coefficient of regularization part")
parser.add_argument('--epi_param', type=float, default=0.1, help="coefficient of regularization part")
parser.add_argument('--regularization', type=str, default=None, help="distance_norm or p_norm")

parser.add_argument('--ways', type=int, default=5, help="to define the number of classes, like N way K shot")
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--shots', type=int, default=1, help="to define the number of samples to use at the training "
                                                       "phase, like N way K shot")
parser.add_argument('--test_shots', type=int, default=15, help="to define the number of samples to use at the "
                                                             "testing phase")

parser.add_argument('--image_channels', type=int, default=3, help='number of channels of input images')
parser.add_argument('--image_height', type=int, default=84, help='height  of input images')
parser.add_argument('--image_width', type=int, default=84, help='width of channels of input images')
# Conv 4
parser.add_argument('--hidden_size', type=int, default=32, help='hidden size for CNN')
parser.add_argument('--cnn_num_filters', type=int, default=48, help='number of filters for CNN')

# ResNet12
parser.add_argument('--num_stages', type=int, default=4, help='number of stages for ResNet12')
parser.add_argument('--max_pooling', type=bool, default=True,
                    help='whether to use max_pooling layer in the Residual blocks')
parser.add_argument('--per_step_bn_statistics', type=bool, default=False,
                    help='whether to use max_pooling layer in the Residual blocks')
parser.add_argument('--norm_layer', type=str, default='batch_norm', metavar='N', help='number of stages for ResNet12')

parser.add_argument('--learnable_bn_gamma', type=bool, default=True,
                    help='whether to learn gamma')
parser.add_argument('--learnable_bn_beta', type=bool, default=True,
                    help='whether to learn beta ')
parser.add_argument('--enable_inner_loop_optimizable_bn_params', type=bool, default=False,
                    help='whether to enable_inner_loop_optimizable_bn_params')
parser.add_argument('--number_of_evaluation_steps_per_iter', type=int, default=5, help='number_of_evaluation_steps_per_iter for ResNet12')


parser.add_argument('--learn_lr', type=bool, default=False,
                    help='whether to learn beta ')
parser.add_argument('--batch_size', type=int, default=4, help="number of batches for meta training phase")
parser.add_argument('--inner_loop', type=int, default=10, help="the number of loops for inner gradient steps")
parser.add_argument('--meta_iter', type=int, default=80000, help="total number of iterations for meta training")
parser.add_argument('--test_loop', type=int, default=10, help="total number of iterations for meta training")
parser.add_argument('--eval_interval', type=int, default=500, help="total number of iterations for meta training")
parser.add_argument('--log_interval', type=int, default=100, help="total number of iterations for meta training")
parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate for inner training steps")
parser.add_argument('--min_learning_rate', type=float, default=0.001, help="learning rate for inner training steps")
parser.add_argument('--dfc', type=bool,default=False, help='two fully connect layers')
parser.add_argument('--task_conv', type=int,default=0, help='number of conv layers before fully connect layer as base learner')
parser.add_argument('--spectral_norm',type=bool,default=False,help='whether to user spectral normalization')
parser.add_argument('--Notes', type=str, default=' ', metavar='N', help='Additional Notes')
args = parser.parse_args()

def acc(out, target):
    pred = out.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item() / len(target)


def p_norm_reg(parameters,exp,epi):
    loss = 0
    for w in parameters:
        loss += (torch.norm(w,2)+torch.norm(epi*torch.ones_like(w),2))**(exp/2)
    return loss

def bias_reg_f(bias, params):
    # l2 biased regularization
    return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])

def distance_reg(output,label, params, hparams):
    # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
    return F.cross_entropy(output, label) + args.reg_param * bias_reg_f(hparams, params)

def main():

    log_interval = 100
    eval_interval = 500
    inner_log_interval = None
    inner_log_interval_test = None

    n_tasks_test = 1000  # usually 1000 tasks are used for testing

    T_test = args.test_loop

    loc = locals()
    loss_func= F.cross_entropy
    print(args, '\n', loc, '\n')

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'omniglot':
        dataset = omniglot("data", ways=args.ways, shots=args.shots, test_shots=args.test_shots, meta_train=True, download=True)
        test_dataset = omniglot("data", ways=args.ways, shots=args.shots, test_shots=args.test_shots, meta_test=True, download=True)

        meta_model_x, meta_model_y = get_cnn_omniglot(hidden_size=64, n_classes=args.ways)
        meta_model_x = meta_model_x.to(device)
        meta_model_y = meta_model_y.to(device)
    elif args.dataset == 'miniimagenet' or args.dataset == 'tieredimagenet':
        if args.dataset == 'miniimagenet':
            dataset = miniimagenet("data", ways=args.ways, shots=args.shots, test_shots=args.test_shots, meta_train=True,
                                   download=True)
            test_dataset = miniimagenet("data", ways=args.ways, shots=args.shots, test_shots=args.test_shots,
                                        meta_test=True, download=True)
        elif args.dataset == 'tieredimagenet':
            dataset = tieredimagenet("data",  ways=args.ways, shots=args.shots, test_shots=args.test_shots, meta_train=True,
                                   download=True)
            test_dataset = tieredimagenet("data",  ways=args.ways, shots=args.shots, test_shots=args.test_shots, meta_test=True,
                                        download=True)
        if args.network == 'convnet':
            if args.dfc:
                meta_model_x = ConvolutionalNeuralNetwork(in_channels=3,out_features=args.ways,hidden_size=args.hidden_size,device=device, task_conv = args.task_conv)

                meta_model_y = TaskLinearLayer(in_shape=args.hidden_size*5*5,out_features=args.ways,hidden_size=args.hidden_size,task_conv=args.task_conv,dfc=True)
                initialize(meta_model_x)
                initialize(meta_model_y)
            else:
                meta_model_x = ConvolutionalNeuralNetwork(in_channels=3,out_features=args.ways,hidden_size=args.hidden_size,device=device, task_conv = args.task_conv)

                meta_model_y = TaskLinearLayer(in_shape=args.hidden_size*5*5,out_features=args.ways,hidden_size=args.hidden_size,task_conv=args.task_conv)
                initialize(meta_model_x)
                initialize(meta_model_y)
        elif args.network == 'resnet' :
                meta_model_x = ResNet12(im_shape=(args.batch_size, args.image_channels,
                                                                  args.image_height, args.image_width), num_output_classes=args.ways,device=device,args=args)
                meta_model_y = MetaLinearLayer((meta_model_x.out.shape[0], np.prod(meta_model_x.out.shape[1:])),
                                               num_filters=meta_model_x.num_output_classes, use_bias=True)
        meta_model_x = meta_model_x.to(device)
        meta_model_y = meta_model_y.to(device)


    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, **kwargs)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_size, **kwargs)

    x_opt = torch.optim.Adam(meta_model_x.trainable_parameters())
    y_opt= torch.optim.SGD(meta_model_y.parameters(), lr=args.learning_rate)
    y_lr_schedular= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=y_opt, T_max=args.meta_iter,
                                                         eta_min=args.min_learning_rate)

    z_opt = torch.optim.Adam(meta_model_y.parameters())

    log_path = "{}_ways{}_shots{}_network_{}_hiddenSize{}_batchSize{}_innerLoop{}_testLoop{}_LearningRate{}_" \
               "learnLr{}_MetaIter{}_TaskConv{}_regParam{}_regular{}_SN{}_Notes{}.csv".format(args.dataset,
                                                                                                      args.ways, args.shots, args.network, args.hidden_size, args.batch_size, args.inner_loop,
                                                                                                      args.test_loop, args.learning_rate, args.learn_lr, args.meta_iter, args.task_conv, args.reg_param
                                                                                                      ,  args.regularization, args.spectral_norm, args.Notes)
    with open(log_path, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ['IAPTT-GM', 'val_loss', 'val_acc ', 'test_loss_mean', 'test_loss_std', 'test_acc_mean', 'test_acc_std',
             'forward time', 'backward time','k_max_list'])

    with tqdm(dataloader, total=args.meta_iter,desc="Meta Training Phase") as pbar:
        for meta_iter, batch in enumerate(pbar):
            start_time = time.time()
            meta_model_x.train()
            meta_model_y.train()

            tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
            tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)

            for x in meta_model_x.trainable_parameters():
                if x.grad is not None:
                    x.grad = None
            for y in meta_model_y.parameters():
                if y.grad is not None:
                    y.grad = None

            val_loss, val_acc = 0, 0
            forward_time, backward_time = 0, 0
            pmax_list=[]
            for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
                start_time_task = time.time()
                F_list = []
                with higher.innerloop_ctx(meta_model_y, y_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    forward_time_task = time.time()
                    for y_idx in range(args.inner_loop):
                        if args.network == 'resnet':
                            out_F = meta_model_x.forward(x=tr_x, num_step=y_idx,
                                                                                 training=True,backup_running_statistics=False)
                            out_f = fmodel(out_F)
                        else:
                            out_f = fmodel(meta_model_x(tr_x))
                        if args.regularization == 'distance_norm':
                            loss_f = distance_reg(output=out_f, label=tr_y,
                                                  params=fmodel.parameters(), hparams=fmodel.parameters(time=0))
                        elif args.regularization == 'p_norm':
                            loss_f = loss_func(out_f, tr_y) + args.reg_param * p_norm_reg(parameters=fmodel.parameters(),
                                                exp=args.exp_param,epi=args.epi_param)
                        else:
                            loss_f = loss_func(out_f, tr_y)
                        diffopt.step(loss_f)
                        if args.network == 'resnet':
                            val_out_F = meta_model_x.forward(x=tst_x, num_step=y_idx,
                                                         training=True,backup_running_statistics=False)
                            val_out_f = fmodel(val_out_F)
                        else:
                            val_out_f = fmodel(meta_model_x(tst_x))
                        F_list.append(loss_func(val_out_f, tst_y).item())
                    k_max = F_list.index(max(F_list))
                    pmax_list.append(k_max)
                    forward_time_task = time.time() - forward_time_task
                    forward_time += forward_time_task
                    backward_time_task = time.time()
                    if args.network == 'resnet':
                        val_out_F = meta_model_x.forward(x=tst_x, num_step=y_idx,
                                                     training=True,backup_running_statistics=False)
                        val_out_f = fmodel(val_out_F, params=fmodel.parameters(time=k_max + 1))
                    else:
                        val_out_f = fmodel(meta_model_x(tst_x), params=fmodel.parameters(time=k_max + 1))
                    loss_F = loss_func(val_out_f, tst_y)
                    grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                    grad_x = torch.autograd.grad(loss_F, meta_model_x.trainable_parameters(), retain_graph=True)
                    for p, x in zip(grad_x, meta_model_x.trainable_parameters()):
                        if args.dataset == 'miniimagenet' or args.dataset == 'tieredimagenet':
                            if x.requires_grad:
                                p.data.clamp_(-10, 10)
                        if x.grad == None:
                            x.grad = p
                        else:
                            x.grad += p
                    for p, y in zip(grad_z, meta_model_y.parameters()):
                        if y.grad == None:
                            y.grad = p
                        else:
                            y.grad += p

                    backward_time_task = time.time() - backward_time_task
                    backward_time += backward_time_task
                val_loss += loss_F.item()
                val_acc += acc(val_out_f, tst_y)

            z_opt.step()
            x_opt.step()
            y_lr_schedular.step()

            if meta_iter % log_interval == 0:
                pbar.set_postfix(forward_time='{0:.5f}'.format(forward_time_task),backward_time='{0:.5f}'.format(backward_time_task),
                                 val_acc='{0:.4f}'.format(val_acc / args.batch_size), val_loss='{0:.4f}'.format(val_loss))
            if meta_iter % eval_interval == 0:
                test_losses, test_accs = evaluate(n_tasks=n_tasks_test, dataloader=test_dataloader, meta_model_x=meta_model_x, meta_model_y=meta_model_y, y_loop=T_test)

                print("Test loss {:.2e} +- {:.2e}: Test acc: {:.2f} +- {:.2e} (mean +- std over {} tasks)."
                      .format(test_losses.mean(), test_losses.std(), 100. * test_accs.mean(),
                              100. * test_accs.std(), len(test_losses)))
                print('learning_rate: {} \n'.format(y_lr_schedular.get_last_lr()))
                with open(log_path, 'a', encoding='utf-8', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(
                        [meta_iter, val_loss, val_acc, test_losses.mean(), test_losses.std(), 100. * test_accs.mean(),
                         100. * test_accs.std(), forward_time, backward_time,pmax_list])
            if meta_iter >= args.meta_iter:
                break


def evaluate(n_tasks, dataloader, meta_model_x=None, meta_model_y=None, y_loop=10):
    loss_func= F.cross_entropy
    meta_model_y.train()
    device = next(meta_model_y.parameters()).device

    val_losses, val_accs = [], []
    y_opt = torch.optim.SGD(meta_model_y.parameters(), lr=0.1)
    for k, batch in enumerate(dataloader):
        tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)

        for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):

            with higher.innerloop_ctx(meta_model_y, y_opt, track_higher_grads=False) as (fmodel, diffopt):
                for y_idx in range(y_loop):
                    y_opt.zero_grad()
                    if args.network == 'resnet':
                        val_out_F = meta_model_x.forward(x=tr_x, num_step=y_idx,
                                                         training=True,backup_running_statistics=False).detach()
                        out_y = fmodel(val_out_F)
                    else:
                        out_y = fmodel(meta_model_x(tr_x).detach())
                    loss_y = loss_func(out_y, tr_y)
                    diffopt.step(loss_y)

                if args.network == 'resnet':
                    val_out_F = meta_model_x.forward(x=tst_x, num_step=y_idx,
                                                     training=True,backup_running_statistics=False).detach()
                    out_val = fmodel(val_out_F)
                else:
                    out_val = fmodel(meta_model_x(tst_x).detach())
                val_loss = loss_func(out_val, tst_y)
                val_acc = acc(out_val, tst_y)
                val_losses.append(val_loss.item())
                val_accs.append(val_acc)

                if len(val_accs) >= n_tasks:
                    return np.array(val_losses), np.array(val_accs)

def get_cnn_omniglot(hidden_size, n_classes, spectral_norm=False):
    def conv_layer(ic, oc, spectral_norm=False):
        if spectral_norm:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ic, oc, 3, padding=1), n_power_iterations=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.BatchNorm2d(oc, momentum=1., affine=True,
                               track_running_stats=True  # When this is true is called the "transductive setting"
                               )
            )
        else:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.BatchNorm2d(oc, momentum=1., affine=True,
                               track_running_stats=True  # When this is true is called the "transductive setting"
                               )
            )

    net2_x = nn.Sequential(
        conv_layer(1, hidden_size, spectral_norm),
        conv_layer(hidden_size, hidden_size, spectral_norm),
        conv_layer(hidden_size, hidden_size, spectral_norm),
        conv_layer(hidden_size, hidden_size, spectral_norm),
        nn.Flatten())
    net2_y = nn.Linear(hidden_size, n_classes)
    net2_z = nn.Linear(hidden_size, n_classes)
    return net2_x, net2_y, net2_z

def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            # m.bias.data = torch.ones(m.bias.data.size())
            # m.weight.data.zero_()
            m.bias.data.zero_()
    return net


if __name__ == '__main__':
    main()
