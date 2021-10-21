import torch
import random
from torchvision.datasets import MNIST,FashionMNIST,CIFAR10
import torch.nn.functional as F
import copy
import numpy as np
import time
import csv
import argparse
import higher
import numpy
parser = argparse.ArgumentParser(description='Data Hyper-Cleaning')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--z_loop', type=int, default=100)
parser.add_argument('--y_loop', type=int, default=100)
parser.add_argument('--x_loop', type=int, default=3000)
parser.add_argument('--z_L2_reg', type=float, default=0.01)
parser.add_argument('--y_L2_reg', type=float, default=0.001)
parser.add_argument('--y_ln_reg', type=float, default=0.1)
parser.add_argument('--reg_decay', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--learn_h', type=bool, default=False)
parser.add_argument('--x_lr', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--y_lr', type=float, default=0.03)
parser.add_argument('--z_lr', type=float, default=0.01)
parser.add_argument('--x_lr_decay_rate', type=float, default=0.1)
parser.add_argument('--x_lr_decay_patience', type=int, default=1)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', type=str, default='nonconvex')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nestrov', type=bool, default=False)
parser.add_argument('--Notes', type=str, default=' ', metavar='N', help='Additional Notes')
args = parser.parse_args()
if args.dataset=='MNIST':
    dataset = MNIST(root="./data/mnist", train=True, download=True)
elif args.dataset=='FashionMNIST':
    dataset = FashionMNIST(root="./data/fashionmnist", train=True, download=True)
elif args.dataset=='CIFAR10':
    dataset=CIFAR10(root="./data/cifar10", train=True, download=True)
    dataset.targets=torch.from_numpy(numpy.array(dataset.targets)).long()
    dataset.data=torch.from_numpy(dataset.data)
print(args)

cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class Dataset:
    def __init__(self, data, target, polluted=False, rho=0.0):
        self.data = data.float() / torch.max(data)
        print(list(target.shape))
        if not polluted:
            self.clean_target = target
            self.dirty_target = None
            self.clean = np.ones(list(target.shape)[0])
        else:
            self.clean_target = None
            self.dirty_target = target
            self.clean = np.zeros(list(target.shape)[0])
        self.polluted = polluted
        self.rho = rho
        self.set = set(target.numpy().tolist())

    def data_polluting(self, rho):
        assert self.polluted == False and self.dirty_target is None
        number = self.data.shape[0]
        number_list = list(range(number))
        random.shuffle(number_list)
        self.dirty_target = copy.deepcopy(self.clean_target)
        for i in number_list[:int(rho * number)]:
            dirty_set = copy.deepcopy(self.set)
            dirty_set.remove(int(self.clean_target[i]))
            self.dirty_target[i] = random.randint(0, len(dirty_set))
            self.clean[i] = 0
        self.polluted = True
        self.rho = rho

    def data_flatten(self):
        try :
            self.data = self.data.view(self.data.shape[0], self.data.shape[1] * self.data.shape[2])
        except BaseException:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1] * self.data.shape[2] * self.data.shape[3])

    def to_cuda(self):
        self.data = self.data.to(device)
        if self.clean_target is not None:
            self.clean_target = self.clean_target.to(device)
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.to(device)


def data_splitting(dataset, tr, val, test):
    assert tr + val + test <= 1.0 or tr > 1
    number = dataset.targets.shape[0]
    number_list = list(range(number))
    random.shuffle(number_list)
    if tr < 1:
        tr_number = tr * number
        val_number = val * number
        test_number = test * number
    else:
        tr_number = tr
        val_number = val
        test_number = test

    train_data = Dataset(dataset.data[number_list[:int(tr_number)], :, :],
                         dataset.targets[number_list[:int(tr_number)]])
    val_data = Dataset(dataset.data[number_list[int(tr_number):int(tr_number + val_number)], :, :],
                       dataset.targets[number_list[int(tr_number):int(tr_number + val_number)]])
    test_data = Dataset(
        dataset.data[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)], :, :],
        dataset.targets[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)]])
    return train_data, val_data, test_data

def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def loss_L1(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 1)
    return loss

def loss_Lq(parameters,q,epi):
    loss = 0
    for w in parameters:
        loss += (torch.norm(w,2)+torch.norm(epi*torch.ones_like(w),2))**(q/2)
    return loss


def accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def Binarization(x):
    x_bi = np.zeros_like(x)
    for i in range(x.shape[0]):
        # print(x[i])
        x_bi[i] = 1 if x[i] >= 0 else 0
    return x_bi


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to(device).requires_grad_(True))

    def forward(self, y):
        # if torch.norm(torch.sigmoid(self.x), 1) > 2500:
        #     y = torch.sigmoid(self.x) / torch.norm(torch.sigmoid(self.x), 1) * 2500 * y
        # else:
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y


def copy_parameter(y, z):
    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    return y


tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(args.pollute_rate)
tr.data_flatten()
val.data_flatten()
test.data_flatten()
tr.to_cuda()
val.to_cuda()
test.to_cuda()
log_path = "{}_network{}_outerLoop{}_inner_loop{}_outer_lr{}_inner_lr{}_Notes{}.csv".format(args.dataset,args.convex,args.x_loop,args.y_loop,args.x_lr,args.y_lr, args.Notes)
with open(log_path, 'a', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([args])
    csv_writer.writerow(['IAPTT-GM', 'acc', 'F1 score','total_time_iter','forward_time','backward_time','val loss','p','r','pmax_list'])
d = 28 ** 2
n = 10
z_loop = args.z_loop
y_loop = args.y_loop
x_loop = args.x_loop
alpha=args.alpha
# x = torch.zeros(tr.data.shape[0]).to("cuda").requires_grad_(True)
x = Net_x(tr)
if args.dataset=='MNIST' or args.dataset=='FashionMNIST':
    d = 28 ** 2
    n = 10
    if args.convex=='convex':
        y = torch.nn.Sequential(torch.nn.Linear(d, n)).to(device)
# non-convex
    else:
        y = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)

elif args.dataset=='CIFAR10':
    d=3*32**2
    n=10
    y = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)

x_opt = torch.optim.Adam(x.parameters(), lr=args.x_lr)
if args.nestrov:
    y_opt = torch.optim.SGD(y.parameters(), lr=args.y_lr,momentum=args.momentum, nesterov=True)
else:
    y_opt=torch.optim.SGD(y.parameters(), lr=args.y_lr)
acc_history = []
clean_acc_history = []
print(sum(p.numel() for p in y.parameters()))
Lq=1/sum(p.numel() for p in y.parameters())
loss_x_l = 0
F1_score_last=0
lr_decay_rate = 1
reg_decay_rate = 1
dc=0
for x_itr in range(x_loop):
    # x_opt.param_groups[0]['lr']=args.x_lr/(1+(1e-5)*(x_itr+1))

    for xp in x.parameters():
        if xp.grad is not None:
            xp.grad = None
    for yp in y.parameters():
        if yp.grad is not None:
            yp.grad = None
    # print(loss_L1(y.parameters()).item())
    start_time_task = time.time()
    forward_time, backward_time = 0, 0

    F_list = []
    with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
        forward_time_task = time.time()
        for y_idx in range(y_loop):
            out_f = fmodel(tr.data)
            # print(loss_Lq(fmodel.parameters(),0.5,0.1))
            loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
            diffopt.step(loss_f)
            F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
        pmax = F_list.index(max(F_list))
        print(pmax)
        out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))
        forward_time_task = time.time() - forward_time_task
        forward_time += forward_time_task
        backward_time_task = time.time()
        #out_F = fmodel(val.data, params=fmodel.parameters())
        loss_F = F.cross_entropy(out_F, val.clean_target)
        grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
        grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
        for p, xp in zip(grad_x, x.parameters()):
            if xp.grad == None:
                xp.grad = p
            else:
                xp.grad += p
        for p, yp in zip(grad_z, y.parameters()):
            if yp.grad == None:
                yp.grad = p
            else:
                yp.grad += p

        backward_time_task = time.time() - backward_time_task
        backward_time += backward_time_task
        total_time_iter=time.time() - start_time_task
        y_opt.step()
        x_opt.step()

        if x_itr % 10 == 0:
            with torch.no_grad():
                out = y(test.data)
                acc = accuary(out, test.clean_target)
                x_bi = Binarization(x.x.cpu().numpy())
                clean = x_bi * tr.clean
                acc_history.append(acc)
                p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                r = clean.mean() / (1. - tr.rho)
                F1_score = 2 * p * r / (p + r + 1e-8)
                dc=0
                if F1_score_last>F1_score:
                    dc=1
                F1_score_last=F1_score

                # x_opt_l.step(F1_score)
                # y_opt_l.step(acc)
                # z_opt_l.step(acc)
                loss_F = F.cross_entropy(out, test.clean_target)
                print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax{}'.format(x_itr,
                                                                                     100 * accuary(out,
                                                                                                   test.clean_target),
                                                                                     100 * p, 100 * r, 100 * F1_score,loss_F,pmax))
                with open(log_path, 'a', encoding='utf-8', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax])

# with torch.no_grad():
#     out = y(test.data)
#     print(100*accuary(out,test.clean_target))
