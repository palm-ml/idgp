# import
import argparse
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import os
import pickle
import time
import numpy as np
import random
from utils2.models import mlp_model, linear_model, save_model, LeNet
from utils2.utils_algo import accuracy_check
from utils2.earlystopping import EarlyStopping
from cifar_models import resnet, densenet
from utils2.cifar10 import load_cifar10
from utils2.cifar100 import load_cifar100
from utils2.fmnist import load_fmnist
from utils2.kmnist import load_kmnist
from utils2.mnist import load_mnist

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract_args():
    # main setting
    parser = argparse.ArgumentParser(
        prog='IDGP demo file.',
        usage='Demo with partial labels.',
        epilog='end',
        add_help=True
    )
    # optional args
    parser.add_argument('--lr', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('--wd', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--lr_g', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('--wd_g', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--bs', help='batch size', type=int, default=256)
    parser.add_argument('--ep', help='number of epochs', type=int, default=500)
    parser.add_argument('--mo', type=str, default="resnet")
    parser.add_argument('--ds', help='specify a dataset', type=str, choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'lost', 'MSRCv2', 'birdac', 'spd', 'LYN'])
    parser.add_argument('--rate', help='flipping probability', type=float, default=0.4)
    parser.add_argument('--warm_up', help='number of warm-up epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0, required=False)
    # loss paramters
    parser.add_argument('--alpha','-alpha', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--beta','-beta', type=float, default=1,help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--theta','-theta', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--gamma','-gamma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--delta','-delta', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--eta',  '-eta',   type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--T_1', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--T_2', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    # model args
    parser.add_argument('--lo', type=str, default="idgp")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args

# settings
args = extract_args()
print(args)
set_seed(args.seed)
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else 'cpu')

def weighted_crossentropy_f(f_outputs, weight, eps=1e-12):
    l = weight * torch.log(f_outputs)
    loss = (-torch.sum(l)) / l.size(0)
    
    return loss

def weighted_crossentropy_f_with_g(f_outputs, g_outputs, targets, eps=1e-12):
    weight = g_outputs.clone().detach() * targets
    weight[weight == 0] = 1.0
    logits1 = (1 - weight) / weight
    logits2 = weight.prod(dim=1, keepdim=True)
    weight = logits1 * logits2
    weight = weight * targets
    weight = weight / weight.sum(dim=1, keepdim=True)
    weight = weight.clone().detach()
    
    l = weight * torch.log(f_outputs)
    loss = (-torch.sum(l)) / l.size(0)
    
    return loss


def weighted_crossentropy_g_with_f(g_outputs, f_outputs, targets, eps=1e-12):
 
    weight = f_outputs.clone().detach() * targets
    weight = weight / weight.sum(dim=1, keepdim=True)
    l = weight * ( torch.log((1 - g_outputs) / (g_outputs)))
    l = weight * (torch.log(1 - g_outputs))
    loss = ( - torch.sum(l)) / ( l.size(0)) + \
        ( - torch.sum(targets * torch.log(g_outputs) + (1 - targets) * torch.log(1 - g_outputs))) / (l.size(0))
    
    return loss

def weighted_crossentropy_g(g_outputs, weight, eps=1e-12):
    l = weight * torch.log(g_outputs) + (1 - weight) * torch.log(1 - g_outputs)
    loss = ( - torch.sum(l)) / (l.size(0))

    return loss



def update_d(f_outputs, targets):
    new_d = f_outputs.clone().detach() * targets.clone().detach()
    new_d = new_d / new_d.sum(dim=1, keepdim=True)
    return new_d

def update_b(g_outputs, targets):
    new_b = g_outputs.clone().detach() * targets.clone().detach()
    
    return new_b




def warm_up(config, f, f_opt, train_loader, valid_loader, test_loader):
    d_array = train_loader.dataset.given_label_matrix.clone().detach().to(device)
    d_array = d_array / d_array.sum(dim=1, keepdim=True)
    print("Begin warm-up, warm up epoch {}".format(config.warm_up))

    for epoch in range(0, config.warm_up):
        f.train()
        for features, features_w, features_s, targets, trues, indexes in train_loader:
            features_o, features_w, features_s, targets, trues = map(lambda x: x.to(device), (features, features_w, features_s, targets, trues))
            f_logits_o, f_logits_w, f_logits_s                 = map(lambda x: f(x), (features_o, features_w, features_s))
            f_outputs_o, f_outputs_w, f_outputs_s = map(lambda x: F.softmax(x / config.T_1, dim=1), (f_logits_o, f_logits_w, f_logits_s))
            L_f_o, L_f_w, L_f_s = map(lambda x: weighted_crossentropy_f(x, d_array[indexes,:]), (f_outputs_o, f_outputs_w, f_outputs_s))
            L_f = L_f_o + L_f_w + L_f_s
            f_opt.zero_grad()
            L_f.backward()
            f_opt.step()
            d_array[indexes,:] = update_d(f_outputs_o, targets)
        f.eval()
        valid_acc = accuracy_check(loader=valid_loader, model=f, device=device)
        test_acc  = accuracy_check(loader=test_loader,  model=f, device=device)
        print("Warm_up Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(epoch+1, valid_acc, test_acc))
    return f, d_array

def warm_up_g(config, g, g_opt, d_array, train_loader, valid_loader, test_loader):
    b_array = train_loader.dataset.given_label_matrix.clone().detach().to(device)
    print("Begin warm-up, warm up epoch {}".format(config.warm_up))

    for epoch in range(0, config.warm_up):
        g.train()
        for features, features_w, features_s, targets, trues, indexes in train_loader:
            features_o, features_w, features_s, targets, trues   = map(lambda x: x.to(device), (features, features_w, features_s, targets, trues))
            g_logits_o, g_logits_w, g_logits_s                   = map(lambda x: g(x), (features_o, features_w, features_s))
            g_outputs_o, g_outputs_w, g_outputs_s                = map(lambda x: torch.sigmoid(x / config.T_2),    (g_logits_o, g_logits_w, g_logits_s))
            L_g_o, L_g_w, L_g_s = map(lambda x: weighted_crossentropy_g(x, b_array[indexes,:]),     (g_outputs_o, g_outputs_w, g_outputs_s))
            L_g = L_g_o + L_g_w + L_g_s

            g_opt.zero_grad()
            L_g.backward()
            g_opt.step()
            b_array[indexes,:] = update_b(g_outputs_o, targets)
        g.eval()
        
    return g, b_array

def accuracy_check_g(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, _, _, targets, labels, _ in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = torch.sigmoid(model(images))
            _, pred = torch.max(outputs.data, dim=1)
            total += (pred == labels).sum().item()
            num_samples += labels.size(0)

    return 100*(total/num_samples)

def noisy_output(outputs, d_array, targets):
    _, true_labels = torch.max(d_array * targets, dim=1)
    pseudo_matrix  = F.one_hot(true_labels, outputs.shape[1]).float().cuda().detach()
    return pseudo_matrix * (1 - outputs) + (1 - pseudo_matrix) * outputs


def train(config):
    if args.ds == "cifar10":
        train_loader, valid_loader, test_loader, dim, K = load_cifar10(args.ds, batch_size=args.bs, device=device)
    if args.ds == "cifar100":
        train_loader, valid_loader, test_loader, dim, K = load_cifar100(args.ds, batch_size=args.bs, device=device)
    if args.ds == "fmnist":
        train_loader, valid_loader, test_loader, dim, K = load_fmnist(args.ds, batch_size=args.bs, device=device)
    if args.ds == "kmnist":
        train_loader, valid_loader, test_loader, dim, K = load_kmnist(args.ds, batch_size=args.bs, device=device)
    if args.ds == "mnist":
        train_loader, valid_loader, test_loader, dim, K = load_mnist(args.ds, batch_size=args.bs, device=device)
    
    if args.mo == 'mlp':
        f = mlp_model(input_dim=dim, output_dim=K)
        g = mlp_model(input_dim=dim, output_dim=K)
    elif args.mo == 'linear':
        f = linear_model(input_dim=dim, output_dim=K)
        g = linear_model(input_dim=dim, output_dim=K)
    elif args.mo == 'lenet':
        f = LeNet(out_dim=K)
        g = LeNet(out_dim=K)
    elif args.mo == 'densenet':
        f = densenet(num_classes=K)
        g = densenet(num_classes=K)
    elif args.mo == 'resnet':
        f = resnet(depth=32, num_classes=K)
        g = resnet(depth=32, num_classes=K)
    set_seed(config.seed)
    
    f, g = map(lambda x: x.to(device), (f, g))



    # training
    f_opt = torch.optim.SGD(list(f.parameters()), lr=config.lr,   weight_decay=config.wd,   momentum=0.9)
    g_opt = torch.optim.SGD(list(g.parameters()), lr=config.lr_g, weight_decay=config.wd_g, momentum=0.9)
    consistency_criterion_f = nn.KLDivLoss(reduction='batchmean').cuda()
    consistency_criterion_g = nn.KLDivLoss(reduction='batchmean').cuda()
    # warm up
    f, d_array = warm_up(config, f, f_opt, train_loader, valid_loader, test_loader)
    g, b_array = warm_up_g(config, g, g_opt, d_array, train_loader, valid_loader, test_loader)
    
    save_path = "checkpoints/{}/{}/".format(args.ds, args.lo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    early = EarlyStopping(patience=50, path=os.path.join(save_path, "{}_lo={}_seed={}.pt".format(args.ds, args.lo, args.seed)))

    best_val, best_test, best_epoch = -1, -1, -1

    for epoch in range(0, config.ep):
        f.train()
        g.train()
        for features, features_w, features_s, targets, trues, indexes in train_loader:
            features_o, features_w, features_s, targets, trues = map(lambda x: x.to(device), (features, features_w, features_s, targets, trues))
            f_logits_o, f_logits_w, f_logits_s                 = map(lambda x: f(x), (features_o, features_w, features_s))
            g_logits_o, g_logits_w, g_logits_s                 = map(lambda x: g(x), (features_o, features_w, features_s))

            f_outputs_o, f_outputs_w, f_outputs_s = map(lambda x: F.softmax(x / config.T_1, dim=1), (f_logits_o, f_logits_w, f_logits_s))
            g_outputs_o, g_outputs_w, g_outputs_s = map(lambda x: torch.sigmoid(x / config.T_2),    (g_logits_o, g_logits_w, g_logits_s))

            L_f_o, L_f_w, L_f_s = map(lambda x: weighted_crossentropy_f(x, d_array[indexes,:]), (f_outputs_o, f_outputs_w, f_outputs_s))
            L_f = L_f_o + L_f_w + L_f_s
            L_f_g_o, L_f_g_w, L_f_g_s = map(lambda x: weighted_crossentropy_f_with_g(*x, targets), \
                                            zip((f_outputs_o, f_outputs_w, f_outputs_s), map(lambda x: noisy_output(x, d_array[indexes, :], targets), (g_outputs_o, g_outputs_w, g_outputs_s))))
            L_f_g = L_f_g_o + L_f_g_w + L_f_g_s
            
            L_g_o, L_g_w, L_g_s = map(lambda x: weighted_crossentropy_g(x, b_array[indexes,:]), (g_outputs_o, g_outputs_w, g_outputs_s))
            L_g = L_g_o + L_g_w + L_g_s

            
            L_g_f_o, L_g_f_w, L_g_f_s = map(lambda x: weighted_crossentropy_g_with_f(*x, targets), \
                                            zip((g_outputs_o, g_outputs_w, g_outputs_s), (f_outputs_o, f_outputs_w, f_outputs_s)))
            L_g_f = L_g_f_o + L_g_f_w + L_g_f_s

            f_outputs_log_o, f_outputs_log_w, f_outputs_log_s = map(lambda x: torch.log_softmax(x, dim=-1), (f_logits_o, f_logits_w, f_logits_s))
            f_consist_loss0, f_consist_loss1, f_consist_loss2 = map(lambda x: consistency_criterion_f(x, d_array[indexes,:]), (f_outputs_log_o, f_outputs_log_w, f_outputs_log_s))
            f_consist_loss = f_consist_loss0 + f_consist_loss1 + f_consist_loss2
            g_outputs_log_o, g_outputs_log_w, g_outputs_log_s = map(lambda x: nn.LogSigmoid()(x), (g_logits_o, g_logits_w, g_logits_s))
            g_consist_loss0, g_consist_loss1, g_consist_loss2 = map(lambda x: consistency_criterion_g(x, b_array[indexes,:]), (g_outputs_log_o, g_outputs_log_w, g_outputs_log_s))
            g_consist_loss = g_consist_loss0 + g_consist_loss1 + g_consist_loss2
            lam = min(epoch / 100, 1)

            L_F = config.alpha * L_f + config.beta  * L_f_g + lam * config.delta * f_consist_loss
            L_G = config.theta * L_g + config.gamma * L_g_f + lam * config.eta   * g_consist_loss
            f_opt.zero_grad()
            L_F.backward()
            f_opt.step()
            g_opt.zero_grad()
            L_G.backward()
            g_opt.step()
            d_array[indexes,:] = update_d(f_outputs_o, targets)
            b_array[indexes,:] = update_b(g_outputs_o, targets)

        f.eval()
        g.eval()
        valid_acc = accuracy_check(loader=valid_loader, model=f, device=device)
        test_acc  = accuracy_check(loader=test_loader,  model=f, device=device)
        print("Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(epoch+1, valid_acc, test_acc))
        
        if epoch >= 1:
            early(valid_acc, f, epoch)
        if early.early_stop:
            break
        if valid_acc > best_val:
            best_val = valid_acc
            best_epoch = epoch
            best_test = test_acc
    print("Best Epoch {:>3d}, Best valid acc: {:.2f}, test acc: {:.2f}. ".format(best_epoch, best_val, best_test))
    nni.report_final_result(best_test)



if __name__ == "__main__":
    
    train(args)
