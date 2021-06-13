import  torch, os
import  numpy as np
from    fer2013NShot import FerNShot
import  argparse
import time
from    meta import Meta

def main(args):

    seed=int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    print(args)
    config =[
        ('conv2d',[64,3,3,3,1,1]),
        ('bn',[64]),
        ('relu',[True]),
        ('conv2d',[64,64,3,3,1,1]),
        ('bn',[64]),
        ('relu',[True]),
        ('max_pool2d',[2,2,0]),
        ('conv2d',[128,64,3,3,1,1]),
        ('bn',[128]),
        ('relu',[True]),
        ('conv2d',[128,128,3,3,1,1]),
        ('bn',[128]),
        ('relu',[True]),
        ('max_pool2d',[2,2,0]),
        ('conv2d',[256,128,3,3,1,1]),
        ('bn',[256]),
        ('relu',[True]),
        ('conv2d',[256,256,3,3,1,1]),
        ('bn',[256]),
        ('relu',[True]),
        ('conv2d',[256,256,3,3,1,1]),
        ('bn',[256]),
        ('relu',[True]),
        ('conv2d',[256,256,3,3,1,1]),
        ('bn',[256]),
        ('relu',[True]),
        ('max_pool2d',[2,2,0]),
        ('conv2d',[512,256,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('conv2d',[512,512,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('conv2d',[512,512,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('conv2d',[512,512,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('max_pool2d',[2,2,0]),
        ('conv2d',[512,512,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('conv2d',[512,512,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('conv2d',[512,512,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('conv2d',[512,512,3,3,1,1]),
        ('bn',[512]),
        ('relu',[True]),
        ('max_pool2d',[2,2,0]),
        ('avg_pool2d',[1,1,0]),
        ('flatten', []),
        ('linear', [args.n_way,512])
    ]

    device = args.device
    maml = Meta(args,config).to(device)
    if not args.reload_model is None:
        maml.load_state_dict(torch.load(args.reload_model,map_location=device))
        maml.train()
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = FerNShot('fer',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    accs = []
    lccs = []
    for testid in range(100//args.task_num):
                # test
        x_spt, y_spt, x_qry, y_qry = db_train.next('test')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            test_acc,losses= maml.finetunning_for_eval(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            accs.append( test_acc )
            lccs.append(losses)


            # [b, update_step+1]
    accs = np.array(accs).mean(axis=0).astype(np.float16)
    lccs = np.array(lccs).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)
    print('Test loss:', lccs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=7)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--model', type=str, help='vgg19 or resnet', default='vgg19')
    argparser.add_argument('--device', type=str, help='cpu or cuda', default='cuda:1')
    argparser.add_argument('--reload_model', type=str,help='reload maml model',default=None)

    args = argparser.parse_args()

    main(args)
