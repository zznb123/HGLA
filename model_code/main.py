import torch
from param_parser import parameter_parser

from trainer import Trainer
from utils import tab_printer
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
    args = parameter_parser()
    print(args)
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda")
    torch.manual_seed(args.seed)
    tab_printer(args)
    

    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
