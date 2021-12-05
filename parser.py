import argparse
import yaml 

def parse_args_train_multi():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(
        description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(
        description='Multilabel K-fold cross validation')

    parser.add_argument('--folder-name', type=str,
                        default='CVmulti', help='folder name')
    # Dataset / Model parameters
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--model', default='tv_resnet50', type=str, metavar='MODEL',
                        help='efficientnetv2_m or tv_resnet50')
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='freeze top layers')
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--eval-metric', type=str, default="acc",
                        help='evaluation metric = loss, acc')
    parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to root output folder (default: none, current dir)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--pretrain-num-classes', type=int, default=2, metavar='N',
                        help='number of label classes of the pretrain model (Model default if None)')
    
    # preprocessing options
    parser.add_argument('--transparent2white','-t2w', action='store_true', default=False,
                        help='creating white background for image with transparency ')
    parser.add_argument('--color2grayscale','-c2g', action='store_true', default=False,
                        help='convert all colour images to grayscale')
    parser.add_argument('--mean', type=list, nargs='+', default=[0.9852, 0.9852, 0.9852], metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=list, nargs='+', default=[0.1079, 0.1079, 0.1079], metavar='STD',
                        help='Override std deviation of of dataset')

    # augmentation options
    parser.add_argument('--aug', action='store_true', default=False,
                        help='augment all labels')

    # Formulated Imbalanced Dataset Sampler parameters
    parser.add_argument('--min-perct', type=float, default=0, metavar='N',
                        help='percentage threshold to be considered as minority label combination')
    parser.add_argument('--add-perct', type=float, default=0, metavar='N',
                        help='potential percentage of minority label combination to be added to dataset')
    parser.add_argument('--maxI', type=int, default=0, metavar='N',
                        help='maximum images that can be added per instance')



    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def parse_args_infer():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(
        description='Infer Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(
        description='Multilabel K-fold cross validation Infer')

    # Dataset / Model parameters
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--model', default='tv_resnet50', type=str, metavar='MODEL',
                        help='efficientnetv2_m or tv_resnet50')
    parser.add_argument('--checkpoint-dir', default='', type=str, metavar='PATH',
                        help='path to dir containing cross validation folders which in turn contain checkpoint')
    
    # preprocessing options
    parser.add_argument('--transparent2white','-t2w', action='store_true', default=False,
                        help='creating white background for image with transparency ')
    parser.add_argument('--color2grayscale','-c2g', action='store_true', default=False,
                        help='convert all colour images to grayscale')
    parser.add_argument('--mean', type=list, nargs='+', default=[0.9852, 0.9852, 0.9852], metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=list, nargs='+', default=[0.1079, 0.1079, 0.1079], metavar='STD',
                        help='Override std deviation of of dataset')

    parser.add_argument("--gradfile", type=list, nargs="+", default=[''], 
                      help="gradcam output for list of filenames e.g. ['N_24_5.png','N_12_6.png']")

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def parse_args_train_binary():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(
        description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(
        description='Binary K-fold cross validation')

    parser.add_argument('--folder-name', type=str,
                        default='CVbin', help='folder name')
    # Dataset / Model parameters
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--model', default='tv_resnet50', type=str, metavar='MODEL',
                        help='efficientnetv2_m or tv_resnet50')
    parser.add_argument('--freeze', action='store_true', default=True,
                        help='freeze top layers')
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--eval-metric', type=str, default="top1",
                        help='evaluation metric = loss, top1')
    parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to root output folder (default: none, current dir)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--pretrain-num-classes', type=int, default=2, metavar='N',
                        help='number of label classes of the pretrain model (Model default if None)')
    parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                    help='number of label classes of the pretrain model (Model default if None)')
    # preprocessing options
    parser.add_argument('--transparent2white','-t2w', action='store_true', default=False,
                        help='creating white background for image with transparency ')
    parser.add_argument('--color2grayscale','-c2g', action='store_true', default=False,
                        help='convert all colour images to grayscale')
    parser.add_argument('--mean', type=list, nargs='+', default=[0.9852, 0.9852, 0.9852], metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=list, nargs='+', default=[0.1079, 0.1079, 0.1079], metavar='STD',
                        help='Override std deviation of of dataset')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text