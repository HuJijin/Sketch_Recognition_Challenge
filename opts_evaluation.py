import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings

    ## model info
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet50')
    parser.add_argument(
        '--evaluation_path',
        type=str,
        default='./evaluation')
    parser.add_argument(
        '--best',
        type=str,
        default='../baseline/checkpoint_resnet50_2022-05-06-13-31/best.pth')
    ## parameter for dataset
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)
    parser.add_argument(
        '--testset_path',
        type=str,
        default='/root/Data/SPG_png/test/seen/')
    parser.add_argument(
        '--file_path',
        type=str,
        default='../test_seen_label.txt'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=20)
    parser.add_argument(
        '--test_type',
        type=str,
        default='single'  # single, multi
        )
    parser.add_argument(
        '--gpus',
        type=str,
        default='0')
    parser.add_argument(
        '--debug',
        action='store_true')
    args = parser.parse_args()

    return args




