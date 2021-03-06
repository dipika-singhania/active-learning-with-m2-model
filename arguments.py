import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size used for training and testing')
    parser.add_argument('--train_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--data_path', type=str, default='/mnt/data/captioning_dataset/active_learning/',
                        help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1,
                        help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=1,
                        help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1,
                        help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='/mnt/data/captioning_dataset/active_learning/results_debug/',
                        help='Path to where the output log will be')
    parser.add_argument('--test_acc_only', action='store_true', help="Pick up model and test accuracy on test data")
    parser.add_argument('--resume', action='store_true', help="Pick up model and test accuracy on test data")
    parser.add_argument('--start_resume', type=int, default=1, help="How much percentage data to be used for resuming")
    parser.add_argument('--lr_vae', type=float, default=3e-4, help="Learning rate of semi-supervised M2 model")
    parser.add_argument('--lr_ad', type=float, default=3e-4, help="Learning rate of adversarial loss")
    parser.add_argument('--find_lr_ad', action='store_true', help="Find the learning rate for discrimninator data")
    parser.add_argument('--find_lr_vae', action='store_true', help="Find the learning rate for discrimninator data")
    parser.add_argument('--tensorboard', action='store_true', help="If you want to visualize tensorboard logs")
    parser.add_argument('--visualization', action='store_true', help="Reconstruct some images from vae to vizualize")
    parser.add_argument('--m1_model', action='store_true', help='M1 model of Vaal paper')
    parser.add_argument('--log_file', type=str, help="Log file to save the logs into")
    parser.add_argument('--random_sampling', action='store_true', help="Random sampling of new samples")
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    return args
