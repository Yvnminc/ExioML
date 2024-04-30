from train import DeepModel
import argparse

def set_args():
  """
  Set up needed arguments
  """
  parser = argparse.ArgumentParser(description="training script")
  parser.add_argument("--type", type = str, default= "pxp", choices = ['pxp', 'ixi'], help="Data type.")
  parser.add_argument('--data', type= str, default= "clean", choices = ['clean', 'LOOE'], help="Prepared data.")
  parser.add_argument('--batch_size', type=int, default=512, help="Batch size.")
  parser.add_argument('--max_epochs', type=int, default=30, help="Max epochs.")
  parser.add_argument('--n_trials', type=int, default=1, help="Tuning times.")
  parser.add_argument("--model", type=str, default="CEM", help="Model to tune.")
  
  
  args = parser.parse_args()
  return args

def print_options(args):
    """
    print the options
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


def main(args):
    
    cem_params = {
        "layers": ["256-128-64-32", "128-64-32-16" , "256-128-64", "128-64-32", "64-32-16"],
        "dropout": [0.0, 0.05, 0.1],
        "learning_rate": [0.01, 0.001],
        }
    
    gandalf_params = {
        "gflu_stages": [1,2,3,4,5],
        "gflu_dropout": [0.0, 0.05, 0.1],
        "gflu_feature_init_sparsity": [0.0, 0.05, 0.1],
        "learning_rate": [0.01, 0.001],
        }
    
    ftt_params = {
        "num_heads": [4, 8],
        "num_attn_blocks": [1, 2, 3, 4],
        "learning_rate": [0.01, 0.001],
        "ff_hidden_multiplier": [1, 2, 3, 4],
    }
        
    if args.model == "CEM":
        search_space = cem_params
    elif args.model == "Gandalf":
        search_space = gandalf_params
    elif args.model == "FTT":
        search_space = ftt_params
    
    DeepTrainer = DeepModel(args.type, args.data, batch_size= args.batch_size, max_epochs= args.max_epochs)

    result = DeepTrainer.tune(args.model, search_space= search_space, n_trials= args.n_trials)
    
    result.to_csv(f"results/{args.model}_{args.type}_{args.data}_tune.csv")

if __name__ == '__main__':
    # Set args
    args = set_args()
    print_options(args)
    main(args)