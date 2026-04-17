import argparse


def create_parser():
    """Create argument parser with organized groups"""
    
    parser = argparse.ArgumentParser(
        description="Stage 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_name', type=str, 
                            default='deepvk/RuModernBERT-base',
                            help='Pre-trained model to use')
    model_group.add_argument('--tokenizer', type=str, 
                            default=None,
                            help='The model tokenizer to use, based on the model name')
    model_group.add_argument('--embedding_dim', type=int, default=768,
                            help='Dimension of embeddings') 
    model_group.add_argument('--dropout_rate', type=int, default=0.1,
                            help='Dropout rate for training')  
    model_group.add_argument('--run_name', type=str,
                            help='The version of the run')
    model_group.add_argument("--test_only", action='store_true', required=False, 
                            help="Determines whether you want only to load the fine-tuned model for testing")
    model_group.add_argument("--resume_from_checkpoint", type=str, default=None, required=False, 
                            help="A path to a partially or fully fine-tuned model, only used when the training was disrupted or you want only test")                                     
    model_group.add_argument("--seed", type=int, default=42, required=False, 
                            help="A random seed for reproducibility")
    model_group.add_argument("--pure_baseline", action='store_true', required=False, 
                            help="IF you want to run pure model baseline")                                   
    model_group.add_argument('--pre_train', action='store_true', default=False,
                             help='Enables the pre-training phase') 
                      
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_path', type=str, required=False, help='Path to skills JSON file (training data)')
    data_group.add_argument('--index_path', type=str, required=False, help='Path to skills JSON file (training data)')
    data_group.add_argument('--result_path', type=str, required=False, help='Path to skills JSON file (training data)')
    data_group.add_argument('--datasets', type=list, default=['hr'], required=False, help='Path to skills JSON file (training data)')
    
    data_group.add_argument('--top_k', type=int, default=5, required=False,
                           help='The number of top k most similar items to retrieve')    
    data_group.add_argument('--k_values', type=int, default=[1, 2, 5, 8, 10, 60, 100], required=False,
                           help='The number of top k most similar items to retrieve')   

    data_group.add_argument('--extension', type=str, default='flat', help='Whether to truncate sequences')
    
    data_group.add_argument('--max_length_query', type=int, default=512,
                           help='Maximum length for sentence tokenization, consider the augmentation strategy')
    data_group.add_argument('--max_length_passage', type=int, default=512,
                           help='Maximum length for sentence tokenization, consider the augmentation strategy')
        
    # Negative sampling arguments
    neg_group = parser.add_argument_group('Negative Sampling Configuration')
    neg_group.add_argument('--hard_negative_strategy', type=str, default='in_batch',
                          choices=['in_batch'],
                          help='Negative sampling strategy')
    neg_group.add_argument('--num_negatives', type=int, default=3,
                          help='Number of hard negatives to sample')

    loss_group = parser.add_argument_group('Loss Function Configuration')
    loss_group.add_argument('--loss_type', type=str, default='ntxent',
                           choices=['ntxent', 'margin'],
                           help='Loss function type')
    loss_group.add_argument('--temperature_main', type=float, default=0.05,
                           help='Temperature for NT-Xent loss (main phase)')   
    loss_group.add_argument('--temperature_pre_train', type=float, default=0.03,
                           help='Temperature for NT-Xent loss (pre-train phase)')
    loss_group.add_argument('--margin', type=float, default=0.2,
                           help='Margin for margin loss')
    loss_group.add_argument('--reduction', type=str, default='mean',
                           choices=['mean', 'sum'],
                           help='Loss reduction method')
    
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch_size', type=int, default=8,
                            help='Training batch size')
    train_group.add_argument('--log_with', type=str, default='tensorboard',
                            help='The logging strategy. Model configured to use tensorboard only.')    
    train_group.add_argument('--learning_rate_main', type=float, default=2e-5,
                            help='Learning rate for main training phase')
    train_group.add_argument('--learning_rate_pre_train', type=float, default=3e-5,
                            help='Learning rate for pre_train phase')
    train_group.add_argument('--num_epochs', type=int, default=4,
                            help='Number of training epochs')
    train_group.add_argument('--weight_decay', type=float, default=0.01,
                            help='Weight Decay')    
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=16,
                            help='Gradient accumulation steps')
    train_group.add_argument('--early_stopping_patience', type=int, default=2,
                            help='Determines the number of epoch after which if the performance is not improved, training must stop')
    train_group.add_argument('--fp16', action="store_true", required=False,
                            help='A boolean to determine whether to use half-precision')
    train_group.add_argument('--run_initial_eval', type=bool, default=False,
                            help='A boolean to determine whether to run evaluation using a vanilla model (before training starts)')
    train_group.add_argument("--save_baseline", action="store_true",
                            help="Save the initial (pre-training) model as baseline_initial")   
    train_group.add_argument("--log_freq", type=int, default=1,
                            help="The frequency of parameters updates as logged in logger")        
    train_group.add_argument("--max_grad_norm", type=int, default=1, required=False, 
                            help="Gradient Clipping")
    
    train_group.add_argument("--warmup_percent", type=float, default=0.05, required=False, help="The warmup ratio")  
    train_group.add_argument("--use_cosine_schedule", type=bool, default=True, required=False, help="Whether to use cosine schedule or not")

    pt_group = parser.add_argument_group('Pre-train Stage Training Configuration')
    pt_group.add_argument('--data_path_pt', type=str, required=False, help='Path to the Pre-train data')
    
    collator_group = parser.add_argument_group('Collator Configuration')
    collator_group.add_argument('--truncation', type=bool, default=True,
                               help='Whether to truncate sequences')

    collator_group.add_argument('--ranking_margin', type=float, default=0.2,
                               help='Margin for relative ranking')
    
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--log_dir', type=str, default='./logs',
                             help='Directory for outputs')
    output_group.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                             help='Directory for checkpoints')
    output_group.add_argument('--save_freq', type=int, default=1,
                             help='Save checkpoint every N epochs')
    
    return parser
