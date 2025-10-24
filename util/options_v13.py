# util/options_v13.py
import argparse

def args_parser(return_parser=False): # (保持 v11 GPU 修复方案)
    parser = argparse.ArgumentParser()
    
    # === FedAvg 核心参数 ===
    parser.add_argument('--rounds', type=int, default=900, help="rounds of training (T)")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs (E)")
    parser.add_argument('--frac', type=float, default=0.1, help="fraction of clients to select (C)")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size (B)")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")

    # === 实验环境参数 ===
    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=2, help="GPU ID, default: 0")
    parser.add_argument('--seed', type=int, default=13, help="random seed")
    
    # --- 数据分布与噪声 ---
    parser.add_argument('--iid', type=bool, default=False, help="i.i.d. (True) or non-i.i.d. (False)")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=1)
  
    # --- 噪声参数 ---
    parser.add_argument('--level_n_system', type=float, default=0.6, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")

    # === v10/v11/v12 的参数 (v13 继承) ===
    parser.add_argument('--base_threshold', type=float, default=0.8, help="Base cosine similarity threshold (v9) or CLIP confidence threshold (v11/v12/v13).")
    parser.add_argument('--aggregation_alpha', type=float, default=0.7, help="Alpha for balancing data size and quality score in server aggregation (0 to 1).")
    parser.add_argument('--lambda_distill', type=float, default=1.0, help="Weight for the Global Knowledge Distillation loss (L_Distill).")
    parser.add_argument('--temp_distill', type=float, default=2.0, help="Temperature for L_Distill.")

    # === v11/v12/v13: CLIP 噪声预言机参数 ===
    parser.add_argument('--clip_model_name', type=str, default="ViT-B/32", help="CLIP model for noise oracle.")
    parser.add_argument('--clip_corr_thresh', type=float, default=0.1, help="CLIP confidence threshold for label correction.")
    


    if return_parser:
        return parser 

    return parser.parse_args()