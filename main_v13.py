# main_v13.py
import os
import argparse
import copy
import numpy as np
import random
import time

# --- (v11 GPU 修复方案) ---
from util.options_v13 import args_parser # 1. 导入 v13
parser = args_parser(return_parser=True) # 2. 获取 v13 parser
args_pre, _ = parser.parse_known_args() # 3. 预解析
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_pre.gpu) # 4. 设置环境变量
print(f"[Main Init] Set CUDA_VISIBLE_DEVICES='{args_pre.gpu}'")
# --- (修复结束) ---

import torch 
import torch.nn as nn

# 5. 导入 v13 组件
from util.local_training_v13 import LocalUpdate, globaltest
from util.fedavg import FedAvg, FedAvgWeighted
from util.util_v0 import add_noise
from util.dataset import get_dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Version 13: Hyper-Streamlined (v12 - L_PKD)
- Based on v12 (CLIP Oracle + Prompt Ensembling + L_Distill).
- v13 Simplification:
  - Removed L_PKD (v9's prototype loss) and all associated code.
  - This significantly speeds up Phase A (no ResNet feature extraction needed)
    and simplifies the Phase B loss calculation.
- Final Loss: L_CE (CLIP corrected) + L_Distill (Global model)
- This version retains the core innovations (CLIP Oracle + Global Distill)
  while being much faster than v11/v12.
"""

if __name__ == '__main__':

    start_time = time.time()
    
    args = parser.parse_args()
    
    print(f"[Main Start] args.gpu = {args.gpu}")
    print(f"[Main Start] os.environ['CUDA_VISIBLE_DEVICES'] = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Main Start] PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Main Start] PyTorch device count: {torch.cuda.device_count()}")
        print(f"[Main Start] PyTorch current device: {torch.cuda.current_device()}")
        print(f"[Main Start] PyTorch device name: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = "./record/"

    dataset_train, dataset_test, dict_users = get_dataset(args)
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    if not os.path.exists(rootpath + 'txtsave/'): os.makedirs(rootpath + 'txtsave/')
    
    # 7. 更新日志路径 (v13)
    txtpath = rootpath + 'txtsave/V13_Streamlined_CLIP_%s_%s_NL_%.1f_LB_%.1f_Rnd_%d_E_%d_Frac_%.2f_LR_%.3f_Seed_%d' % (
        args.clip_model_name.replace('/', '-'), args.dataset, args.level_n_system, args.level_n_lowerb, 
        args.rounds, args.local_ep, args.frac, args.lr, args.seed)

    if args.iid: txtpath += "_IID"
    else: txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    f_acc = open(txtpath + '_acc.txt', 'a')

    f_acc.write("="*50 + "\n")
    f_acc.write("Training Parameters (V13 - Streamlined CLIP Oracle + L_Distill):\n")
    f_acc.write(str(args) + "\n")
    f_acc.write("="*50 + "\n")
    f_acc.flush()
    
    netglob = build_model(args)
    
    max_accuracy = 0.0

    # ============================ Training Loop (与 v12 相同) ============================
    print("\n" + "="*25 + " Stage: FedAvg with V13 Local Loss (Streamlined) " + "="*25, flush=True)
    final_accuracies = []
    
    m = max(int(args.frac * args.num_users), 1)

    for rnd in range(args.rounds):
        print(f"\n--- FedAvg Round: {rnd+1}/{args.rounds} ---", flush=True)
        w_locals, local_losses, quality_scores = [], [], []
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"Selected clients for this round: {idxs_users}", flush=True)
        
        for idx in idxs_users:
            print(f"--> Training client {idx}...")
            # 使用 v13 的 LocalUpdate
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            w, loss, quality_score = local.update_weights(
                net=copy.deepcopy(netglob).to(args.device), 
                seed=args.seed,
                w_g=netglob.state_dict(),
                epoch=args.local_ep
            )
            w_locals.append({'id': idx, 'w': copy.deepcopy(w)})
            local_losses.append(copy.deepcopy(loss))
            quality_scores.append(quality_score)
            print(f"  [Client {idx}] Final CLIP-based Quality Score: {quality_score:.4f}")

        # === 质量感知聚合 (v13 继承) ===
        client_data_sizes = np.array([len(dict_users[d['id']]) for d in w_locals])
        weights_by_size = client_data_sizes / np.sum(client_data_sizes)
        
        quality_scores = np.array(quality_scores)
        if np.sum(quality_scores) == 0:
            print("Warning: All quality scores are zero, falling back to size-based weights.")
            weights_by_quality = weights_by_size
        else:
            quality_scores_exp = np.exp(quality_scores - np.max(quality_scores))
            weights_by_quality = quality_scores_exp / np.sum(quality_scores_exp)
        
        alpha = args.aggregation_alpha
        final_weights = (1 - alpha) * weights_by_size + alpha * weights_by_quality
        final_weights = final_weights / np.sum(final_weights)

        print(f"Final combined aggregation weights (alpha={alpha}): {np.round(final_weights, 3)}")

        w_glob = FedAvgWeighted([d['w'] for d in w_locals], final_weights)
        netglob.load_state_dict(copy.deepcopy(w_glob))

        # --- 评估 ---
        acc_s3 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        final_accuracies.append(acc_s3)
        max_accuracy = max(max_accuracy, acc_s3)
        
        print(f"Test Accuracy after round {rnd+1}: {acc_s3:.4f}", flush=True)
        f_acc.write(f"round {rnd}, test acc  {acc_s3:.4f} \n"); f_acc.flush()

    # ============================ Final Result Output (与 v11/v12 相同) ============================
    print("\n" + "="*30 + " Final Results " + "="*30, flush=True)
    if len(final_accuracies) >= 10:
        last_10_accuracies = final_accuracies[-10:]
        mean_acc = np.mean(last_10_accuracies)
        var_acc = np.var(last_10_accuracies)
        print(f"Mean of last 10 rounds test accuracy: {mean_acc:.4f}", flush=True)
        print(f"Variance of last 10 rounds test accuracy: {var_acc:.6f}", flush=True)
        f_acc.write(f"\nMean of last 10 rounds test accuracy: {mean_acc:.4f}\n")
        f_acc.write(f"Variance of last 10 rounds test accuracy: {var_acc:.6f}\n")
    elif len(final_accuracies) > 0:
        mean_acc = np.mean(final_accuracies)
        var_acc = np.var(final_accuracies)
        print(f"Mean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}", flush=True)
        print(f"Variance of final {len(final_accuracies)} rounds test accuracy: {var_acc:.6f}", flush=True)
        f_acc.write(f"\nMean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}\n")
        f_acc.write(f"Variance of final {len(final_accuracies)} rounds test accuracy: {var_acc:.6f}\n")
    
    print(f"\nMaximum test accuracy achieved: {max_accuracy:.4f}", flush=True)
    f_acc.write(f"\nMaximum test accuracy achieved: {max_accuracy:.4f}\n")

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal Training Time: {hours}h {minutes}m {seconds}s", flush=True)
    f_acc.write(f"\nTotal Training Time: {hours}h {minutes}m {seconds}s\n")


    f_acc.close()
    torch.cuda.empty_cache()
    print("\nTraining Finished!", flush=True)