# util/local_training_v13.py
# (v13 精简版: 移除了 L_PKD 及其在 Phase A 的计算)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np 
import copy
import clip
from PIL import Image

# 导入 v13 的工具
from .util_v13 import (
    noise_detection_and_correction_clip
)
# 导入 v13 的 CLIP 管理器
from .clip_manager_v13 import get_class_names, get_text_prototypes

class DatasetSplit(Dataset):
    # (与 v11/v12 相同)
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        # (与 v12 相同)
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs = list(idxs)
        self.local_dataset = DatasetSplit(dataset, self.idxs) 

        print(f"[Client {self.idxs[0] if len(self.idxs)>0 else 'N/A'}] Loading CLIP Oracle: {args.clip_model_name} onto {args.device}")
        self.clip_model, self.clip_preprocess = clip.load(args.clip_model_name, device=args.device)
        self.clip_model.eval() 
        class_names = get_class_names(args)
        if class_names is None:
            raise ValueError(f"Dataset {args.dataset} not supported by clip_manager_v13.")
        self.clip_text_features = get_text_prototypes(self.clip_model, class_names)
        self.clip_text_features = self.clip_text_features.to(self.clip_model.dtype)
        print(f"[Client] CLIP Oracle initialized (persistent on {args.device}).")
        
    def update_weights(self, net, seed, w_g, epoch, mu=0):
        
        net_g = copy.deepcopy(net).to(self.args.device)
        net_g.load_state_dict(w_g)
        net_g.eval()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        
        # --- v13: 数据准备 (与 v12 相同) ---
        images_resnet = torch.stack([self.local_dataset[i][0] for i in range(len(self.local_dataset))])
        original_images_data = [self.local_dataset.dataset.data[i] for i in self.local_dataset.idxs]
        images_clip = torch.stack(
            [self.clip_preprocess(Image.fromarray(img) if isinstance(img, np.ndarray) else img) 
             for img in original_images_data]
        )
        original_targets = np.array([self.local_dataset.dataset.targets[i] for i in self.local_dataset.idxs])
        current_labels_tensor = torch.tensor(original_targets, dtype=torch.long)
        
        # === v13 精简: 阶段 A (仅运行 CLIP 修正) ===
        print(f"  [Client] Performing one-time label correction (Phase A is now 2x faster)...")
            
        # 1. v13: 使用 CLIP 预言机进行噪声检测和修正 (运行一次)
        corrected_labels_np, _, client_quality_score = noise_detection_and_correction_clip(
            self.clip_model, 
            self.clip_text_features, 
            images_clip, 
            current_labels_tensor.numpy(), 
            self.args
        )
        current_labels = torch.tensor(corrected_labels_np, dtype=torch.long)

        # 2. v13: (移除) 不再需要计算 ResNet 原型
        # (REMOVED) temp_resnet_dataset = ...
        # (REMOVED) features_resnet, _ = get_output(...)
        # (REMOVED) sub_prototypes = calculate_sub_prototypes(...)

        print(f"  [Client] Correction (Quality: {client_quality_score:.4f}) is now fixed for all {epoch} epochs.")
        # === v13 精简: 阶段 A 结束 ===
        
        train_dataset = TensorDataset(images_resnet, current_labels)

        for iter_ in range(epoch):
            # --- 步骤B: v13 损失函数训练 (L_CE + L_Distill) ---
            net.train()
            ldr_train = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
            batch_loss = []

            for batch_idx, (batch_images, batch_labels) in enumerate(ldr_train):
                batch_images, batch_labels = batch_images.to(self.args.device), batch_labels.to(self.args.device)
                net.zero_grad()
                
                # 我们只需要 logits
                # (ResNet forward 默认也会返回 features, 但我们不使用它)
                logits = net(batch_images, latent_output=False) 
                
                with torch.no_grad():
                    teacher_logits = net_g(batch_images, latent_output=False)

                # 1. CE 损失 (使用固定的修正标签)
                loss_ce = self.loss_func(logits, batch_labels)
                
                # 2. L_PKD (REMOVED)
                
                # 3. L_Distill (v10 核心)
                loss_distill = F.kl_div(
                    F.log_softmax(logits / self.args.temp_distill, dim=1),
                    F.softmax(teacher_logits / self.args.temp_distill, dim=1),
                    reduction='batchmean'
                ) * (self.args.temp_distill ** 2)
                
                # 4. 组合损失 (已精简)
                loss = loss_ce + self.args.lambda_distill * loss_distill
                
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if (iter_ + 1) % 2 == 0 or (iter_ + 1) == epoch:
                # v13: 更新打印信息
                print(f"  [Client] Local Epoch {iter_+1}/{epoch}, Avg Loss: {epoch_loss[-1]:.4f} (CE: {loss_ce.item():.4f}, Distill: {loss_distill.item():.4f})")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), client_quality_score

def globaltest(net, test_dataset, args):
    # (此函数与 v9/v10/v11/v12 相同)
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc