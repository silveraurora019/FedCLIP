# util/local_training_v12.py
# (基于 v11 精简版 + GPU 修复)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np 
import copy
import clip
from PIL import Image

# 导入 v12 的工具
from .util_v11 import (
    get_output, 
    calculate_sub_prototypes, 
    noise_detection_and_correction_clip, 
    prototype_knowledge_distillation_loss
)
# 导入 v12 的 CLIP 管理器
from .clip_manager_v12 import get_class_names, get_text_prototypes

class DatasetSplit(Dataset):
    # (与 v11 相同)
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
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs = list(idxs)
        self.local_dataset = DatasetSplit(dataset, self.idxs) 

        # === v12: 初始化 CLIP 预言机 (常驻 GPU) ===
        print(f"[Client {self.idxs[0] if len(self.idxs)>0 else 'N/A'}] Loading CLIP Oracle: {args.clip_model_name} onto {args.device}")
        
        self.clip_model, self.clip_preprocess = clip.load(args.clip_model_name, device=args.device)
        self.clip_model.eval() 
        
        class_names = get_class_names(args)
        if class_names is None:
            raise ValueError(f"Dataset {args.dataset} not supported by clip_manager_v12.")
            
        # === v12 修改点 ===
        # (调用 v12 的集成函数，不再需要 args)
        self.clip_text_features = get_text_prototypes(self.clip_model, class_names)
        # === v12 修改结束 ===
        
        self.clip_text_features = self.clip_text_features.to(self.clip_model.dtype)
        
        print(f"[Client] CLIP Oracle initialized (persistent on {args.device}).")
        
    def update_weights(self, net, seed, w_g, epoch, mu=0):
        # (此函数的 *所有* 逻辑与 v11 (精简版) 完全相同)
        
        net_g = copy.deepcopy(net).to(self.args.device)
        net_g.load_state_dict(w_g)
        net_g.eval()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        
        # --- v11/v12: 数据准备 ---
        images_resnet = torch.stack([self.local_dataset[i][0] for i in range(len(self.local_dataset))])
        original_images_data = [self.local_dataset.dataset.data[i] for i in self.local_dataset.idxs]
        images_clip = torch.stack(
            [self.clip_preprocess(Image.fromarray(img) if isinstance(img, np.ndarray) else img) 
             for img in original_images_data]
        )
        original_targets = np.array([self.local_dataset.dataset.targets[i] for i in self.local_dataset.idxs])
        current_labels_tensor = torch.tensor(original_targets, dtype=torch.long)
        
        # === v11/v12 精简: 阶段 A (仅运行一次) ===
        print(f"  [Client] Performing one-time label correction and proto generation...")
        net.eval()
            
        corrected_labels_np, _, client_quality_score = noise_detection_and_correction_clip(
            self.clip_model, 
            self.clip_text_features, 
            images_clip, 
            current_labels_tensor.numpy(), 
            self.args
        )
        current_labels = torch.tensor(corrected_labels_np, dtype=torch.long)

        temp_resnet_dataset = TensorDataset(images_resnet, current_labels)
        eval_loader_resnet = DataLoader(temp_resnet_dataset, batch_size=self.args.local_bs, shuffle=False)
        features_resnet, _ = get_output(eval_loader_resnet, net, self.args, latent=True)
        sub_prototypes = calculate_sub_prototypes(features_resnet, current_labels.numpy(), self.args)

        print(f"  [Client] Correction (Quality: {client_quality_score:.4f}) and prototypes are now fixed for all {epoch} epochs.")
        
        train_dataset = TensorDataset(images_resnet, current_labels)

        for iter_ in range(epoch):
            # --- 步骤B: v10/v11/v12 损失函数训练 ---
            net.train()
            ldr_train = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
            batch_loss = []

            for batch_idx, (batch_images, batch_labels) in enumerate(ldr_train):
                batch_images, batch_labels = batch_images.to(self.args.device), batch_labels.to(self.args.device)
                net.zero_grad()
                features = net(batch_images, latent_output=True)
                logits = net(batch_images, latent_output=False)
                with torch.no_grad():
                    teacher_logits = net_g(batch_images, latent_output=False)

                loss_ce = self.loss_func(logits, batch_labels)
                loss_pkd = prototype_knowledge_distillation_loss(features, logits, sub_prototypes, self.args)
                loss_distill = F.kl_div(
                    F.log_softmax(logits / self.args.temp_distill, dim=1),
                    F.softmax(teacher_logits / self.args.temp_distill, dim=1),
                    reduction='batchmean'
                ) * (self.args.temp_distill ** 2)
                
                loss = loss_ce + self.args.lambda_pkd * loss_pkd + self.args.lambda_distill * loss_distill
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if (iter_ + 1) % 2 == 0 or (iter_ + 1) == epoch:
                print(f"  [Client] Local Epoch {iter_+1}/{epoch}, Avg Loss: {epoch_loss[-1]:.4f}")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), client_quality_score

def globaltest(net, test_dataset, args):
    # (此函数与 v9/v10/v11 相同)
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