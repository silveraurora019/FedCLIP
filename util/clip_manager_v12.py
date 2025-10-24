# util/clip_manager_v12.py
import torch
import clip

# === v12: 提示词集成模板 ===
# (一个常用的 7 模板列表)
PROMPT_TEMPLATES = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a picture of a {}.',
    'a picture of the {}.',
    'a good photo of a {}.',
    'a drawing of a {}.',
    'a cropped photo of a {}.',
]

# (数据集类别定义与 v11 相同)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
DATASET_CLASSES = {
    'cifar10': CIFAR10_CLASSES,
    'cifar100': CIFAR100_CLASSES,
}

def get_class_names(args):
    """(此函数与 v11 相同)"""
    return DATASET_CLASSES.get(args.dataset.lower(), None)

def get_text_prototypes(clip_model, class_names):
    """
    v12 核心: 使用*提示词集成*为给定的类名生成文本特征 (原型)。
    """
    if not class_names:
        raise ValueError(f"Dataset not supported by clip_manager_v12.")
    
    print(f"[CLIP Manager] Creating text prototypes using {len(PROMPT_TEMPLATES)} ensembled prompts.")
    
    # 确保模型在正确的设备上
    device = next(clip_model.parameters()).device
    
    with torch.no_grad():
        all_class_prototypes = []
        
        # 遍历每个类
        for class_name in class_names:
            all_prompt_features_for_class = []
            
            # 1. 对该类，编码*所有*的提示模板
            for template in PROMPT_TEMPLATES:
                text = clip.tokenize(template.format(class_name)).to(device)
                text_features = clip_model.encode_text(text)
                
                # 归一化每个提示的特征
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_prompt_features_for_class.append(text_features)
            
            # 2. 对该类的所有模板特征取平均
            # (N_templates, Feature_dim) -> (Feature_dim)
            class_prototype = torch.stack(all_prompt_features_for_class).mean(dim=0)
            
            # 3. 再次归一化平均后的原型
            class_prototype = class_prototype / class_prototype.norm(dim=-1, keepdim=True)
            
            all_class_prototypes.append(class_prototype)
            
        # 4. 拼接所有类的原型
        # (N_classes, 1, Feature_dim) -> (N_classes, Feature_dim)
        text_features = torch.cat(all_class_prototypes)
    
    print(f"[CLIP Manager] Ensembled text prototypes created. Shape: {text_features.shape}")
    return text_features