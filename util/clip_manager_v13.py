# util/clip_manager_v13.py
# (此文件与 v12 完全相同)
import torch
import clip

PROMPT_TEMPLATES = [
    'a photo of a {}.', 'a photo of the {}.', 'a picture of a {}.',
    'a picture of the {}.', 'a good photo of a {}.', 'a drawing of a {}.',
    'a cropped photo of a {}.',
]

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
    return DATASET_CLASSES.get(args.dataset.lower(), None)

def get_text_prototypes(clip_model, class_names):
    if not class_names:
        raise ValueError(f"Dataset not supported by clip_manager_v13.")
    
    print(f"[CLIP Manager] Creating text prototypes using {len(PROMPT_TEMPLATES)} ensembled prompts.")
    device = next(clip_model.parameters()).device
    
    with torch.no_grad():
        all_class_prototypes = []
        for class_name in class_names:
            all_prompt_features_for_class = []
            for template in PROMPT_TEMPLATES:
                text = clip.tokenize(template.format(class_name)).to(device)
                text_features = clip_model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_prompt_features_for_class.append(text_features)
            
            class_prototype = torch.stack(all_prompt_features_for_class).mean(dim=0)
            class_prototype = class_prototype / class_prototype.norm(dim=-1, keepdim=True)
            all_class_prototypes.append(class_prototype)
            
        text_features = torch.cat(all_class_prototypes)
    
    print(f"[CLIP Manager] Ensembled text prototypes created. Shape: {text_features.shape}")
    return text_features