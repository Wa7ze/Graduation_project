import os
import torch
import json
from PIL import Image

def load_model_once():
    from .mcn.utils import prepare_dataloaders
    from .mcn.model import CompatModel

    _, _, _, _, test_dataset, _ = prepare_dataloaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(test_dataset.vocabulary)).to(device)
    model.load_state_dict(torch.load('./model_train_relation_vse_type_cond_scales.pth'))
    model.eval()

    return model, test_dataset.transform, device

model = None
transform = None
device = None


def ensure_model_loaded():
    global model, transform, device
    if model is None or transform is None or device is None:
        model, transform, device = load_model_once()

def score_outfit(parts):
    ensure_model_loaded()
    with torch.no_grad():
        input_tensor = torch.stack(parts)[:5].unsqueeze(0).to(device)
        score, *_ = model._compute_score(input_tensor)
        return score.item()

def generate_outfit_from_items(wardrobe_items, root_folder="media"):
    """
    Given a list of wardrobe items (from DB), return best scoring 3–4 piece outfit.
    """
    from itertools import combinations
    ensure_model_loaded()
    best_score = -float('inf')
    best_outfit = []

    valid_items = []
    for item in wardrobe_items:
        if not item.photo_path:
            continue
        img_path = os.path.join(root_folder, item.photo_path.name)
        if not os.path.exists(img_path):
            continue
        try:
            img = transform(Image.open(img_path).convert('RGB'))
            valid_items.append((item.id, img))
        except:
            continue

    if len(valid_items) < 3:
        return []

    # Try combinations of 3 to 5
    for r in [3, 4, 5]:
        for combo in combinations(valid_items, r):
            ids, imgs = zip(*combo)
            try:
                score = score_outfit(list(imgs))
                if score > best_score:
                    best_score = score
                    best_outfit = list(ids)
            except:
                continue

    return best_outfit
