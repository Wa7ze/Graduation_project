import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torchvision import models

from mcn import resnet
from mcn.utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders
from mcn.model import CompatModel

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fashion Compatibility Evaluation.')
    parser.add_argument('--vse_off', action="store_true")
    parser.add_argument('--pe_off', action="store_true")
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--conv_feats', type=str, default="1234")
    parser.add_argument('--model_path', type=str, default="./model_train_relation_vse_type_cond_scales.pth")
    parser.add_argument('--data_file', type=str, default="train_no_dup_with_category_3more_name.json")
    args = parser.parse_args()

    print(args)
    vse_off = args.vse_off
    pe_off = args.pe_off
    mlp_layers = args.mlp_layers
    conv_feats = args.conv_feats
    model_path = args.model_path
    data_file = args.data_file

    print("Current Working Directory:", os.getcwd())

    # Dataloader
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        prepare_dataloaders(data_file_name=data_file)
    )

    # Load pretrained weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                        vse_off=vse_off, pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats).to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = nn.BCELoss()

    # Compatibility AUC test
    model.eval()
    total_loss = 0
    outputs = []
    targets = []
    batch_num = 0
    for batch_num, batch in enumerate(test_loader, 1):
        print("\r#{}/{}".format(batch_num, len(test_loader)), end="", flush=True)
        lengths, images, names, offsets, set_ids, labels, is_compat = batch
        images = images.to(device)
        target = is_compat.float().to(device)
        with torch.no_grad():
            output, _, _, _ = model._compute_score(images)
            output = output.squeeze(dim=1)
            loss = criterion(output, target)
        total_loss += loss.item()
        outputs.append(output)
        targets.append(target)

    if batch_num > 0:
        print()
        print("Test Loss: {:.4f}".format(total_loss / batch_num))
        outputs = torch.cat(outputs).cpu().data.numpy()
        targets = torch.cat(targets).cpu().data.numpy()
        unique_labels = set(targets)
        print("Target label counts:", {int(t): targets.tolist().count(t) for t in unique_labels})
        if len(unique_labels) < 2:
            print(f"⚠️ Skipping AUC calculation — only one class present in targets: {unique_labels}")
        else:
            auc_score = metrics.roc_auc_score(targets, outputs)
            print("AUC: {:.4f}".format(auc_score))
    else:
        print("No valid batches found for AUC evaluation.")

    # Fill in the blank evaluation
    """is_correct = []
    for i in range(len(test_dataset)):
        print("\r#{}/{}".format(i, len(test_dataset)), end="", flush=True)
        try:
            items, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i)
            question_part = {"upper": 0, "bottom": 1, "shoe": 2, "bag": 3, "accessory": 4}.get(question_part)
            images = [items]

            for option in options:
                new_outfit = items.clone()
                new_outfit[question_part] = option
                images.append(new_outfit)
            images = torch.stack(images).to(device)
            output, _, _, _ = model._compute_score(images)

            if output.argmax().item() == 0:
                is_correct.append(True)
            else:
                is_correct.append(False)
        except Exception as e:
            print("\n[Skipping fitb sample {}] Reason: {}".format(i, e))

    if len(is_correct) > 0:
        print()
        print("FitB ACC: {:.4f}".format(sum(is_correct) / len(is_correct)))
    else:
        print("No valid samples for FITB evaluation.")"""
