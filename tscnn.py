import torch
from torch.nn import functional as F
import numpy as np
from typing import Union, NamedTuple

def main():
    lmc_logits = torch.load("LMC")
    mc_logits = torch.load("MC")
    class_labels = torch.load("labels.pt")
    file_lables = torch.load("files.pt")
    final_scores = torch.Tensor()

    logits_length = lmc_logits.size()
    for i in range (0, logits_length[0]):
        lmc_scores = F.softmax(lmc_logits[i, :])
        mc_scores = F.softmax(mc_logits[i, :])
        ave_scores = torch.tensor((lmc_scores + mc_scores) / 2)
        if( i == 0):
            final_scores = torch.cat([final_scores, ave_scores], dim=0)
        else:
            final_scores = torch.stack([final_scores, ave_scores], dim=0)

    pca = compute_pca(class_labels, file_labels, final_scores)

    print(f"class 1 accuracy: {pca[0] * 100:2.2f}")
    print(f"class 2 accuracy: {pca[1] * 100:2.2f}")
    print(f"class 3 accuracy: {pca[2] * 100:2.2f}")
    print(f"class 4 accuracy: {pca[3] * 100:2.2f}")
    print(f"class 5 accuracy: {pca[4] * 100:2.2f}")
    print(f"class 6 accuracy: {pca[5] * 100:2.2f}")
    print(f"class 7 accuracy: {pca[6] * 100:2.2f}")
    print(f"class 8 accuracy: {pca[7] * 100:2.2f}")
    print(f"class 9 accuracy: {pca[8] * 100:2.2f}")
    print(f"class 10 accuracy: {pca[9] * 100:2.2f}")


def compute_pca(
    class_labels: Union[torch.Tensor, np.ndarray],
    file_labels: Union[torch.Tensor, np.ndarray],
    scores: Union[torch.Tensor, np.ndarray],
):

    file_label_dict = {}    #to store correct class label of each file
    file_count_dict = {}    #to store number of segments relating to each file
    file_score_dict = {}    #to store scores of each segment to related file

    scores_size = scores.size()
    for i in range (0, scores_size[0]):
        x = file_labels[i]                        # x = file of segment with scores[i]
        file_label_dict[x] = class_labels[i]      #save actual label for file[x] in dictionary
        if x in file_score_dict:
            file_count_dict[x] += 1
            file_score_dict[x] += scores[i]
        else:
            file_count_dict[x] = 1
            file_score_dict[x] = scores[i]

    file_avg_dict = {}      #to store average score for each file
    file_pred_dict = {}     #to store class prediction for each file

    for key, val in file_score_dict.items():
        file_avg_dict[key] = val/file_count_dict[key]
        file_pred_dict[key] = file_avg_dict[key].argmax(dim=-1).cpu().numpy()

    #Number of files for each class
    total_class_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    #Correctly predicted classes
    correct_class_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    #PCA
    pca_dict = {}

    for key, val in file_label_dict.items():          #for all files
        total_class_dict[val] += 1                    #count number of files for each class
        if(val == file_pred_dict[key]):               #if file is correctly predicted...
            correct_class_dict[val] += 1              #count correct prediction of file to class

    for key, val in total_class_dict.items():         #calculate pca
        if(total_class_dict[key] != 0):
          pca_dict[key] = (correct_class_dict[key]/total_class_dict[key])
        else:
          pca_dict[key] = 0

    return pca_dict
