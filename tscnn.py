import torch
from torch.nn import functional as F
import numpy as np
from typing import Union, NamedTuple

lmcLogits = torch.tensor(([0.1,1.3], [1.0,0.3]))
mcLogits = torch.tensor(([0.1,3.1], [3.0,0]))
labels = [1,0]
files = [3,5]

final_scores = torch.Tensor()
results = {"preds": [], "labels": []}

length = lmcLogits.size()

for i in range (0, length[0]):
    lmcScores = F.softmax(lmcLogits[i, :])
    mcScores = F.softmax(mcLogits[i, :])
    aveScores = (lmcPreds + mcPreds) / 2
    pred = aveScores.argmax(dim=-1).cpu().numpy()
    results["preds"].append(int(pred))
    results["labels"].append(labels[i])
    torch.cat((final_scores, aveScores), dim=0)

accuracy = compute_accuracy(
       np.array(results["labels"]), np.array(results["preds"]), files, np.array(final_scores)
    )

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

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
    files: Union[torch.Tenosr, np.ndarray],
    scores: Union[torch.Tensor, np.ndarray]
    ):
    assert len(labels) == len(preds)

    # stores total number of examples for each class
    class_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    # stores total number of correct predictions for each class
    correct_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    # stores accuracy for each class
    accuracy_dict = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0, 9:0.0}

"""
    for i in range(0,len(labels)-1):
        class_dict[labels[i]] += 1
        if labels[i] == preds[i]:
            correct_dict[labels[i]] += 1

    for key, val in pca_dict.items():
        pca_dict[key] = (correct_dict[key]/class_dict[key])

    return pca_dict
"""
