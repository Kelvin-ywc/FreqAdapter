import numpy as np

def compute_retrieval_acc(label, sim_matrix, topk=1):
    retrieval_acc = 0
    for i in range(len(label)):
        sim = sim_matrix[i]
        top_k_indices = np.argsort(sim)[::-1][:topk]

        if any(idx in label[i] for idx in top_k_indices):
            retrieval_acc += 1
    retrieval_acc /= len(label)
    return retrieval_acc