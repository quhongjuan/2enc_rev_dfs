import pylev
import numpy as np
if __name__ == "__main__":
    pre_path = "predictions22.txt"
    predicts = []
    with open(pre_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            predicts.append(line.strip().split())
    f.close()
    print(len(predicts))

    k = 10
    tar = []
    with open("/mnt/xk/datasets/2-encoders/test/new_tgt-test-d.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            tar.append(line.strip().split())
    f.close()
    print(len(tar))
    scores = 0
    a = 0
    for i in range(len(tar)):
        tmp = []
        for j in range(k):
        #score = sentence_bleu([tar[i]], predicts[i], weights=[0.25,0.25,0.25,0.25])
            dis = pylev.levenshtein(tar[i], predicts[i*k + j])
            a = np.max([len(tar[i]),len(predicts[i])])
            dis = dis / a # normalization
            tmp.append(dis)
        scores += np.min(tmp)
    print(scores / len(tar))
