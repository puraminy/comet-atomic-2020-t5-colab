# +
import sys
sys.path.append("/home/pouramini/comet-atomic-2020/")

import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from split.utils import read_jsonl, remove_prefix
from evaluation.eval import QGEvalCap
from tabulate import tabulate


# -

def get_refs_preds(l, type=1):
    if type==1:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        prompt = l["prompt"]
        generations = l["generations"]
        gens = [remove_prefix(g, prompt).strip() for g in generations]
    if type==2:
        tails = l["tails"]
        head = l["head"]
        gens = l["generations"]
    if type==3:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        gens = l["generations"]
    if type==4:
        tails = l["target"]
        head = l["source"]
        gens = l["generations"]

    return gens, tails, head

def get2(l):
    return list(zip(*l))[1]


def topk_eval(model_name, data, data_type, k):

    topk_gts = {}
    topk_res = {}
    topk_exact_match = []
    topk_exact_match_not_none = []
    topk_bleu_score = []

    topk_is_head = []

    for i, l in enumerate(data):
        (gens, tails, head) = get_refs_preds(l, type=data_type)
        #print("Gens:", gens)
        #print("Tails:", tails)
        #print("Head:", head)

        sentence_tails = [t.lower() for t in tails]
        print("sentence_tails:", sentence_tails)
        split_tails = [t.lower().split() for t in tails]

        for (j, g) in enumerate(gens[:k]):
            key = str(i) + "_" + str(j)
            topk_gts[key] = sentence_tails
            topk_res[key] = [g.lower()]
            #print("g.lower", g.lower().split())
            #print("split_tails", split_tails)

            b = sentence_bleu(sentence_tails, 
                              g.lower(), 
                              weights=(0.5, 0.5))
            #print("b1:",b)

            b = sentence_bleu(split_tails, 
                              g.lower().split(), 
                              weights=(0.5, 0.5))
            #print("b2:",b)
            
            topk_bleu_score.append((l, b))
            if g in sentence_tails:
                topk_exact_match.append((l, 1))
                if g != "none":
                    topk_exact_match_not_none.append((l, 1))
            else:
                topk_exact_match.append((l, 0))
                if g != "none":
                    topk_exact_match_not_none.append((l, 0))
            if g == head:
                topk_is_head.append((l, 1))
            else:
                topk_is_head.append((l, 0))

    print("---------------TOP K={}---------------".format(k))
    print(np.mean(get2(topk_exact_match)))
    print(np.mean(get2(topk_exact_match_not_none)))
    print(np.mean(get2(topk_bleu_score)))
    QGEval = QGEvalCap(model_name, topk_gts, topk_res)
    scores,_ = QGEval.evaluate()
    scores["Exact_match"] = np.mean(get2(topk_exact_match))
    #scores["TailIsHead"] = np.mean(get2(topk_is_head))
    print(scores)
    return scores


def eval(data_file, data_type, model_name):

    data = read_jsonl(data_file)

    return topk_eval(model_name, data, data_type, k=1)

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]

# +
expts = [['/home/pouramini/atomic_models/mygpt/pred_generations.jsonl',
           'MYGPT', 4]]

add_column = True
print(expts)
for (f, m, t) in expts:
    print(f)
    s = eval(f, data_type=t, model_name=m)
    columns = list(s.keys())
    s_row = toRow(m, s, columns)
    if add_column:
        rows = [[""] + columns]
        add_column = False
    rows.append(s_row)

print(tabulate(rows, headers='firstrow', tablefmt='latex', floatfmt='#.3f'))
# +
import sacrebleu
sys = ["This is cat."] 
refs = [["This is a cat."], 
        ["This is a bad cat."]] 

bleu = sacrebleu.corpus_bleu(sys, refs)
print("bleu", bleu.score)
print("bleu", round(bleu.score,2))

# -



