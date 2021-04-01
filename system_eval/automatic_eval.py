# +
import sys
sys.path.append("/home/pouramini/comet-atomic-2020-t5-colab/")

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
        tails = [l["target"]]
        head = l["source"]
        gens = l["generations"]

    return gens, tails, head

def get2(l):
    return list(zip(*l))[1]


# +
import sacrebleu

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
        new_tails = []
        for t in tails:
            end_index = t.index("[EOS]") if "[EOS]" in t else len(t)
            t = t[:end_index]
            new_tails.append(t)
        tails = new_tails

        sentence_tails = [t.lower().strip() for t in tails]
        #print("sentence_tails:", sentence_tails)
        split_tails = [t.lower().split() for t in tails]

        for (j, g) in enumerate(gens[:k]):
            #print("g:",g)
            start_index = g.index("[GEN]") if "[GEN]" in g else 0
            end_index = g.index("[EOS]") if "[EOS]" in g else 0
            if start_index > 0 and end_index > 0:
                g = g[start_index+5:end_index].strip()
            elif end_index > 0:
                g = g[:end_index].strip()
            #print("g2:",g)
            key = str(i) + "_" + str(j)
            topk_gts[key] = sentence_tails
            topk_res[key] = [g.lower()]
            
            #print("g.lower()", g.lower())
            #print("split_tails", split_tails)
            #print("g.lower().split", g.lower().split())

            #b = sacrebleu.sentence_bleu(sentence_tails, 
            #                  [g.lower()])
            #print("b1:",b.score)

            b = sentence_bleu(split_tails, 
                              g.lower().split(), 
                              weights=(0.5, 0.5))
            #print("b2:",b)
            
            topk_bleu_score.append((l, b))
            if g in sentence_tails:
                #print("Exact match between", g, " and ", sentence_tails)
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
    print("Exact Match:", np.mean(get2(topk_exact_match)))
    print("Exact Match Not None", np.mean(get2(topk_exact_match_not_none)))
    print("Mean sent BLEU score", np.mean(get2(topk_bleu_score)))
    QGEval = QGEvalCap(model_name, topk_gts, topk_res, calc_bert_score=False)
    scores,_ = QGEval.evaluate()
    scores["Exact_match"] = np.mean(get2(topk_exact_match))
    scores["Data rows"] = len(data)
    scores["Records"] = len(topk_gts)
    scores["TopK"] = k
    #scores["TailIsHead"] = np.mean(get2(topk_is_head))
    print(scores)
    return scores

def eval(data_file, data_type, model_name, topk=1):
    if type(data_file) is str:
        data = read_jsonl(data_file)
        print("Len data:", len(data))
        return topk_eval(model_name, data, data_type, k=topk)
    else:
        src = data_file["source"]
        target = data_file["target"]
        gens = data_file["gens"]
        with open(src, "r") as f:
            src_lines = f.readlines()
        with open(target, "r") as f:
            target_lines = f.readlines()
        with open(gens, "r") as f:
            gens_lines = f.readlines()
        print("src", len(src_lines), " t:", len(target_lines), "gen:", len(target_lines))
        
        data = []
        old_s, old_t, old_g = "","",""
        dups = 0
        new_gens = 0
        for s,t,g in zip(src_lines, target_lines, gens_lines):
            if s != old_s:
                d = {}
                d["source"] = s
                d["target"] = t
                d["generations"] = [g]
                data.append(d)
            elif g != old_g:
                d["generations"].append(g)
                #print("new gen for ", d)
                new_gens += 1
            else:
                #print("duplicate ", s) 
                dups += 1
            old_s, old_t, old_g = s, t, g
        print("New gens:", new_gens, " duplicates: ", dups)

        print("len of data:", len(data))
        #return ""
        return topk_eval(model_name, data, data_type, k=topk)

            
            

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]

