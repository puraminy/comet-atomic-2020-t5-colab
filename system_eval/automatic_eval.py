# +
import sys
sys.path.append("/home/ahmad/comet-atomic-2020/")

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
                print("Exact match between", g, " and ", sentence_tails)
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
    print(np.mean(get2(topk_exact_match_not_none)))
    print(np.mean(get2(topk_bleu_score)))
    QGEval = QGEvalCap(model_name, topk_gts, topk_res)
    scores,_ = QGEval.evaluate()
    scores["Exact_match"] = np.mean(get2(topk_exact_match))
    #scores["TailIsHead"] = np.mean(get2(topk_is_head))
    print(scores)
    return scores

def eval(data_file, data_type, model_name):
    if type(data_file) is str:
        data = read_jsonl(data_file)
        print(data[:4])
        return topk_eval(model_name, data, data_type, k=1)
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
        for s,t,g in zip(src_lines, target_lines, gens_lines):
            d = {}
            d["source"] = s
            d["target"] = t
            d["generations"] = [g]
            data.append(d)
        print(data[:4])
        print("len of data:", len(data))
        #return ""
        return topk_eval(model_name, data, data_type, k=1)

            
            

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]

expts = [['/home/ahmad/comet-atomic-2020-t5-colab/pred_generations.jsonl',
           'MYGPT', 4]
        ]
expts = [
         [{"source":"../test.source","target":"../test.target","gens":"../test_generations.txt"}, 
          "BART", 4]        
        ]

add_column = True

for (f, m, t) in expts:
    print(f)
    print(m)
    print(t)
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
sys = ["This is a cat bad koon"] 
refs = [["This is a cat int"], 
        ["This is a bad cat bad"]] 

type(sys) == list
# +
refs_split = [r[0].split() for r in refs]
sys_split = [r.split() for r in sys][0]

print(refs_split)
print(sys_split)
bleu = sacrebleu.sentence_bleu(sys[0],[r[0] for r in refs])
print("bleu", bleu.score)
print("bleu", round(bleu.score,2))

bleu = sentence_bleu(refs_split, sys_split)
print("bleu", bleu)
print("bleu", round(bleu,2))

hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
               'ensures', 'that', 'the', 'military', 'always',
               'obeys', 'the', 'commands', 'of', 'the', 'party']

hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
               'forever', 'hearing', 'the', 'activity', 'guidebook',
               'that', 'party', 'direct']

reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
              'ensures', 'that', 'the', 'military', 'will', 'forever',
              'heed', 'Party', 'commands']

reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
              'guarantees', 'the', 'military', 'forces', 'always',
              'being', 'under', 'the', 'command', 'of', 'the',
               'Party']

reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
              'army', 'always', 'to', 'heed', 'the', 'directions',
              'of', 'the', 'party']

b = sentence_bleu([['This', 'is', 'a', 'cat'], ['This', 'is', 'a', 'bad', 'cat']]
, ['This', 'is', 'a', 'cat', 'bad']) # doctest: +ELLIPSIS

print("bleu score:", round(b,2))
# -
sentence_tails = ['to go home', "to go out"]
g = "to go home" 
g in sentence_tails


