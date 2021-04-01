from automatic_eval import *

expts = [['/home/pouramini/atomic_models/myt5_2/pred_generations_prefix_eval.jsonl','T5_prefix_eval', 4, 1]]
expts = [['/home/pouramini/results/res_t5_large_1_checkpoint-10000__2020_test_sample_20.jsonl','T5_large_sample_test', 2, 1]]
#expts = [['/home/pouramini/comet-commonsense/data/atomic/outputs_t5_small_full_2020/results/res_2020_train_all.jsonl','T5_2020_train', 2, 1]]
#expts = [
#         [{"source":"../test.source","target":"../test.target","gens":"../test_generations.txt"}, 
#          "BART", 4, 1]        
#        ]

add_column = True

for (f, m, t, k) in expts:
    print(f)
    print(m)
    print(t)
    s = eval(f, data_type=t, model_name=m, topk=k)
    columns = list(s.keys())
    s_row = toRow(m, s, columns)
    if add_column:
        rows = [[""] + columns]
        add_column = False
    print(s)
    rows.append(s_row)
    res_fname = "/home/pouramini/atomic_results/" + m + "_" + ".txt"
    with open(res_fname, "w") as out:
        print(f)
        print(f, file=out)
        for x in s:
            print (x,':',s[x])
            print (x,':',s[x], file=out)
    print("results were written in", res_fname)

#print(tabulate(rows, headers='firstrow', tablefmt='latex', floatfmt='#.3f'))

# +
#import sacrebleu
#sys = ["This is a cat bad koon"] 
#refs = [["This is a cat int"], 
#        ["This is a bad cat bad"]] 
#
# # +
#refs_split = [r[0].split() for r in refs]
#sys_split = [r.split() for r in sys][0]
#
#print(refs_split)
#print(sys_split)
#bleu = sacrebleu.sentence_bleu(sys[0],[r[0] for r in refs])
#print("bleu", bleu.score)
#print("bleu", round(bleu.score,2))
#
#bleu = sentence_bleu(refs_split, sys_split)
#print("bleu", bleu)
#print("bleu", round(bleu,2))
#
