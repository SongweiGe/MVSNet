python evaluate.py -g 0 -n mvsnet_large_pretrain239 -ie 734
python evaluate.py -g 1 -n mvsnet_large_pretrain99 -ie 594
python baseline_triangulation.py > logs/baseline_triangulation.txt

python evaluate_psmnet.py -g 0 -n mvsnet_PSMNet_pretrain0 -ie 19
python evaluate_psmnet.py -g 1 -n mvsnet_PSMNet_pretrain239 -ie 259

python warmup_mvsnet.py -n pretrain_psmnet_neg
python warmup_mvsnet.py -n pretrain_psmnet_neg > logs/pretrain_psmnet_neg.txt