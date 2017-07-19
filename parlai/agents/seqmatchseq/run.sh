PYTHONPATH=. python examples/display_data.py -t wikiqa
PYTHONPATH=. python parlai/agents/seqmatchseq/build_dict.py -t wikiqa --dict-file data/wikiqa.dict --pretrained_word False

PYTHONPATH=. python examples/train_model.py -m seqmatchseq -t wikiqa -bs 10 -e 10 -vtim 60 --valid_metric MAP --embedding_file ~/data/glove/glove.6B.300d.txt --fix_embedding True --dict-file data/wikiqa.dict -mf data/myseqmatchseq --random_seed 245
