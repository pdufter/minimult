##########################################################################################
######## SETUP and FUNCTIONS
# myenv multl
WORKDIR="/mounts/work/${USER}/multlxnli10Mcrv"
mkdir -p ${WORKDIR}/corpora
mkdir -p ${WORKDIR}/vocab
mkdir -p ${WORKDIR}/models/pretrained
mkdir -p ${WORKDIR}/models/finetuned
mkdir -p ${WORKDIR}/results
mkdir -p ${WORKDIR}/vecmap

export CUDA_VISIBLE_DEVICES=0
export seed=100
echo $CUDA_VISIBLE_DEVICES $seed


pretrain() {
	python -m real.run_lm \
	--seed ${1} \
	--train_data_file ${2} \
	--eval_data_file ${3} \
	--output_dir ${WORKDIR}/models/pretrained/${1},${5} \
	--model_type bert \
	--mlm \
	--config_name configs/bert-${6}.json \
	--tokenizer_name ${7} \
	--language_id ${4} \
	--do_train \
	--do_eval \
	--per_gpu_train_batch_size ${8} \
	--per_gpu_eval_batch_size ${8} \
	--gradient_accumulation_steps ${9} \
	--num_train_epochs ${10} \
	--warmup_steps ${11} \
	--logging_steps ${12} \
	--save_steps ${12} \
	--block_size 128 \
	--learning_rate ${13} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--overwrite_output_dir \
	--overwrite_cache \
	--line_by_line \
	${14}
}

finetune() {
	python -m real.run_xnli \
	--seed ${1} \
	--data_dir /mounts/work/${USER}/data/xnli/ \
	--model_name_or_path ${WORKDIR}/models/pretrained/${2} \
	--language ${3} \
	--language_id ${4} \
	--tokenizer_name ${5} \
	--output_dir ${WORKDIR}/models/finetuned/${2}-${3} \
	--model_type bert \
	--max_seq_length 128 \
	--do_train \
	--per_gpu_train_batch_size ${6} \
	--per_gpu_eval_batch_size ${6} \
	--gradient_accumulation_steps ${7} \
	--num_train_epochs ${8} \
	--warmup_steps 0 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--learning_rate ${10} \
	--weight_decay 0.0 \
	--adam_epsilon 1e-8 \
	--overwrite_output_dir \
	--overwrite_cache \
	${11}
}

evaluate() {
	python -m real.run_xnli \
	--seed ${1} \
	--data_dir /mounts/work/${USER}/data/xnli/ \
	--language ${4} \
	--language_id ${5} \
	--tokenizer_name ${6} \
	--output_dir ${WORKDIR}/models/finetuned/${2}-${3} \
	--model_name_or_path ${WORKDIR}/models/finetuned/${2}-${3} \
	--eval_output_file ${WORKDIR}/results/seeds.txt \
	--model_type bert \
	--max_seq_length 128 \
	--do_eval \
	--per_gpu_train_batch_size ${7} \
	--per_gpu_eval_batch_size ${7} \
	--gradient_accumulation_steps ${8} \
	--overwrite_cache \
	${9}
}



##########################################################################################
######## DATA PREPROCESSING
langs="en de hi"
for lang in ${langs}
do
	wc -l /mounts/work/${USER}/data/wiki/wiki_${lang}/wiki${lang}_senttok.txt
done
# 104883247 /mounts/work/${USER}/data/wiki/wiki_en/wikien_senttok.txt
# 48412055 /mounts/work/${USER}/data/wiki/wiki_de/wikide_senttok.txt
# 796566 /mounts/work/${USER}/data/wiki/wiki_hi/wikihi_senttok.txt



# START
# subsample to 1GB of data
for lang in ${langs}
do
	shuf -n 10000000 /mounts/work/${USER}/data/wiki/wiki_${lang}/wiki${lang}_senttok.txt > ${WORKDIR}/corpora/${lang}raw.txt
	head -c 1000000000 ${WORKDIR}/corpora/${lang}raw.txt > ${WORKDIR}/corpora/${lang}.txt
	shuf -n 100000 /mounts/work/${USER}/data/wiki/wiki_${lang}/wiki${lang}_senttok.txt > ${WORKDIR}/corpora/${lang}_dev.txt
	head -n 256 ${WORKDIR}/corpora/${lang}.txt > ${WORKDIR}/corpora/${lang}_toy.txt
done
# END


# check vocab size
for lang in ${langs}
do
	python -m utils.check_vocabsize \
		--corpus ${WORKDIR}/corpora/${lang}.txt
done

# get vocabulary for each language and store them
for lang in ${langs}
do
	mkdir -p ${WORKDIR}/vocab/bert-${lang}_20000
	python utils/get_vocabulary.py \
		--infile ${WORKDIR}/corpora/${lang}.txt \
		--outfolder ${WORKDIR}/vocab/bert-${lang}_20000 \
		--vocabsize 20000
done


##########################################################################################
######## VECMAP SPACES

# prepare corpora
for lang in ${langs}
do
	python utils/prepare_for_fasttext.py \
	--corpus ${WORKDIR}/corpora/${lang}.txt \
	--vocab ${WORKDIR}/vocab/bert-${lang}_20000/ \
	--prefix 0${lang}0 \
	--outfile ${WORKDIR}/vecmap/${lang}.txt
done


# train monolingual spaces
for lang in ${langs}
do
nice -n 19 /mounts/Users/cisintern/philipp/Dokumente/fastText-0.9.1/fasttext skipgram \
-input ${WORKDIR}/vecmap/${lang}.txt \
-output ${WORKDIR}/vecmap/vectors_${lang} \
-dim 300 \
-thread 96 
done

for lang in ${langs}
do
	if [ ${lang} != "en" ]; then
		# align them with vecmap
		# export CUPY_CACHE_DIR="/mounts/work/${USER}/tmp/.cuda"
		# myenv default
		# in /mounts/work/${USER}/bert_alignment/vecmap
		python map_embeddings.py --unsupervised \
		${WORKDIR}/vecmap/vectors_${lang}.vec \
		${WORKDIR}/vecmap/vectors_en.vec \
		${WORKDIR}/vecmap/vectorsmapped_${lang}.vec \
		${WORKDIR}/vecmap/vectorsmapped_en_${lang}.vec \
		--orthogonal \
		--cuda \
		-v
	fi
done

tail -n +2 ${WORKDIR}/vecmap/vectorsmapped_en_de.vec >> ${WORKDIR}/vecmap/vectorsmapped.vec
for lang in ${langs}
do
	if [ ${lang} != "en" ]; then
		echo ${lang}
		tail -n +2 ${WORKDIR}/vecmap/vectorsmapped_${lang}.vec >> ${WORKDIR}/vecmap/vectorsmapped.vec
	fi
done


##########################################################################################
######## PRETRAINING
WORKDIR="/mounts/work/${USER}/multlxnli10Mcrv"
export CUDA_VISIBLE_DEVICES=2
export seed=1

train_corpora="${WORKDIR}/corpora/en.txt,${WORKDIR}/corpora/de.txt,${WORKDIR}/corpora/hi.txt"
#train_corpora="${WORKDIR}/corpora/en_toy.txt,${WORKDIR}/corpora/de_toy.txt,${WORKDIR}/corpora/hi_toy.txt"
train_corpora_2lines="${WORKDIR}/corpora/en_toy.txt,${WORKDIR}/corpora/de_toy.txt,${WORKDIR}/corpora/hi_toy.txt"
eval_corpora="${WORKDIR}/corpora/en_dev.txt,${WORKDIR}/corpora/de_dev.txt,${WORKDIR}/corpora/hi_dev.txt"
vocabs="${WORKDIR}/vocab/bert-en_20000/,${WORKDIR}/vocab/bert-de_20000/,${WORKDIR}/vocab/bert-hi_20000/"
n_epochs=4
warmups=3000
batch_size=32
accumulation=8
logging_steps=72896

### original ones (BERT-Based)
pretrain $seed ${train_corpora} ${eval_corpora} en,de,hi 0,orig base ${vocabs} 16 16 2 ${warmups} ${logging_steps} 1e-4 ""
pretrain $seed ${train_corpora} ${eval_corpora} en,de,hi 3,orig base ${vocabs} 16 16 2 ${warmups} ${logging_steps} 1e-4 "--invert_order --invert_langs de"
pretrain $seed ${train_corpora} ${eval_corpora} en,de,hi 8,orig base ${vocabs} 16 16 2 ${warmups} ${logging_steps} 1e-4 "--language_specific_positions --do_not_replace_with_random_words --shift_special_tokens"
pretrain $seed ${train_corpora} ${eval_corpora} en,de,hi 30,orig base ${vocabs} 16 16 2 ${warmups} ${logging_steps} 1e-4 "--replace_with_nn 5 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"


# copy to after1ep and delete optimizer state for finetuning
# for seed in 1 100
# do
# 	for model in 0,orig 3,orig 8,orig 30,orig
# 	do
# 		cp -r ${WORKDIR}/models/pretrained/${seed},${model}/checkpoint-72896/ ${WORKDIR}/models/pretrained/${seed},${model}/after1ep
# 		rm ${WORKDIR}/models/pretrained/${seed},${model}/after1ep/optimizer.pt
# 		rm ${WORKDIR}/models/pretrained/${seed},${model}/after1ep/scheduler.pt
# 	done
# done



##########################################################################################
######## FINETUNE
batch_size=32
n_epochs=3

# # after 1 epoch
# finetune $seed ${seed},0,orig/after1ep en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 ""
# finetune $seed ${seed},3,orig/after1ep en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 "--invert_order --invert_langs de"
# finetune $seed ${seed},8,orig/after1ep en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 "--language_specific_positions --do_not_replace_with_random_words --shift_special_tokens"
# finetune $seed ${seed},30,orig/after1ep en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 "--replace_with_nn 5 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"

# after 2 epochs
finetune $seed ${seed},0,orig en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 ""
finetune $seed ${seed},3,orig en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 "--invert_order --invert_langs de"
finetune $seed ${seed},8,orig en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 "--language_specific_positions --do_not_replace_with_random_words --shift_special_tokens"
finetune $seed ${seed},30,orig en en,de,hi ${vocabs} ${batch_size} 1 ${n_epochs} 99999 2e-5 "--replace_with_nn 5 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"




##########################################################################################
######## EVALUATE
langs="en de hi"

# # after 1 epoch
# for lang in ${langs}
# do
# 	evaluate $seed 42,0,orig/after1ep en ${lang} en,de,hi ${vocabs} ${batch_size} 1 ""
# 	evaluate $seed 42,3,orig/after1ep en ${lang} en,de,hi ${vocabs} ${batch_size} 1 "--invert_order --invert_langs de"
# 	evaluate $seed 42,17/after1ep en ${lang} en,de,hi ${vocabs} ${batch_size} 1 "--language_specific_positions --do_not_replace_with_random_words --shift_special_tokens"
# 	evaluate $seed 42,18,orig en ${lang} en,de,hi ${vocabs} ${batch_size} 1 ""
# 	evaluate $seed 42,100,orig/after1ep en ${lang} en,de,hi ${vocabs} ${batch_size} 1 ""
# done


# after 2 epochs
for seed in 1 42 100
do
	for lang in ${langs}
	do
		evaluate $seed ${seed},0,orig en ${lang} en,de,hi ${vocabs} ${batch_size} 1 ""
		evaluate $seed ${seed},3,orig en ${lang} en,de,hi ${vocabs} ${batch_size} 1 "--invert_order --invert_langs de"
		evaluate $seed ${seed},8,orig en ${lang} en,de,hi ${vocabs} ${batch_size} 1 "--language_specific_positions --do_not_replace_with_random_words --shift_special_tokens"
		evaluate $seed ${seed},30,orig en ${lang} en,de,hi ${vocabs} ${batch_size} 1 ""
	done
done




