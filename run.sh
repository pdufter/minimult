##########################################################################################
######## SETUP
# myenv multl
WORKDIR="/mounts/work/philipp/multlrffinal"
export CUDA_VISIBLE_DEVICES=1
export seed=42

runexperiment() {
	# run pretraining
	python run_language_modeling.py \
	--train_data_file ${WORKDIR}/corpora/eng_easytoread${6}.txt \
	--output_dir ${WORKDIR}/models/${10},${1} \
	--model_type bert \
	--mlm \
	--config_name configs/bert-${3}.json \
	--tokenizer_name ${WORKDIR}/vocab/bert-eng_easytoread_2000/ \
	--do_train \
	--do_eval \
	--per_gpu_train_batch_size ${4} \
	--num_train_epochs ${2} \
	--warmup_steps 50 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--overwrite_output_dir \
	--block_size 128 \
	--line_by_line \
	--eval_data_file ${WORKDIR}/corpora/eng_easytoread.txt \
	--per_gpu_eval_batch_size ${4} \
	--learning_rate ${7} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--seed ${10} \
	--overwrite_cache \
	--gradient_accumulation_steps ${5} \
	--add_fake_english \
	${11}
	# evaluate multilinguality
	python evaluate.py \
	--model_name_or_path ${WORKDIR}/models/${10},${1} \
	--eval_data_file ${WORKDIR}/corpora/eng_kingjames_10K.txt \
	--exid ${1} \
	--seed ${10} \
	--modeltype bert \
	--take_n_sentences ${8} \
	--outfile ${WORKDIR}/results/kingjames10K.txt \
	${11}
	# perplexity on dev
	python run_language_modeling.py \
	--train_data_file ${WORKDIR}/corpora/eng_kingjames_10K.txt \
	--model_type bert \
	--output_dir ${WORKDIR}/models/${10},${1} \
	--model_name_or_path ${WORKDIR}/models/${10},${1} \
	--mlm \
	--do_eval \
	--per_gpu_train_batch_size ${4} \
	--num_train_epochs ${2} \
	--warmup_steps 50 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--overwrite_output_dir \
	--block_size 128 \
	--line_by_line \
	--eval_data_file ${WORKDIR}/corpora/eng_kingjames_10K.txt \
	--per_gpu_eval_batch_size ${4} \
	--learning_rate ${7} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--seed ${10} \
	--overwrite_cache \
	--gradient_accumulation_steps ${5} \
	--add_fake_english \
	--eval_output_file ${WORKDIR}/results/perpl_mask_only_kingjames_10K.txt \
	--replacement_probs 1.0,0.0,0.0 \
	${11}
	# perplexity on train
	python run_language_modeling.py \
	--train_data_file ${WORKDIR}/corpora/eng_easytoread.txt \
	--model_type bert \
	--output_dir ${WORKDIR}/models/${10},${1} \
	--model_name_or_path ${WORKDIR}/models/${10},${1} \
	--mlm \
	--do_eval \
	--per_gpu_train_batch_size ${4} \
	--num_train_epochs ${2} \
	--warmup_steps 50 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--overwrite_output_dir \
	--block_size 128 \
	--line_by_line \
	--eval_data_file ${WORKDIR}/corpora/eng_easytoread.txt \
	--per_gpu_eval_batch_size ${4} \
	--learning_rate ${7} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--seed ${10} \
	--overwrite_cache \
	--gradient_accumulation_steps ${5} \
	--add_fake_english \
	--eval_output_file ${WORKDIR}/results/perpl_mask_only.txt \
	--replacement_probs 1.0,0.0,0.0 \
	${11}
}

######## CREATE KNN-REPLACE
# prepare corpora
mkdir -p ${WORKDIR}/vecmap/
python utils/prepare_for_fasttext.py \
--corpus ${WORKDIR}/corpora/eng_easytoread.txt \
--vocab ${WORKDIR}/vocab/bert-eng_easytoread_2000/ \
--prefix "::" \
--outfile ${WORKDIR}/vecmap/corpus_fake.txt

python utils/prepare_for_fasttext.py \
--corpus ${WORKDIR}/corpora/eng_easytoread.txt \
--vocab ${WORKDIR}/vocab/bert-eng_easytoread_2000/ \
--prefix "" \
--outfile ${WORKDIR}/vecmap/corpus_english.txt

# train monolingual spaces
for lang in english fake 
do
nice -n 19 /mounts/Users/cisintern/philipp/Dokumente/fastText-0.9.1/fasttext skipgram \
-input ${WORKDIR}/vecmap/corpus_${lang}.txt \
-output ${WORKDIR}/vecmap/vectors_${lang} \
-dim 300 \
-thread 48 
done

# align them with vecmap
# export CUPY_CACHE_DIR="/mounts/work/${USER}/tmp/.cuda"
# myenv default
# in /mounts/work/${USER}/bert_alignment/vecmap
# maybe decrease number of iterations?
python map_embeddings.py --unsupervised \
${WORKDIR}/vecmap/vectors_english.vec \
${WORKDIR}/vecmap/vectors_fake.vec \
${WORKDIR}/vecmap/vectorsmapped_english.vec \
${WORKDIR}/vecmap/vectorsmapped_fake.vec \
--cuda \
-v

tail -n +3 ${WORKDIR}/vecmap/vectorsmapped_english.vec >> ${WORKDIR}/vecmap/vectorsmapped.vec
tail -n +3 ${WORKDIR}/vecmap/vectorsmapped_fake.vec >> ${WORKDIR}/vecmap/vectorsmapped.vec


##########################################################################################
######## EXPERIMENTS

for seed in 0 42 43 100 101
do
	runexperiment 0 100 small 256 1 "" 2e-3 -1 135 ${seed} ""
	runexperiment 1 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions"
	runexperiment 2 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--shift_special_tokens"
	runexperiment 3 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--invert_order"
	runexperiment 4 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--do_not_replace_with_random_words"
	runexperiment 5 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens"
	runexperiment 6 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --do_not_replace_with_random_words"

	runexperiment 7 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--shift_special_tokens --do_not_replace_with_random_words"
	runexperiment 8 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
	runexperiment 9 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words --invert_order"

	runexperiment 18 1 small 2 1 _2lines 2e-3 -1 100000 ${seed} ""
	runexperiment 19 1 small 2 1 _2lines 2e-3 -1 100000 ${seed} "--language_specific_positions"
	runexperiment 21 200 small 256 1 "" 2e-3 -1 100000 ${seed} "--no_parallel_data"
	runexperiment 21b 200 small 256 1 "" 2e-3 -1 100000 ${seed} "--no_parallel_data --language_specific_positions"

	runexperiment 15 100 base 16 16 "" 1e-4 -1 100000 ${seed} ""
	runexperiment 16 100 base 16 16 "" 1e-4 -1 100000 ${seed} "--language_specific_positions"
	runexperiment 17 100 base 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"

	runexperiment 30 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--replace_with_nn 1 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"
	runexperiment 31 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--replace_with_nn 5 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"
done

######## CURVES

for seed in 0 42 43 100 101
for seed in 100
do
runexperiment 100curve0 100 small 256 1 "" 2e-3 -1 675 ${seed} ""
runexperiment 100curve15 100 base 16 16 "" 1e-4 -1 675 ${seed} ""
runexperiment 100curve17 100 base 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
runexperiment 100curve30 100 small 256 1 "" 2e-3 -1 675 ${seed} "--replace_with_nn 1 --replacement_probs 0.5,0.2,0.6 --vecmap ${WORKDIR}/vecmap/vectorsmapped.vec"
done


# evaluate individual checkpoints in the curves
runexperiment() {
	python evaluate.py \
	--model_name_or_path ${WORKDIR}/models/${10},${1} \
	--eval_data_file ${WORKDIR}/corpora/eng_kingjames_10K.txt \
	--exid ${1} \
	--seed ${10} \
	--modeltype bert \
	--take_n_sentences ${8} \
	--outfile ${WORKDIR}/results/kingjames10K_100curves.txt \
	${11}
	# perplexity on dev
	python run_language_modeling.py \
	--train_data_file ${WORKDIR}/corpora/eng_kingjames_10K.txt \
	--model_type bert \
	--output_dir ${WORKDIR}/models/${10},${1} \
	--model_name_or_path ${WORKDIR}/models/${10},${1} \
	--mlm \
	--do_eval \
	--per_gpu_train_batch_size ${4} \
	--num_train_epochs ${2} \
	--warmup_steps 50 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--overwrite_output_dir \
	--block_size 128 \
	--line_by_line \
	--eval_data_file ${WORKDIR}/corpora/eng_kingjames_10K.txt \
	--per_gpu_eval_batch_size ${4} \
	--learning_rate ${7} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--seed ${10} \
	--overwrite_cache \
	--gradient_accumulation_steps ${5} \
	--add_fake_english \
	--eval_output_file ${WORKDIR}/results/perpl_mask_only_kingjames_10K_100curves.txt \
	--replacement_probs 1.0,0.0,0.0 \
	${11}
	# perplexity on train
	python run_language_modeling.py \
	--train_data_file ${WORKDIR}/corpora/eng_easytoread.txt \
	--model_type bert \
	--output_dir ${WORKDIR}/models/${10},${1} \
	--model_name_or_path ${WORKDIR}/models/${10},${1} \
	--mlm \
	--do_eval \
	--per_gpu_train_batch_size ${4} \
	--num_train_epochs ${2} \
	--warmup_steps 50 \
	--logging_steps ${9} \
	--save_steps ${9} \
	--overwrite_output_dir \
	--block_size 128 \
	--line_by_line \
	--eval_data_file ${WORKDIR}/corpora/eng_easytoread.txt \
	--per_gpu_eval_batch_size ${4} \
	--learning_rate ${7} \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--seed ${10} \
	--overwrite_cache \
	--gradient_accumulation_steps ${5} \
	--add_fake_english \
	--eval_output_file ${WORKDIR}/results/perpl_mask_only_100curves.txt \
	--replacement_probs 1.0,0.0,0.0 \
	${11}
}

for seed in 43 100 101
do
for steps in 5 10 20 40 60 80 95
do
	runexperiment 100curve0/checkpoint-$[135 * $steps] 100 small 256 1 "" 2e-3 -1 675 ${seed} ""
	runexperiment 100curve15/checkpoint-$[135 * $steps] 100 base 16 16 "" 1e-4 -1 675 ${seed} ""
	runexperiment 100curve17/checkpoint-$[135 * $steps] 100 base 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
	runexperiment 100curve30/checkpoint-$[135 * $steps] 100 small 256 1 "" 2e-3 -1 675 ${seed} ""
done
done

