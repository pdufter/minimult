##########################################################################################
######## SETUP
# myenv multl
WORKDIR="/mounts/work/philipp/multlrf"
export CUDA_VISIBLE_DEVICES=2
export seed=43

runexperiment() {
	python -m projects.multlrf.run_language_modeling \
	--train_data_file ${WORKDIR}/corpora/eng_easytoread${6}.txt \
	--output_dir ${WORKDIR}/models2/${10},${1} \
	--model_type bert \
	--mlm \
	--config_name projects/multlrf/configs/bert-${3}.json \
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

	python -m projects.multlrf.evaluate \
	--model_name_or_path /mounts/work/philipp/multlrf/models2/${10},${1} \
	--eval_data_file ${WORKDIR}/corpora/eng_easytoread.txt \
	--exid ${1} \
	--seed ${10} \
	--modeltype bert \
	--take_n_sentences ${8} \
	--outfile ${WORKDIR}/results2/debug.txt \
	${11}
}

echo $CUDA_VISIBLE_DEVICES $seed

##########################################################################################
######## DEBUG
runexperiment replnn,0l 100 small 256 1 "" 2e-3 -1 1350 42 "--language_specific_positions"
runexperiment replnn,1l 100 small 256 1 "" 2e-3 -1 1350 42 "--language_specific_positions --replace_with_nn"
runexperiment replnn,2l 100 small 256 1 "" 2e-3 -1 1350 42 "--language_specific_positions --replace_with_nn --replacement_probs 0.0,1.0"


##########################################################################################
######## MAINS

# alls
# takes around 36 hours
for seed in 0, 42, 43, 100, 101
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
runexperiment 10 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--delete_position_segment_embeddings"
runexperiment 11 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--delete_position_segment_embeddings --shift_special_tokens --do_not_replace_with_random_words"
runexperiment 12 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--replacement_probs 0.4,0.9"

runexperiment 13 200 small 256 1 "" 2e-3 -1 135 ${seed} "--replacement_probs 0.4,0.9"
runexperiment 18 1 small 2 1 _2lines 2e-3 -1 100000 ${seed} ""
runexperiment 19 1 small 2 1 _2lines 2e-3 -1 100000 ${seed} "--language_specific_positions"
runexperiment 20 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--no_parallel_data"
runexperiment 21 200 small 256 1 "" 2e-3 -1 100000 ${seed} "--no_parallel_data"
runexperiment 21b 200 small 256 1 "" 2e-3 -1 100000 ${seed} "--no_parallel_data --language_specific_positions"
runexperiment 22 100 tiny 256 1 "" 2e-3 -1 100000 ${seed} ""

runexperiment 14 50 original 16 16 "" 1e-4 -1 100000 ${seed} ""
runexperiment 15 100 original 16 16 "" 1e-4 -1 100000 ${seed} ""
runexperiment 15b 250 original 16 16 "" 1e-4 -1 675 ${seed} ""
runexperiment 16 100 original 16 16 "" 1e-4 -1 100000 ${seed} "--language_specific_positions"
runexperiment 16b 100 original 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions"
runexperiment 16c 250 original 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions"
runexperiment 17 250 original 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
done

##########################################################################################
######## CURVE
# specials: Curve
runexperiment 1curve 300 small 256 1 "" 2e-3 -1 135 42 "--language_specific_positions"
# evaluate manually

evalonly() {
	python -m projects.multlrf.run_language_modeling \
	--train_data_file ${WORKDIR}/corpora/eng_easytoread${6}.txt \
	--model_name_or_path ${WORKDIR}/models2/${10},${1} \
	--model_type bert \
	--output_dir ${WORKDIR}/models2/${10},${1} \
	--mlm \
	--config_name projects/multlrf/configs/bert-${3}.json \
	--tokenizer_name ${WORKDIR}/vocab/bert-eng_easytoread_2000/ \
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
}

evalonly 4 100 small 256 1 "" 2e-3 -1 100000 ${seed} "" # up to 40!
evalonly 6 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions"
evalonly 7 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--shift_special_tokens"
evalonly 8 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens"
evalonly 9 100 small 256 1 "" 2e-3 -1 100000 ${seed} "--language_specific_positions --shift_special_tokens --invert_order"
evalonly 17 250 original 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens"


for seed in 0 42 43 101
do
	evalonly 17/checkpoint-6750 250 original 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
	runexperiment 17/checkpoint-6750 250 original 16 16 "" 1e-4 -1 675 ${seed} "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
done



evalall() {
	python -m projects.multlrf.run_language_modeling \
	--train_data_file ${WORKDIR}/corpora/eng_easytoread${6}.txt \
	--output_dir ${WORKDIR}/models2/${10},${1} \
	--model_type bert \
	--mlm \
	--config_name projects/multlrf/configs/bert-${3}.json \
	--tokenizer_name ${WORKDIR}/vocab/bert-eng_easytoread_2000/ \
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
	--eval_all_checkpoints \
	${11}
}

# evalall 1curve 300 small 256 1 "" 2e-3 -1 135 42 "--language_specific_positions"
# evalall 13 200 small 256 1 "" 2e-3 -1 135 42 "--replacement_probs 0.4,0.9"

# for steps in 1 2 4 8 16 32 64 128 192 256 300
# do
# 	runexperiment 1curve/checkpoint-$[135 * $steps] 300 small 256 1 "" 2e-3 -1 135 42 "--language_specific_positions"
# done

# for steps in 5 15 30 65 130 195 250
# do
# 	evalonly 17/checkpoint-$[135 * $steps] 250 original 16 16 "" 1e-4 -1 675 42 "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
# 	runexperiment 17/checkpoint-$[135 * $steps] 250 original 16 16 "" 1e-4 -1 675 42 "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
# done


##########################################################################################
######## DEBUG
runexperiment 0debug 20 small 256 1 "" 2e-3 200 135 43 ""
runexperiment 0feedpositionids 20 small 256 1 "" 2e-3 200 100000 ""
runexperiment 1debug 5 small 256 1 "" 2e-3 -1 100000 43 "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words --invert_order"
runexperiment 9 100 small 256 1 " " 2e-3 100 100000 "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words --invert_order"
runexperiment 10 100 small 256 1 "" 2e-3 -1 100000 "--delete_position_segment_embeddings"
runexperiment 8 100 small 256 1 "" 2e-3 -1 135 "--language_specific_positions --shift_special_tokens --do_not_replace_with_random_words"
runexperiment 1_512 100 small 256 1 "" 2e-3 -1 100000 "--language_specific_positions"
runexperiment 1_1024 100 small1024 256 1 "" 2e-3 -1 100000 "--language_specific_positions"


##########################################################################################
######## ANALYSIS

# analysis
python -m projects.multlrf.analysis plot \
--model ${WORKDIR}/models2/42,0/ \
--maxpos 128 \
--outname ${WORKDIR}/models2/42,0/pos.png \
--title original \
--half
python -m projects.multlrf.analysis plot \
--model ${WORKDIR}/models2/42,1/ \
--maxpos 256 \
--outname ${WORKDIR}/models2/42,1/pos.png \
--title lang-pos 0 \
--half
python -m projects.multlrf.analysis plot \
--model ${WORKDIR}/models2/42,9/ \
--maxpos 256 \
--outname ${WORKDIR}/models2/42,9/pos.png \
--title "lang-pos;shift-special;no-random;inv-order" \
--half


# plot learning
for steps in 1 2 4 8 16 32 64 128 192 256 300
do
	runexperiment 1curve/checkpoint-$[135 * $steps] 300 small 256 1 "" 2e-3 -1 135 42 "--language_specific_positions"
done

runexperiment 1curve 100 small 256 1 "" 2e-3 -1 100000 "--language_specific_positions"


##########################################################################################
######## Synchronity

# note requires installation: https://github.com/cisnlp/simalign 
python -m projects.multlrf.synchronity \
--xnli_dir /mounts/work/philipp/data/xnli/XNLI/xnli.dev.tsv \
--language2 de,ar \
--maxn 100 \
--outfile ${WORKDIR}/synchron/debug \
--distance kendall

