# set working directory
USER="philipp"
WORKDIR=/mounts/work/${USER}/multlcrv
mkdir -p ${WORKDIR}

# create some folders
mkdir -p ${WORKDIR}/corpora
mkdir -p ${WORKDIR}/models
mkdir -p ${WORKDIR}/results
mkdir -p ${WORKDIR}/tmp
mkdir -p ${WORKDIR}/vocab


# install deopendencies (# not tested)
conda create --prefix /mounts/work/${USER}/envs/mult python=3.7
conda activate /mounts/work/${USER}/envs/mult
pip install -r requirements.txt


# get some data
# requires existence of the Parallel Bible Corpus by Cysouw et al.
python -m projects.multlpub.utils.prepare_bible \
--edition eng_easytoread \
--outpath ${WORKDIR}/corpora/eng_easytoread.txt \
--clean_punctuation

# get some dev data
python -m projects.multlpub.utils.prepare_bible \
--edition eng_kingjames \
--outpath ${WORKDIR}/corpora/eng_kingjames.txt \
--clean_punctuation \
--old_testament_only

# cut dev data to 10K
head -n 10000 ${WORKDIR}/corpora/eng_kingjames.txt > ${WORKDIR}/corpora/eng_kingjames_10K.txt


# create a vocabulary
mkdir -p ${WORKDIR}/vocab/bert-eng_easytoread_2000
python -m projects.multlpub.utils.get_vocabulary \
--infile ${WORKDIR}/corpora/eng_easytoread.txt \
--outfolder ${WORKDIR}/vocab/bert-eng_easytoread_2000 \
--vocabsize 2048


# get normal config of bert-base-multilingual-cased as a json
python -m projects.multlpub.utils.get_config \
--outpath projects/multlpub/configs/bert-base-multilingual-cased.json

# MANUALLY: create a smaller config

# MANUALLY: create 2lines corpus
