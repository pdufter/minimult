# set working directory
WORKDIR=/mounts/work/philipp/multlrf
mkdir -p ${WORKDIR}

# create some folders
mkdir -p ${WORKDIR}/corpora
mkdir -p ${WORKDIR}/models
mkdir -p ${WORKDIR}/results
mkdir -p ${WORKDIR}/tmp
mkdir -p ${WORKDIR}/vocab


# install deopendencies
conda create --prefix /mounts/work/philipp/envs/mult python=3.7
conda activate /mounts/work/philipp/envs/mult
pip install -r requirements.txt


# get some data
# requires existence of the Parallel Bible Corpus by Cysouw et al.
python -m projects.multlrf.utils.prepare_bible \
--edition eng_easytoread \
--outpath ${WORKDIR}/corpora/eng_easytoread.txt \
--clean_punctuation


# create a vocabulary
mkdir -p ${WORKDIR}/vocab/bert-eng_easytoread_2000
python -m projects.multlrf.utils.get_vocabulary \
--infile ${WORKDIR}/corpora/eng_easytoread.txt \
--outfolder ${WORKDIR}/vocab/bert-eng_easytoread_2000 \
--vocabsize 2048


# get normal config of bert-base-multilingual-cased as a json
python -m projects.multlrf.utils.get_config \
--outpath projects/multlrf/configs/bert-base-multilingual-cased.json

# MANUALLY: create a smaller config

