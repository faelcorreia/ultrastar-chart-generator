#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n audio-processor python=3.10 -y
conda activate audio-processor
conda install -c conda-forge montreal-forced-aligner -y

# MFA models bundled for the supported input languages.
# Portuguese (Brazil)
mfa models download acoustic portuguese_mfa
mfa models download dictionary portuguese_brazil_mfa
mfa models download g2p portuguese_brazil_mfa

# English (US)
mfa models download acoustic english_mfa
mfa models download dictionary english_us_mfa
mfa models download g2p english_us_mfa

# Spanish (Latin America)
mfa models download acoustic spanish_mfa
mfa models download dictionary spanish_latin_america_mfa
mfa models download g2p spanish_latin_america_mfa

pip install -r requirements.txt \
            --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple
