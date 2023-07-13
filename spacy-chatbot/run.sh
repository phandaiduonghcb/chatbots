#!/bin/bash
yes | conda install pytorch cpuonly -c pytorch && yes | conda install -c pytorch torchtext torchdata
yes | conda install -c anaconda numpy
yes | conda install -c conda-forge 'portalocker>=2.0.0'
yes | conda install -c conda-forge spacy
yes | python -m spacy download en_core_web_lg
pip install concise-concepts