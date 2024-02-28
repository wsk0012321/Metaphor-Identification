# Metaphor identification with domains and inter-sentence context
The repertoire contains the code and part of the result of data processing as example. 

## VUA corpus
The model is compatible with VU Amsterdam Metaphor Corpus. Open access to the dataset and corresponding annotation information through: 
http://www.vismet.org/metcor/documentation/home.html

## Enviroment
pip install -r requirements.txt

## Run
1. Run Processing.py on VUAMC.xml
2. Run DataLoder.py on the output files of the first step
3. Run core_model.py on the output of the second step for training and testing

## Train and test partition
The train and test partition are separated in accordance with the setting of the Second Shared Task on Metaphor Detection [1] https://competitions.codalab.org/competitions/22188#learn_the_details

[1] Leong, Chee Wee, et al. "A report on the 2020 VUA and TOEFL metaphor detection shared task." Proceedings of the second workshop on figurative language processing. 2020.
