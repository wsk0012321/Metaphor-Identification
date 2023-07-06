import torch
import pandas as pd
import transformers
import nltk
import re
from tqdm import tqdm
from nltk.stem import SnowballStemmer
from transformers import RobertaModel, RobertaTokenizer
from collections import Counterz

ss = SnowballStemmer('english')
base_model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def encode_data(df, max_sequence_length, tokenizer):
    
    # get target sentence mask
    def get_sentence_mask(input_ids, tokenizer, initial_final):
        attention_mask = [0 if token_type_id in [tokenizer.cls_token_id, tokenizer.sep_token_id] else 1 for token_type_id in input_ids]
      
        # make sentence mask
        if initial_final > 0:
            count = 0
            for i,ind in enumerate(attention_mask):
                if ind == 0:
                    count += 1
                    if count == initial_final:
                        start_id = i
                    elif count == initial_final+1:
                        end_id = i
                    else:
                        pass
        elif initial_final < 0:
            special_token_position = [i for i,ind in enumerate(attention_mask) if ind == 0]
            end_id = special_token_position[initial_final]
            start_id = special_token_position[initial_final-1]

        sentence_mask = []
        for i, value in enumerate(attention_mask):
            if i in range(start_id+1, end_id):
                sentence_mask.append(1)
            else:
                sentence_mask.append(0)
                
        return sentence_mask, attention_mask, start_id, end_id

    # get target word mask
    def get_target_mask(input_ids, target_word, start_id, end_id, sentence_mask):
        
        def is_sublist(sublist, main_list):
            sub_length = len(sublist)
            main_length = len(main_list)
    
            for i in range(main_length - sub_length + 1):
                if main_list[i:i+sub_length] == sublist:
                    return True
            return False

        # create a container
        container = 'it is '+ target_word
        target_word_tokenized = tokenizer.tokenize(container)[2:]
        target_word_tokenized = [subword.lstrip('Ġ') for subword in target_word_tokenized]
        encoded_sent = tokenizer.convert_ids_to_tokens(input_ids[start_id+1: end_id])
        encoded_sent = [token.lstrip('Ġ') for token in encoded_sent]

        if is_sublist(target_word_tokenized, encoded_sent) is True:
            start = encoded_sent.index(target_word_tokenized[0])
            end = encoded_sent.index(target_word_tokenized[-1])
            index = list(range(start+start_id + 1,end+2+start_id))
            target_word_encodings = input_ids[start+start_id+1:end+start_id+2]
            
        else:
            subwords = [subword.lstrip('Ġ') for subword in tokenizer.tokenize(target_word)]
            if subwords[0] in encoded_sent and subwords[-1] in encoded_sent:
                start = encoded_sent.index(subwords[0])
                end = encoded_sent.index(subwords[-1])
            else:
                s = 0
                e = 0
                for token in encoded_sent:
                    if s == 0:
                        if subwords[0] in token:
                            s = 1
                            start = encoded_sent.index(token)
                for token in encoded_sent[start:]:
                    if e == 0:
                        if subwords[-1] in token:
                            e = 1
                            end = encoded_sent.index(token)
            index = list(range(start+start_id + 1,end+2+start_id))
            target_word_encodings = input_ids[start+start_id+1:end+start_id+2]
            
        target_mask = [1 if i in index else 0 for i, label in enumerate(sentence_mask)]
        
        
        
        return target_mask, target_word_encodings

    # domain encoding
    def encode_domain(lst, tokenizer):

        input_ids = [tokenizer.encode(domain, add_special_tokens=True) for domain in lst]
        # ignore the special tokens
        attention_mask = []
        for input_ids_sublist in input_ids:
            attention_mask.append([0 if idx in [tokenizer.cls_token_id, tokenizer.sep_token_id] else 1 for idx in input_ids_sublist])

        return input_ids, attention_mask

    # get encoding of a word's baisic sense by its presence in an example
    def get_word_sense(word, example, tokenizer):
    
        encoded_example_ids = tokenizer.encode_plus(example, add_special_tokens=False)['input_ids']
        encoded_example = [token.lstrip('Ġ') for token in tokenizer.convert_ids_to_tokens(encoded_example_ids)]
        target_ids = None
        for i, token in enumerate(encoded_example):
            # locate the target word by its lemmma
            if ss.stem(token) == ss.stem(word):
                target_ids = i
        # in case that no identity is found, we use minimal editing distance
        if target_ids is None:
            distance = {}
            for i, token in enumerate(encoded_example):
                distance[(token,i)] = nltk.edit_distance(token,word)
                distance = dict(sorted(distance.items(), key = lambda x: -x [1], reverse = True))
            target_ids = list(distance.keys())[0][1]
    
        return [encoded_example_ids[target_ids]]

    # merge the ids to get the encoding for sentence
    def get_sentence_encoding(sentences, word_sense_ids):
        sent_encodings = []
        sent_encoding = []
        start = 0
        for i in range(len(sentences)-1):
            if sentences[i] == sentences[i+1] and i != (len(sentences) - 2):
                sent_encoding.append(word_sense_ids[i][0])
            # for the last element in the whole data
            elif sentences[i] == sentences[i+1] and i == (len(sentences) - 2):
                sent_encoding.append(word_sense_ids[i][0])
                sent_encoding.append(word_sense_ids[i+1][0])
                length = i + 2 - start; 
                for x in range(length):
                    sent_encodings += [sent_encoding]
            else:
                sent_encoding.append(word_sense_ids[i][0])
                length = i + 1 - start
                for x in range(length):
                    sent_encodings += [sent_encoding]
                start = i + 1
                sent_encoding = []
                
        return sent_encodings

    # get contextual sentence encodings
    def encode_context(context,tokenizer,target_word, initial_final):
        encoded = tokenizer.encode_plus(context, add_special_tokens=True)
        input_ids = encoded['input_ids']
        sentence_mask, attention_mask_1, start_id, end_id = get_sentence_mask(input_ids, tokenizer, initial_final)
        word_mask, target_word_encodings = get_target_mask(input_ids, target_word, start_id, end_id, sentence_mask)
    
        return sentence_mask, attention_mask_1, word_mask, target_word_encodings, input_ids
    
    # padding and truncation
    def padding(max_sequence_length, all_encodings):
    
        pad_id = tokenizer.pad_token_id
        
        def pad_mask(max_sequence_length, inputs):
            if len(inputs) < max_sequence_length:
                padding_length = max_sequence_length - len(inputs)
                inputs += [0] * padding_length
            else:
                pass
            
            return inputs
    
        def truncate(max_sequence_length, inputs):
            if len(inputs) > max_sequence_length:
                inputs = inputs[:max_sequence_length]
            else:
                pass
        
            return inputs
    
        def pad(max_sequence_length, inputs):
            if len(inputs) < max_sequence_length:
                padding_length = max_sequence_length - len(inputs)
                inputs += [pad_id] * padding_length
            else:
                pass
        
            return inputs
        
        padded_encodings = []
    
        for sm, wm, wsi, se, di, dm, twe, wsim, sem, ce in all_encodings:
            # truncation
            sm = truncate(max_sequence_length, sm)
            wm = truncate(max_sequence_length, wm)
            wsi = truncate(max_sequence_length, wsi)
            se = truncate(max_sequence_length, se)
            di = truncate(max_sequence_length, di)
            ce = truncate(max_sequence_length, ce)
            dm = truncate(max_sequence_length, dm)
            twe = truncate(max_sequence_length, twe)
            wsim = truncate(max_sequence_length, wsim)
            sem = truncate(max_sequence_length, sem)
            # padding
            sm = pad_mask(max_sequence_length, sm)
            wm = pad_mask(max_sequence_length, wm)
            wsi = pad(max_sequence_length, wsi)
            se = pad(max_sequence_length, se)
            di = pad(max_sequence_length, di)
            ce = pad(max_sequence_length, ce)
            dm = pad_mask(max_sequence_length, dm)
            twe = pad(max_sequence_length, twe)
            wsim = pad_mask(max_sequence_length, wsim)
            sem = pad_mask(max_sequence_length, sem)
        
            padded_encodings.append([sm,wm,wsi,se,di,dm,twe,wsim,sem,ce])
    
        return padded_encodings
    
    tokens = df['tokens'].tolist()
    sentences = [sent.strip() for sent in df['sentence'].tolist()]
    contexts = df['context'].tolist()
    domains = df['domains'].tolist()
    examples = df['examples'].tolist()
    label_list = df['labels'].tolist()
    weights = df['weights'].tolist()
    text = df['text'].tolist()
    text = [re.sub('\n+',' ',word).strip() for word in text]
    sent_ids = df['sent_ids'].tolist()
    
    # encode domain list
    domain_ids, domain_masks = encode_domain(domains, tokenizer)
    
    # encode each token
    word_sense_ids = [get_word_sense(t,e,tokenizer) for t,e in zip(text, examples)]
    
    # create mask
    word_sense_ids_mask = [[1] * len(ids) for ids in word_sense_ids]

    # encode each sentence alone
    sent_encodings = get_sentence_encoding(sentences, word_sense_ids)
    sent_encodings_mask = [[1] * len(encodings) for encodings in sent_encodings]

    # get masks
    sentence_masks, attention_masks, word_masks, target_word_encodings, context_encodings = [],[],[],[],[]
    for c,w,idx in (zip(contexts, text, sent_ids)):
        initial_final = 3
        sentence_mask, attention_mask, word_mask, target_word_encoding, context_encoding = encode_context(c, tokenizer, w, initial_final)
        sentence_masks.append(sentence_mask)
        attention_masks.append(attention_mask)
        word_masks.append(word_mask)
        target_word_encodings.append(target_word_encoding)
        context_encodings.append(context_encoding)
    
    all_encodings = list(zip(sentence_masks, word_masks, word_sense_ids, sent_encodings, domain_ids, domain_masks, target_word_encodings, word_sense_ids_mask, sent_encodings_mask, context_encodings))    
    
    padded_encodings = padding(max_sequence_length, all_encodings)
    
    return list(zip(padded_encodings, weights, label_list))

data = pd.read_csv('./data/train_VUA.csv')
data.dropna(inplace=True)

# add sentence indices
count = 0
sent_ids = []
sents = data['sentence'].tolist()
for i in range(len(sents)):
    if i < len(sents) - 1:
        if sents[i+1] == sents[i]:
            sent_ids.append(count)
        else:
            sent_ids.append(count)
            count+=1
    else:
        sent_ids.append(count)
data['sent_ids'] = sent_ids

max_sequence_length = 150
padded_encodings = encode_data(data, max_sequence_length, tokenizer)
df = pd.DataFrame.from_records(padded_encodings, columns = ['encodings','weights','labels'])

# linear model 1
def tag_match(tag):
    match_rule = {
        '^AJ': 'ADJ',
        '^AV': 'ADV',
        '^CJ': 'CC',
        '^CRD': 'CRD',
        '^D': 'DT',
        '^AT0': 'AT0',
        '^EX0': 'EX0',
        '^ITJ': 'ITJ',
        '^NN': 'NN',
        '^PN': 'PN',
        '^ORD': 'ORD',
        '^NP': 'PN',
        '^POS':'POS',
        '^PR': 'PRP',
        '^V': 'VB',
        '^ZZ0': 'ZZ',
        '^UNC': 'UNC',
        '^TO0': 'TO0',
        '^XX0': 'XX0'
    }
    matched = False
    for pattern in match_rule.keys():
        if matched == False and re.match(pattern, tag):
            tag = match_rule[pattern]
            matched = True
    return tag

tag_list = [tag_match(tag) for tag in data['BNC'].tolist()]
lemma_list = data['tokens'].tolist()
lemma_tags = list(zip(lemma_list, tag_list))
label = data['labels'].to_list()
lemma_tag_label = list(zip((lemma_tags), label))
weights_word = {}
for x,y in Counter(lemma_tag_label).most_common():
    if x[0] not in weights_word.keys():
        weights_word[x[0]] = [0,0]
        # counts of word
        weights_word[x[0]][1] += y
        # ccounts of label 1
        if x[1] == 1:
            weights_word[x[0]][0] += y 
    else:
        if x[1] == 1: 
            weights_word[x[0]][0] += y
        weights_word[x[0]][1] += y
                
priori_prob = []

for pair in lemma_tags:
    prob = weights_word[pair][0] / weights_word[pair][1]
    priori_prob.append(prob)
    
def sigmoid_transform(data, scale_factor=5):
    return 1 / (2 + np.exp(-data * scale_factor))

data = np.array(priori_prob)
transformed_data = sigmoid_transform(data)
median = np.median(transformed_data)
output_value = [(1 + (x - median))*1.05 for x in transformed_data]

df['probablity'] = output_value
df.to_csv('./data/train_set.csv')

