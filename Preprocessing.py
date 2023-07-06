import re
import nltk
from nltk.corpus import wordnet as wn
import spacy
import pandas as pd
from string import punctuation
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm
from string import punctuation

corpus = open(r'./corpus/VUAMC.xml', encoding = 'utf-8').read()
Soup = BeautifulSoup(corpus, 'xml')

# separate the data into train and test set
def separate_data(Soup):
    
    texts = [text for text in Soup.find_all('text')]
    
    train_set = []
    test_set = []
    
    # id of all the texts used for training
    training_partition = [
    'a1e-fragment01',
    'a1f-fragment06',
    'a1f-fragment07',
    'a1f-fragment08',
    'a1f-fragment09',
    'a1f-fragment10',
    'a1f-fragment11',
    'a1f-fragment12',
    'a1g-fragment26',
    'a1g-fragment27',
    'a1h-fragment05',
    'a1h-fragment06',
    'a1j-fragment34',
    'a1k-fragment02',
    'a1l-fragment01',
    'a1m-fragment01',
    'a1n-fragment09',
    'a1n-fragment18',
    'a1p-fragment01',
    'a1p-fragment03',
    'a1x-fragment03',
    'a1x-fragment04',
    'a1x-fragment05',
    'a2d-fragment05',
    'a38-fragment01',
    'a39-fragment01',
    'a3c-fragment05',
    'a3e-fragment03',
    'a3k-fragment11',
    'a3p-fragment09',
    'a4d-fragment02',
    'a6u-fragment02',
    'a7s-fragment03',
    'a7y-fragment03',
    'a80-fragment15',
    'a8m-fragment02',
    'a8n-fragment19',
    'a8r-fragment02',
    'a8u-fragment14',
    'a98-fragment03',
    'a9j-fragment01',
    'ab9-fragment03',
    'ac2-fragment06',
    'acj-fragment01',
    'ahb-fragment51',
    'ahc-fragment60',
    'ahf-fragment24',
    'ahf-fragment63',
    'ahl-fragment02',
    'ajf-fragment07',
    'al0-fragment06',
    'al2-fragment23',
    'al5-fragment03',
    'alp-fragment01',
    'amm-fragment02',
    'as6-fragment01',
    'as6-fragment02',
    'b1g-fragment02',
    'bpa-fragment14',
    'c8t-fragment01',
    'cb5-fragment02',
    'ccw-fragment03',
    'cdb-fragment02',
    'cdb-fragment04',
    'clp-fragment01',
    'crs-fragment01',
    'ea7-fragment03',
    'ew1-fragment01',
    'fef-fragment03',
    'fet-fragment01',
    'fpb-fragment01',
    'g0l-fragment01',
    'kb7-fragment10',
    'kbc-fragment13',
    'kbd-fragment07',
    'kbh-fragment01',
    'kbh-fragment02',
    'kbh-fragment03',
    'kbh-fragment09',
    'kbh-fragment41',
    'kbj-fragment17',
    'kbp-fragment09',
    'kbw-fragment04',
    'kbw-fragment11',
    'kbw-fragment17',
    'kbw-fragment42',
    'kcc-fragment02',
    'kcf-fragment14',
    'kcu-fragment02',
    'kcv-fragment42']
    
    test_partition = [
    
    'a1j-fragment33',
    'a1u-fragment04',
    'a31-fragment03',
    'a36-fragment07',
    'a3e-fragment02',
    'a3m-fragment02',
    'a5e-fragment06',
    'a7t-fragment01',
    'a7w-fragment01',
    'aa3-fragment08',
    'ahc-fragment61',
    'ahd-fragment06',
    'ahe-fragment03',
    'al2-fragment16',
    'b17-fragment02',
    'bmw-fragment09',
    'ccw-fragment04',
    'clw-fragment01',
    'cty-fragment03',
    'ecv-fragment05',
    'faj-fragment17',
    'kb7-fragment31',
    'kb7-fragment45',
    'kb7-fragment48',
    'kbd-fragment21',
    'kbh-fragment04',
    'kbw-fragment09'
    ]
    
    for text in texts:
        # whether the text has attributions
        if text.attrs is not None and 'xml:id' in text.attrs.keys():
            # whether it's in train set
            if text.attrs['xml:id'] in training_partition:
                train_set.append(text)
            elif text.attrs['xml:id'] in test_partition:
                test_set.append(text)
            else:
                pass

    return train_set, test_set

train_set, test_set = separate_data(Soup)

def get_sentences(text):
    sent_list = []
    for sent in text.find_all('s'):
        label = 0
        if sent.find('seg'):
            label = 1
        tokens_list = [element for element in sent.find_all() if element.name in ['w','c']]
        token_with_pos = []
        for token in tokens_list:
            if token.find('seg'):
                token_with_pos.append((token.attrs['lemma'], token.attrs['type'],1,token.text))
            elif token.name == 'c':
                token_with_pos.append((re.sub('[\s\n]+','',token.text), token.attrs['type'],0,token.text))
            else:
                token_with_pos.append((token.attrs['lemma'], token.attrs['type'],0,token.text))
        sent_list.append((re.sub('[\n\s]+',' ', sent.text).strip(), label, token_with_pos))
    df = pd.DataFrame(sent_list, columns = ['sentences','label','tokens'])
    #df.to_csv(r'./data/sentence2.csv', encoding = 'utf-8')
    return df

def assign_weights(tokens):
    if len(tokens) <= 6:
        weights1 = 1.5
    else:
        weights1 = 0.9
    
    if (tokens[0] == 'like' or tokens[0] == 'as') and tokens[1] == 'PRP':
        weights2 = 0.9
    else:
        weights2 = 1
   
    weights = (weights1, weights2)
       
    return weights

def find_context(df):
    sent_list = df['sentences'].tolist()
    label_list = df['label'].tolist()
    token_list = df['tokens'].tolist()
    contexts = []
    for i in range(len(sent_list)):
        
        # surronding four sentences 
        if i == 0:
            context = 2 * ['/n'] + sent_list[:3]
            contexts.append('</s>'.join(context))
       
        elif i == 1:
            context = ['/n'] + sent_list[:4]
            contexts.append('</s>'.join(context))
            
        elif i + 2 < len(sent_list) and i > 1:
            contexts.append('</s>'.join(sent_list[(i-2):(i+3)]))
            
        elif i + 2 == len(sent_list):
            context = sent_list[(i-2):] + ['/n']
            contexts.append('</s>'.join(context))
            
        elif i + 1 == len(sent_list):
            context = sent_list[(i-2):] + 2 * ['/n']
            contexts.append('</s>'.join(context))

    df2 = pd.DataFrame.from_records(zip(label_list, sent_list, contexts,token_list), columns = ['labels', 'sentences', 'context', 'tokens'])

    return df2

# simplify pos tags
def pos_mapping(tag):
    mapping_rules = {
            '^NN': 'n',
            '^RB': 'r',
            '^V': 'v',
            '^AJ': 'a',
            '^PRP': 'IN'
        }
    
    flag= 0
    for pattern in list(mapping_rules.keys()):
        if re.match(pattern, tag) and flag == 0:
            flag = 1
            return mapping_rules[pattern]
    return 'None'
    
# assign domain
def assign_domain(pos_list, punct): 
              
    def get_indices(lst, target):
        indices = []
        for i, value in enumerate(domain_list):
            if value[0] == target:
                indices.append(i)
        return indices
                
    def find_hypernym(term, pos):
        flag = 0
        hypernym = term
        # make synsets
        synsets = wn.synsets(term)
        for i in range(len(synsets)):
            # find the first definition that fits the assigned POS
            if synsets[i].name().split('.')[0] == term and synsets[i]._pos == pos and flag == 0:
                flag = 1
                # get hypernyms
                hypernyms = synsets[i].hypernyms()
                # take the first hypernym
                if len(hypernyms) > 0:
                    hypernym = hypernyms[0].name().split('.')[0]
            else:
                pass

        return hypernym
    
    tokens = [token for token, tag in pos_list]
    tags = [pos_mapping(tag) for token, tag in pos_list]

    count = Counter(tags)
    # check whether the sentence contains more than one verb: verb is assumed to be the root of a local context
    if 'v' in count.keys() and count['v'] > 1 and ',' in tokens and len(tokens) >= 15:
        will_split = True
    else:
        will_split = False
        
    domains = {}
      
    # substitute the tokens with hypernyms
    domain_list = [(find_hypernym(token, tag), tag) for token, tag in zip(tokens, tags)]
    # find split mark for long sentences
    if will_split == True:
        indices = get_indices(domain_list, ',')
        # get the comma in the middle
        size = len(domain_list)
        if size %2 == 0:
            middle = int(len(indices)/2)
        else:
            middle = int((len(indices)-1)/2)
        split_comma_idx = indices[middle]
    # get indices of n, r, v, a, in
    ids = []
    for i, token in enumerate(domain_list):
        if token[1] in ['n','r','v','a','IN']:
            ids.append(i)
    
    for i, token_tag in enumerate(pos_list):
     
        if token_tag[0] not in punct:
            local_domain = []
            if i not in ids:
                ids2 = ids + [i]
            else:
                ids2 = ids
            if will_split == True:
                
                if i < split_comma_idx:
                    ids2 = [x for x in ids2 if x < split_comma_idx]
                    local_domain = [domain_list[a][0] for a in sorted(ids2)]
                    domains[token_tag[0]] = local_domain
                elif i > split_comma_idx:
                    ids2 = [x for x in ids2 if x > split_comma_idx]
                    local_domain = [domain_list[a][0] for a in sorted(ids2)]
                    domains[token_tag[0]] = local_domain
                else:
                    pass
            else:
                local_domain = [domain_list[a][0] for a in sorted(ids2)]
                domains[token_tag[0]] = local_domain
                          
    return domains

def find_examples(token_list, tags, example_data):
    
    def pos_mapping2(tag):
        mapping_rules = {
            '^NN': 'noun',
            '^AV': 'adverb',
            '^V': 'verb',
            '^AJ': 'adjective',
            '^PRP': 'preposition'
               }

        patterns = list(mapping_rules.keys())
        for pattern in patterns:
            if re.match(pattern, tag):
                return mapping_rules[pattern]
        return 'None'
    
    tags = [pos_mapping2(tag) for tag in tags]
    token_tag = list(zip(token_list, tags))
    
    examples = {}
    example_data = example_data.drop([5172,5261,10581]).dropna()
    
    tokens = [eval(token) for token in example_data['tokens'].tolist()]
    contents = example_data['examples'].tolist()
    contents =[nltk.word_tokenize(example)[-1] if re.match('[Ss]ynonym[s]?:',str(example)) else example for example in contents]
    
    for token,content in zip(tokens,contents):
        if token not in examples.keys():
            examples[token] = content
    
    example_list = [examples[token] if token in list(examples.keys()) else token[0] for token in token_tag]
    
    return example_list

example_data = pd.read_csv(r'./data/example_data.csv', encoding = 'utf-8')

# tag mapping according to shared task criterion
def pos_mapping3(tag):
        mapping_rules = {
            '^NN': 'noun',
            '^AV': 'adverb',
            '^V': 'verb',
            '^AJ': 'adjective',
            '^PRP': 'preposition'
               }

        patterns = list(mapping_rules.keys())
        for pattern in patterns:
            if re.match(pattern, tag):
                return mapping_rules[pattern]
        return 'Other'

def process_data(dataset):
    df = get_sentences(dataset)
    sent_list = df['sentences'].tolist()
    df_context = find_context(df)

    df2 = pd.DataFrame(columns = ['labels','tokens','domains','examples','tags','text','sentence','context','weights'])
    punct = list(punctuation)
    for row in df_context.iterrows():
        context = row[1][2]
        sentence = row[1][1]
        tokens = eval(str(row[1][3]))
        token_labels = [token[2] for token in tokens if token[0] not in punct]
        token_list = [token[0] for token in tokens if token[0] not in punct]
        text_list = [token[3].strip() for token in tokens if token[0] not in punct]
        pos_list = [(token[0],token[1]) for token in tokens]
        tags = [token[1] for token in tokens if token[0] not in punct]
        domains = assign_domain(pos_list, punct)
        token_idx = [i for i, token in enumerate(tokens)]
        weights = assign_weights(tokens)
        domain_list =  ['</s>'.join(domains[token]) for token in token_list]
        example_list = find_examples(token_list, tags, example_data)
        tags_BNC = tags
        tags = [pos_mapping3(tag) for tag in tags]
        df_token = pd.DataFrame.from_records(zip(token_labels, token_list, domain_list,example_list, tags, text_list), columns = ['labels','tokens','domains','examples','tags', 'text'])    
        df_token['sentence'] = sentence
        df_token['context'] = context
        df_token['weights'] = str(weights)
        df_token['BNC'] = tags_BNC
        df2 = pd.concat([df2,df_token])
    
    return df2

# get train data
dfs_train = [process_data(data) for data in tqdm(train_set)]
train_data = pd.concat(dfs_train)
train_data.to_csv('./data/train_VUA.csv')

# get test data
dfs_test = [process_data(data) for data in tqdm(test_set)]
test_set = pd.concat(dfs_test)
       
# exclude do, be and have from the test set
tags = test_set['tags'].tolist()
excluded = []
for i,tag in enumerate(tags):
    if tag in ['VBB','VBD','VBG','VBI','VBN','VBZ','VDB','VDD','VDG','VDI','VDN','VDZ','VHB','VHD','VHG','VHI','VHN','VHZ']:
        excluded.append(i)
    
test_set.drop(excluded, inplace=True)
test_Set.reset_index(drop=True, inplace=True)
test_set.to_csv(r'./data/test_VUA.csv', encoding = 'utf-8')

