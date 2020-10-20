import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import tqdm
import json
import sys

def normalize(utterance):
    """
    Normalize the utterances, including remove punctuation, lower and tokenization
    :param utterance: String type, a line of sentence, which need to be normalized
    :return tokens: List type, normalized tokens
    """
    p = re.compile(r"[!\"#$%&\'()*+,-./:;<=>?@\\\[\]^_`{|}~]")
    uttWithoutPun = p.sub(' ', utterance.lower())
    tokens = word_tokenize(uttWithoutPun)
    return tokens

def process_ubuntu(root):
    """
    Process the ubuntu corpus, statistic the utterance, tokens
    :param toot: String type, root dir of corpus
    :return totalUtterance: total number of utterances
    :return totalToken: total number of tokens
    :return dictUtterance: a dict of unique utterances
    :return dictToken: a dict of unique tokens
    """
    totalUtterance=0
    totalToken=0
    dictUtterance=dict() 
    dictToken=dict()
    par = tqdm.tqdm()
    for dirName, subdirList, fileList in os.walk(root):
        par.update(1)
        for fname in fileList:
            if ".tsv" in fname:
                file = os.path.join(dirName, fname)
                totalUtterance, totalToken, dictUtterance, dictToken = statistics_ubuntu(file, \
                    totalUtterance, totalToken, dictUtterance, dictToken)
    return totalUtterance, totalToken, dictUtterance, dictToken

def statistics_ubuntu(file, totalUtterance, totalToken, dictUtterance, dictToken):
    """
    Process a sigle file of ubuntu corpus, statistic the utterance, tokens
    :param file: String type, root dir of corpus
    :param totalUtterance: total number of utterances
    :param totalToken: total number of tokens
    :param dictUtterance: a dict of unique utterances
    :param dictToken:  a dict of unique tokens
    :return totalUtterance: updated total number of utterances
    :return totalToken: updated total number of tokens
    :return dictUtterance: updated dict of unique utterances
    :return dictToken: updated dict of unique tokens
    """
    ucount = 0
    tcount = 0
    with open(file, 'r') as f:
        for line in f:
            utterance =  line.split('\t')[3]
            ucount += 1
            if utterance not in dictUtterance:
                dictUtterance[utterance] = 1
            tokens = normalize(utterance)
            tcount += len(tokens)
            for token in tokens:
                if token not in dictToken:
                    dictToken[token] = 1
    totalUtterance += ucount
    totalToken += tcount
    return totalUtterance, totalToken, dictUtterance, dictToken

def statistics_twitter(file):
    """
    Process the twitter corpus, statistic the utterance, tokens
    :param file: String type, file path of corpus
    :return totalUtterance: total number of utterances
    :return totalToken: total number of tokens
    :return dictUtterance: a dict of unique utterances
    :return dictToken: a dict of unique tokens
    """
    totalUtterance=0
    totalToken=0
    dictUtterance=dict() 
    dictToken=dict()
    ucount = 0
    tcount = 0
    with open(file, 'r') as f:
        par = tqdm.tqdm()
        for line in f:
            par.update(1)
            utterance =  line.split('\t')
            if len(utterance)>1:
                utterance = utterance[1]
                ucount += 1
                if utterance not in dictUtterance:
                    dictUtterance[utterance] = 1
                tokens = normalize(utterance)
                tcount += len(tokens)
                for token in tokens:
                    if token not in dictToken:
                        dictToken[token] = 1
    totalUtterance += ucount
    totalToken += tcount
    return totalUtterance, totalToken, dictUtterance, dictToken

def statistics_json(file):
    """
    Process the json corpus, statistic the utterance, tokens
    :param file: String type, file path of corpus
    :return totalUtterance: total number of utterances
    :return totalToken: total number of tokens
    :return dictUtterance: a dict of unique utterances
    :return dictToken: a dict of unique tokens
    """
    totalUtterance=0
    totalToken=0
    dictUtterance=dict() 
    dictToken=dict()
    ucount = 0
    tcount = 0
    with open(file, 'r') as f:
        dialogues = json.load(f)
        par = tqdm.tqdm()
        for dia in dialogues:
            par.update(1)
            for turn in dialogues[dia]:
                for key in ["sys", "usr"]:
                    if key in turn :
                        utterance = turn[key]
                    if len(utterance)>=1:
                        ucount += 1
                        if utterance not in dictUtterance:
                            dictUtterance[utterance] = 1
                        tokens = normalize(utterance)
                        tcount += len(tokens)
                        for token in tokens:
                            if token not in dictToken:
                                dictToken[token] = 1

    totalUtterance += ucount
    totalToken += tcount
    return totalUtterance, totalToken, dictUtterance, dictToken

def main():
    """
       The main function
    """
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-path':
        inputs = args[1]
    else: 
        print('please input the path of the corpus')
        inputs = input("input:")
    
    corpus=""
    if ".json" in inputs:
        totalUtterance, totalToken, dictUtterance, dictToken = statistics_json(inputs)
        corpus = 'Json corpus'

    elif ".out" in inputs:
        totalUtterance, totalToken, dictUtterance, dictToken = statistics_twitter(inputs)
        corpus = 'Twitter corpus'
    else:
        totalUtterance, totalToken, dictUtterance, dictToken = process_ubuntu(inputs)
        corpus = 'Ubuntu corpus'
    uniqueToken = len(dictToken)
    uniqueUtterance = len(dictUtterance)
    averageUtterance = totalToken/totalUtterance
    print("\n\nFor the {}\n \
    Number of utterances:{}\n \
    average utterance length (in tokens):{}\n \
    number of tokens:{}\n \
    number of unique utterances:{}\n \
    number of unique tokens:{}\n".format(corpus,totalUtterance,\
        averageUtterance,totalToken,uniqueUtterance,uniqueToken))

if __name__ == '__main__':
    main()
