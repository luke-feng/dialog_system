import re
import numpy as np
import sacrebleu
import sys
import os
import warnings
from rouge_metric import PyRouge
import nltk

warnings.filterwarnings('ignore')


def average_utterances_rating(line):
    """
    calculate the average human rating of  the utterances
    :param line: String type, a line of sentence, which includes the human rates
    :return averageRating: float type, average utterances rating
    """
    rates = list(map(int, re.findall(r'(?<=\[|,)\d+', line)))
    npRates = np.array(rates)
    averageRating = npRates.mean()
    return averageRating


def average_human_rating(file):
    """
    calculate the average human rating of  the corpus
    :param file: String type, human rating file name
    :return averageRating: dict type, average human rating for corpus level
    """
    humanScore = dict()
    averageHumanRating = dict()
    for ids in range(1, 22):
        systemId = 'S_' + str(ids)
        humanScore[systemId] = list()
        averageHumanRating[systemId] = 0
    pattern = re.compile(r'^S_\d+')

    with open(file, 'r') as hf:
        for line in hf:
            systemId = pattern.findall(line)
            if len(systemId) > 0:
                averageUtterancesRating = average_utterances_rating(line)
                humanScore[systemId[0]].append(averageUtterancesRating)

    for ids in humanScore:
        npRates = np.array(humanScore[ids])
        averageRating = npRates.mean()
        averageHumanRating[ids] = averageRating
    return averageHumanRating


def get_hypotheses(path):
    """
    get the hypotheses data from the hypotheses file path
    :param path: String type, hypotheses file path
    :return hypotheses: dict type, hypotheses data for each system 
    """
    fileList = os.listdir(path)
    hypotheses = dict()
    for ids in range(1, 21):
        systemId = 'S_' + str(ids)
        hypotheses[systemId] = list()

    for fname in fileList:
        if ".txt" in fname:
            systemId = re.findall(r'^S_\d+', fname)[0]
            file = os.path.join(path, fname)
            with open(file, 'r') as sh:
                for line in sh:
                    hypotheses[systemId].append(line)
    return hypotheses


def get_references(path):
    """
    get the references data from the references file path
    :param path: String type, references file path
    :return references: dict type, references data 
    """
    fileList = os.listdir(path)
    references = dict()
    referencesList = list()
    references['original_refs'] = list()
    for ids in range(1, 11):
        resultId = 'refgen_result' + str(ids)
        references[resultId] = list()

    for fname in fileList:
        if ".txt" in fname:
            resultId = re.split(r'\.', fname)[0]
            file = os.path.join(path, fname)
            with open(file, 'r') as sh:
                for line in sh:
                    references[resultId].append(line)
    for rid in references:
        referencesList.append(references[rid])
    return referencesList


def set_ref_weight(numberUtterances):
    """
    set the ref_weight for calculating the delta BLEU
    :param numberUtterances: int type, number of utterances for each systems
    :return normalBleuWeight: defualt None
    :return deltaBleuUniformWeight: list type, Uniform Weight for the ref_weight
    :return deltaBleuGlobalWeight: list type, Global Weight for the ref_weight
    """
    normalBleuWeight = None
    deltaBleuUniformWeight = [[1], [1], [1], [
        1], [1], [1], [1], [1], [1], [1], [1]]
    deltaBleuGlobalWeight = [[0.3], [0.9], [-0.2], [0.5],
                             [-0.8], [0.4], [0.4], [-0.1], [0], [0.7], [0]]
    pad = [0.3, 0.9, -0.2, 0.5, -0.8, 0.4, 0.4, -0.1, 0, 0.7, 0]
    for i in range(0, numberUtterances-1):
        for j in range(0, len(deltaBleuUniformWeight)):
            deltaBleuUniformWeight[j].append(1)
            deltaBleuGlobalWeight[j].append(pad[j])
    return normalBleuWeight, deltaBleuUniformWeight, deltaBleuGlobalWeight


def get_data(hPath, rPath):
    """
    get the hypotheses, references data from the file pathes
    :param hPath: String type, hypotheses file path
    :param rPath: String type, references file path
    :return hypotheses: dict type, hypotheses data 
    :return references: dict type, references data 
    """
    hypotheses = get_hypotheses(hPath)
    references = get_references(rPath)
    return hypotheses, references


def bleu(hypotheses, references):
    """
    calculate the bleu score for each system
    :param hypotheses: dict type, hypotheses data 
    :param references: dict type, references data 
    :return bleuScore: dict type, including BLER-4, BLEU-4 Uniformed,  BLEU-4 Global
    """
    bleuScore = dict()
    for ids in range(1, 21):
        systemId = 'S_' + str(ids)
        bleuScore[systemId] = list()

    numberUtterances = len(references[0])
    normalBleuWeight, deltaBleuUniformWeight, deltaBleuGlobalWeight = set_ref_weight(
        numberUtterances)
    for systemId in hypotheses:
        pred = hypotheses[systemId]
        normalBleu = sacrebleu.corpus_bleu(
            pred, references,force=True,ref_weights=normalBleuWeight)
        bleuScore[systemId].append(normalBleu)
        deltaBleuUniform = sacrebleu.corpus_bleu(
            pred, references, force=True,ref_weights=deltaBleuUniformWeight)
        bleuScore[systemId].append(deltaBleuUniform)
        deltaBleuGlobal = sacrebleu.corpus_bleu(
            pred, references, force=True,ref_weights=deltaBleuGlobalWeight)
        bleuScore[systemId].append(deltaBleuGlobal)
    return bleuScore


def rouge(hypotheses, references):
    """
    calculate the rouge score for each system
    :param hypotheses: dict type, hypotheses data 
    :param references: dict type, references data 
    :return rougeScore: dict type, including rouge-1, rouge-2, rouge-L
    """
    rougeScore = dict()
    for ids in range(1, 21):
        systemId = 'S_' + str(ids)
        rougeScore[systemId] = list()
    r_references = list(map(list, zip(*references)))

    rouge = PyRouge(rouge_n=2, rouge_l=True)
    for systemId in hypotheses:
        pred = hypotheses[systemId]
        scores = rouge.evaluate(pred, r_references)
        rougeScore[systemId].append(scores)
    return rougeScore


def n_gram(tokens, n):
    """
    return a n-grams tokenized list for the inputs
    :param tokens: list type, splited tokens for the input sentence
    :param n: int type, n for n-grams 
    :return nGram: list type, list for the n-grams 
    """
    nGram = list(nltk.ngrams(tokens, n, pad_right=True))
    return nGram


def sentence_distinct_n(sentence, n):
    """
    calculate the distinct_n score for the sentence level 
    :param sentence: sentence which needed to be calculate
    :param n: int type, n for distinct_n
    :return distinctN: float type, distinct_n score
    """
    tokens = sentence.split()
    distinctN = 0
    if len(tokens) == 0:
        distinctN = 0
    else:
        disNGrams = set(n_gram(tokens, n))
        distinctN = len(disNGrams)/len(tokens)
    return distinctN


def corpus_distinct_n(corpus, ns):
    """
    calculate the distinct_n score for the corpus level 
    :param corpus: hypotheses for each system
    :param ns: list type, all n for the distinct n
    :return corpusDistinctN: list type, all distinct_n scores for the corpus
    """
    corpusDistinctNDict = dict()
    corpusDistinctN = list()
    for n in ns:
        corpusDistinctNDict[n] = list()
    for sentence in corpus:
        for n in ns:
            sentenceDistinctN = sentence_distinct_n(sentence, n)
            corpusDistinctNDict[n].append(sentenceDistinctN)
    for n in corpusDistinctNDict:
        DistinctN = np.array(corpusDistinctNDict[n]).mean()
        corpusDistinctN.append(DistinctN)
    return corpusDistinctN


def distinct_n(hypotheses, ns):
    """
    calculate the distinct_n score for all system
    :param corpus: hypotheses for each system
    :param ns: list type, all n for the distinct n
    :return distinctN: dict type, all distinct_n scores for the each system
    """
    distinctN = dict()
    for ids in range(1, 21):
        systemId = 'S_' + str(ids)
        distinctN[systemId] = corpus_distinct_n(hypotheses[systemId], ns)
    return distinctN


def output_file(averageHumanRating, bleuScore, rougeScore, distinctN, file):
    """
    write all results to the output file
    :param averageHumanRating: dict type, average human rating for each system
    :param bleuScore: dict type, bleu scores for each system
    :param rougeScore: dict type, rouge scores for each system
    :param distinctN: dict type, distinct-n scores for  each system
    :param file: string type, output file path
    """
    firstLine = 'System,Averaged_Human_Rating,BLEU-4,deltaBLEU-4_Uniformed,deltaBLEU-4_Global,ROUGE-2,ROUGE-L,Distinct-1,Distinct-2,Distinct-3'
    with open(file, 'w') as outf:
        outf.write(firstLine+'\n')
        for systemId in averageHumanRating:
            outf.write(systemId+','+str(averageHumanRating[systemId])+',')
            if systemId in bleuScore:
                outf.write(str(bleuScore[systemId][0][0])+','+str(
                    bleuScore[systemId][1][0])+','+str(bleuScore[systemId][2][0])+',')
            else:
                outf.write('nan'+','+'nan'+','+'nan'+',')
            if systemId in rougeScore:
                outf.write(str(rougeScore[systemId][0]['rouge-2']['r']) +
                           ','+str(rougeScore[systemId][0]['rouge-l']['r'])+',')
            else:
                outf.write('nan'+','+'nan'+',')
            if systemId in distinctN:
                outf.write(str(distinctN[systemId][0])+','+str(
                    distinctN[systemId][1])+','+str(distinctN[systemId][2])+'\n')
            else:
                outf.write('nan'+','+'nan'+','+'nan'+'\n')


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
    # get all data
    humanRatingFile = os.path.join(inputs, 'human_rating_scores.txt')
    hypothesesPath = os.path.join(inputs, 'hypotheses')
    referencesPath = os.path.join(inputs, 'references')
    outputFile = os.path.join(inputs, 'output.csv')
    # calculate all metrics
    hypotheses, references = get_data(hypothesesPath, referencesPath)
    averageHumanRating = average_human_rating(humanRatingFile)
    bleuScore = bleu(hypotheses, references)
    rougeScore = rouge(hypotheses, references)
    distinctN = distinct_n(hypotheses, [1, 2, 3])
    # output to the file
    output_file(averageHumanRating, bleuScore,
                rougeScore, distinctN, outputFile)


if __name__ == '__main__':
    main()
