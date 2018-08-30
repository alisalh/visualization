# -*- coding: utf-8 -*-

import os
import nltk
import string
import json
from nltk.corpus import wordnet as wn
from collections import defaultdict

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import Dictionary
from gensim.models import LdaModel

f = open('visualization/extraction/stopwords.txt', mode='r', encoding='utf-8')
english_stopwords = [line.strip() for line in f]
f.close()
english_stopwords.extend(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                          'october', 'november', 'december', 'film', 'movie'])


class EnPreprocess:

    def __init__(self):
        pass

    # 文件读取
    def file_read(self, filespath):
        # all_raw_data存放所有文件内容，[[document1],[document2]......]
        all_raw_data = []

        for files in os.walk(filespath):
            for file in files[2]:
                path = os.path.join(filespath, file)
                with open(path, mode='r', encoding='utf-8', errors='ignore') as f:
                    raw_data = f.read()
                all_raw_data.append(raw_data)
        return all_raw_data

    # 分割为句子
    def sent_token(self, all_raw_data):
        # all_clean_sents存放所有去除数字和符号后的句子
        all_clean_sents = []

        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for document in all_raw_data:
            sents = sent_tokenizer.tokenize(document)
            all_clean_sents.append([self.clean_non_alpha(sent) for sent in sents])
        return all_clean_sents

    # 去除数字和符号
    def clean_non_alpha(self, raw_sent):
        delchar = string.punctuation + string.digits
        identify = str.maketrans('', '', delchar)
        clean_sent = raw_sent.translate(identify)
        return clean_sent

    # 单句分割成单词
    def word_token(self, all_clean_sents):
        # all_clean_words存放所有单词
        all_clean_words = []

        for sents in all_clean_sents:
            words = []
            for sent in sents:
                words.extend(nltk.word_tokenize(sent))
            words = self.clean_stop_words(words)
            all_clean_words.append(words)
        return all_clean_words

    # 去掉停用词和长度小于5的单词
    def clean_stop_words(self, words):
        cleanwords = [word.lower() for word in words if word.lower() not in english_stopwords and len(word) >= 5]
        return cleanwords

    # 词干化
    def stem_words(self, all_clean_words):
        all_stem_words = []

        for clean_words in all_clean_words:
            words = []
            for clean_word in clean_words:
                stem_word = wn.morphy(clean_word)
                if stem_word is not None:
                    words.append(stem_word)
            words = self.clean_stop_words(words)
            all_stem_words.append(words)
        return all_stem_words

    # lda模型提取主题
    def lda_topics(self, all_stem_words):
        # frequency保存单词出现的次数
        frequency = defaultdict(int)

        for words in all_stem_words:
            for word in words:
                frequency[word] += 1
        all_stem_words = [[word for word in words if frequency[word] > 1]
                          for words in all_stem_words]
        dictionary = Dictionary(all_stem_words)
        corpus = [dictionary.doc2bow(words) for words in all_stem_words]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, iterations=500)
        return lda

    # 统计每篇文章中出现的topic次数
    def topics_counter(self, all_stem_words, topics_dict):
        topics_info = []
        for word in topics_dict:
            id = []
            count = []
            for words in all_stem_words:
                if word in words:
                    i = all_stem_words.index(words)
                    id.append(i)
                    n = words.count(word)
                    count.append(n)
            dict_info = {'w': word, 'doc': id, 'count': count}
            topics_info.append(dict_info)

        f = open('result.json', mode='w', encoding='utf-8', errors='ignore')
        topics_info_json = json.dumps(topics_info)
        f.write(topics_info_json)
        f.close()

    def enpre_main(self, filepath):
        # if os.path.exists("allLDA50Topic.mdl"):
        #     lda = LdaModel.load("allLDA50Topic.mdl")
        # else:
        #     all_raw_data = self.file_read(filepath)
        #     all_clean_sents = self.sent_token(all_raw_data)
        #     all_clean_words = self.word_token(all_clean_sents)
        #     all_stem_words = self.stem_words(all_clean_words)
        #     lda = self.lda_topics(all_stem_words)
        #     lda.save("allLDA50Topic.mdl")

        # lda提取主题：word-probability
        lda = LdaModel.load('visualization/extraction/allLDA50Topic.mdl')
        topics_prob = {}
        for topic in lda.print_topics(50, 10):
            topic_list = topic[1].split('+')
            for each in topic_list:
                group = each.split('*')
                p = float(group[0])
                w = eval(group[1])
                if w not in topics_prob:
                    topics_prob[w] = p
                else:
                    topics_prob[w] += p

        # all_raw_data = self.file_read(filepath)
        # all_clean_sents = self.sent_token(all_raw_data)
        # all_clean_words = self.word_token(all_clean_sents)
        # all_stem_words = self.stem_words(all_clean_words)
        #
        # self.topics_counter(all_stem_words, topics_dict)

        # 统计词频：word-doc-count
        with open('visualization/extraction/result.json', 'r') as f:
            topics_info = json.load(f)

        # 获取文件名
        for files in os.walk(filepath):
            files_list = files[2]
        files_name_list = []
        for file in files_list:
            files_name_list.append(os.path.splitext(file)[0])

        return topics_prob, topics_info, files_name_list
