"""Functions for preprocessing text data.

Download the dataset (see run_tv_shows.sh) before run this script.

Author: Lang Liu
Date: 06/10/2019
"""

import os
import re
import unicodedata

import contractions
import inflect
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
import pysrt


##########################################################################
# files I/O
##########################################################################


def _read_lines(path):
    with open(path, 'r') as f:
        out = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    out = [x.strip() for x in out]
    return out


def _save_lines(text, path):
    with open(path, 'w+') as f:
        for item in text:
            f.write(f"{item}\n")


##########################################################################
# normalization
##########################################################################


def strip_html(text):
    """Remove HTML marks."""
    text = text.replace("<i>", '')
    text = text.replace("</i>", '')
    return text


def replace_contractions(text):
    """Replace contractions in the text."""
    text = text.replace("ain't", 'are not')
    text = text.replace("in'", 'ing')
    text = text.replace("\'cause", "because")
    return contractions.fix(text)


def remove_non_ascii(text):
    """Remove non-ASCII characters in the text."""
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


##########################################################################
# the following functions need to be called after tokenization
# perform to_lowercase() first
##########################################################################


def remove_stopwords(words):
    stop_words = _read_lines('stopwords.txt')
    return [word for word in words if word not in stop_words]


def _convert_ordinal_num(word):
    """Convert ordinal numbers to cardinal numbers."""
    ths = re.findall(r'\d+st|\d+nd|\d+rd|\d+th', word)  # find -st, -nd, -rd, -th
    for th in ths:
        num = th[:(len(th) - 2)]
        word = re.sub(th, num, word)
    return word


def _my_num_to_words(word):
    """Convert numbers to their words equivalent."""
    p = inflect.engine()
    nums = p.number_to_words(word)
    nums = nums.replace('-', ' ')
    nums = remove_punctuation(nums)
    nums = nltk.word_tokenize(nums)
    nums = remove_stopwords(nums)
    return nums


def replace_numbers(words):
    """Replace numbers with their words equivalent."""
    new_words = []
    for word in words:
        if word.isdigit():  # '123'
            nums = _my_num_to_words(word)
            new_words += nums
        elif word[0].isdigit():  # '24inches'
            word = _convert_ordinal_num(word)
            seps = re.split(r'(\d+)', word)
            for s in seps:
                if s.isdigit():
                    nums = _my_num_to_words(s)
                    new_words += nums
                elif s != '':
                    new_words.append(s)
        else:
            new_words.append(word)
    return new_words


def remove_numbers(words):
    return [re.sub(r'\d+', '', word) for word in words if not word[0].isdigit() and not word[-1].isdigit()]


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]


def remove_names(words):
    st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')
    tags = st.tag(words)
    new_words = [word for (word, tag) in tags if tag != 'PERSON']
    return new_words


def to_lowercase(words):
    return [word.lower() for word in words]


##########################################################################
# preprocessing
##########################################################################


def preprocess(text):
    """Preprocesse text data.

    Including removing HTML marks, replacing contractions with their original
    forms, removing non-ASCII characters, removing punctuations, tokenizing text
    into words, remove person names, converting to lower case letters, replacing numbers with
    their words equivalent, removing stopwords, and lemmatizing verbs.
    """
    text = strip_html(text)
    text = replace_contractions(text)
    text = remove_non_ascii(text)
    text = text.replace('-', ' ')
    text = remove_punctuation(text)
    words = nltk.word_tokenize(text)  # tokenization
    words = remove_names(words)
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words


if __name__ == '__main__':
    read_path = 'tv_subtitles'
    write_path = 'processed_subtitles'
    if not os.path.exists(read_path):
        raise RuntimeError('Download the dataset first.')
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    print("preprocessing")
    for s in [1, 2]:
        print("Season {}".format(s))
        season = []
        print("friends")
        for e in range(1, 25):
            file_name = read_path + '/Friends_S' + str(s) + 'E' + str(e) + '.srt'
            try:
                subs = pysrt.open(file_name)
            except:
                subs = pysrt.open(file_name, encoding='iso-8859-1')
            if e == 1:
                sents = [sub.text for sub in subs if sub.index not in [1, 2, 9999]]
            else:
                sents = [sub.text for sub in subs if sub.index != 9999]
            text = ' '.join(sents)
            after = preprocess(text)
            season += after
            _save_lines(after, write_path + '/friends_S' + str(s) + 'E' + str(e) + '.txt')
        _save_lines(season, write_path + '/friends_S' + str(s) + '.txt')

        season = []
        print("modern")
        for e in range(1, 25):
            file_name = read_path + '/modern_S' + str(s) + 'E' + str(e) + '.srt'
            try:
                subs = pysrt.open(file_name)
            except:
                subs = pysrt.open(file_name, encoding='iso-8859-1')
            if e == 1:
                sents = [sub.text for sub in subs if sub.index not in [1, 2, 9999]]
            else:
                sents = [sub.text for sub in subs if sub.index != 9999]
            text = ' '.join(sents)
            after = preprocess(text)
            season += after
            _save_lines(after, write_path + '/modern_S' + str(s) + 'E' + str(e) + '.txt')
        _save_lines(season, write_path + '/modern_S' + str(s) + '.txt')

        season = []
        print("soprano")
        for e in range(1, 14):
            file_name = read_path + '/the_sopranos_S' + str(s) + 'E' + str(e) + '.srt'
            try:
                subs = pysrt.open(file_name)
            except:
                subs = pysrt.open(file_name, encoding='iso-8859-1')
            sents = [sub.text for sub in subs if sub.index != 9999]
            text = ' '.join(sents)
            after = preprocess(text)
            season += after
            _save_lines(after, write_path + '/the_sopranos_S' + str(s) + 'E' + str(e) + '.txt')
        _save_lines(season, write_path + '/the_sopranos_S' + str(s) + '.txt')

        season = []
        print("deadwood")
        for e in range(1, 12+s):  # 13 when s = 1; 14 when s = 2
            file_name = read_path + '/deadwood_S' + str(s) + 'E' + str(e) + '.srt'
            try:
                subs = pysrt.open(file_name)
            except:
                subs = pysrt.open(file_name, encoding='iso-8859-1')
            sents = [sub.text for sub in subs if sub.index != 9999]
            text = ' '.join(sents)
            after = preprocess(text)
            season += after
            _save_lines(after, write_path + '/deadwood_S' + str(s) + 'E' + str(e) + '.txt')
        _save_lines(season, write_path + '/deadwood_S' + str(s) + '.txt')
