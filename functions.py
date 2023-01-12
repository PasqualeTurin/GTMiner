import os
from os import listdir
from os.path import isfile, join
from io import open
import torch
import sys
import numpy as np
import time
import math
from math import sin, cos, sqrt, atan2, radians
import json
import pandas as pd
from ast import literal_eval
import config
import wget
import zipfile
import random
import re
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
    
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def get_position(entity):

    words = entity.lower().split()
    latitude = 0
    longitude = 0
    idx = 0
    for i, word in enumerate(words):
        if words[i-3] == 'col' and words[i-2] == 'latitude' and words[i-1] == 'val':
            latitude = float(word)
            longitude = float(words[i+4])
            idx = i

    del words[idx-3:idx+5]

    return ' '.join(words), latitude, longitude


def drop_id(entity):

    words = entity.lower().split()
    idx = 0
    id = ''
    for i, word in enumerate(words):
        if words[i-3] == 'col' and words[i-2] == 'id' and words[i-1] == 'val':
            idx = i
            id = word

    del words[idx-3:]

    return ' '.join(words), id


def textualize_block(row, cols):

    text = ''
    for i, col in enumerate(cols):
        if i != len(cols) - 1:
            text += 'COL ' + str(cols[i].split('_')[1]) + ' VAL ' + str(row[i]) + ' '

    return text


def tokenize_block(row, cols):

    toks = []
    for i, col in enumerate(cols):
        if i != len(cols) - 1:
            for tok in str(row[i]).split():
                toks.append(tok.replace('_', ' '))

    return toks



def textualize(row, cols):

    text_h = ''
    text_t = ''

    lat_h = ''
    lat_t = ''

    lon_h = ''
    lon_t = ''

    id_h = ''
    id_t =''

    for i, col in enumerate(cols):

        if i != len(cols) - 1:

            if cols[i].split('_')[0] == 'h':

                if cols[i].split('_')[1] == 'id':
                    id_h = str(row[i])
                elif cols[i].split('_')[1] == 'latitude':
                    lat_h = str(row[i])
                elif cols[i].split('_')[1] == 'longitude':
                    lon_h = str(row[i])
                else:
                    text_h += 'COL ' + str(cols[i].split('_')[1]) + ' VAL ' + str(row[i]) + ' '
      
            else:

                if cols[i].split('_')[1] == 'id':
                    id_t = str(row[i])
                elif cols[i].split('_')[1] == 'latitude':
                    lat_t = str(row[i])
                elif cols[i].split('_')[1] == 'longitude':
                    lon_t = str(row[i])
                else:
                    text_t += 'COL ' + str(cols[i].split('_')[1]) + ' VAL ' + str(row[i]) + ' '

    return id_h, lat_h, lon_h, text_h, id_t, lat_t, lon_t, text_t


def compute_dist(lat1, lon1, lat2, lon2):

    R = 6373.0
    
    try:
        float(lat1)
    except ValueError:
        return ' '
        
    try:
        float(lon1)
    except ValueError:
        return ' '
        
    try:
        float(lat2)
    except ValueError:
        return ' '
        
    try:
        float(lon2)
    except ValueError:
        return ' '

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
        
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))
                
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return str(round(R * c * 1000))


def norm_d(x):
    return -2*x/config.max_d_filter + 1


def yelp_address(entity):
    if 'location' in entity:
        if 'display_address' in entity['location']:
            return ' '.join((' '+' '.join(entity['location']['display_address'])).split())

    return ''


def osm_address(entity):
    address = ''
    valid = False

    if 'addr:housenumber' in entity['tags']:
        address += ' '+entity['tags']['addr:housenumber']
    if 'addr:street' in entity['tags']:
        valid = True
        address += ' '+entity['tags']['addr:street']
    if 'addr:unit' in entity['tags']:
        address += ' '+entity['tags']['addr:unit']
    if 'branch' in entity['tags']:
        valid = True
        address += ' '+entity['tags']['branch']
    if 'addr:postcode' in entity['tags']:
        valid = True
        address += ' '+entity['tags']['addr:postcode']

    if valid:
        return ' '.join(address.split())
    else:
        return ''


def osm_name(entity):
    name = ''
    if 'name' in entity['tags']:
        name = entity['tags']['name']

    if 'name:en' in entity['tags']:
        name = entity['tags']['name:en']
    else:
        zh = re.findall(r'[\u4e00-\u9fff]+', name)
        if zh:
            for zh_word in zh:
                name = name.replace(zh_word, '')
            name = name.replace('(', '').replace(')', '')

    return ' '.join(name.split())


class CountDictionary(object):
    def __init__(self):
        self.word2count = {}

    def add_word(self, word):
        if word not in self.word2count.keys():
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1


def load_glove_model():
    print("Loading Glove Model")
    glove_model = {}
    file_path = config.glove_folder+config.glove_file+str(config.glove_size)+'d'+config.path_suffix
    if not os.path.isfile(file_path):
        print("Downloading Glove Embeddings...")
        _ = wget.download(config.glove_url, out=config.glove_folder)
        with zipfile.ZipFile(config.glove_folder+config.glove_zip, 'r') as zip_ref:
            print("Unzipping Glove Embeddings...")
            zip_ref.extractall(config.glove_folder)

    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model
