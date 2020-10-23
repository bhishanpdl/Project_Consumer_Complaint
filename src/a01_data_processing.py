
# load the path
import sys
sys.path.append('/Users/poudel/opt/miniconda3/envs/nlp/lib/python3.7/site-packages')

# load the libraries
import numpy as np
import pandas as pd
import time
import re
import string
from urllib.parse import urlparse
import multiprocessing as mp
import nltk
from nltk.corpus import stopwords

import unidecode
import wordninja

time_start = time.time()

# Load the data
df = pd.read_csv('../data/complaints_2019.csv.zip', compression='zip')

# Variables
target = 'product'
maincol = 'complaint'
mc = maincol + '_clean'
mcl = maincol + '_lst_clean'


# ==================== Useful functions ==============
def parallelize_dataframe(df, func):
    ncores = mp.cpu_count()
    df_split = np.array_split(df, ncores)
    pool = mp.Pool(ncores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
#================== Text processing =================
def process_text(text):
    """
    Do a basic text processing.

    Parameters
    -----------
    text : string

    Returns
    --------
    This function returns pandas series having one list
    with clean text.
    
    - decode unicode
    - lowercase
    - remove ellipsis
    - remove url
    - expand apostrophes
    - remove punctuation
    - remove digits
    - remove anonymous word x xx xxx etc
    - remove stop words
    - lemmatize

    Example:
    ========
    import re
    import string
    from nltk.corpus import stopwords
    import nltk
    
    text = "I'm typing text2num! areYou ? If yesyes say yes pals!"
    process_text(text)
    # ['typing', 'textnum', 'yes', 'say', 'yes', 'pal']

    """
    s = pd.Series([text])

    # step: decode unicode characters
    s = s.apply(unidecode.unidecode)
    
    # step: lowercase
    s = s.str.lower()
    
    # step: remove ellipsis
    #s = s.str.replace(r'(\w)\u2026+',r'\1',regex=True)
    s = s.str.replace(r'…+',r'')

    # step: remove url
    #s = s.str.replace('http\S+|www.\S+', '', case=False)
    s = pd.Series([' '.join(y for y in x.split() if not is_url(y)) for x in s])

    # step: expand apostrophes
    map_apos = {
        "you're": 'you are',
        "i'm": 'i am',
        "he's": 'he is',
        "she's": 'she is',
        "it's": 'it is',
        "they're": 'they are',
        "can't": 'can not',
        "couldn't": 'could not',
        "don't": 'do not',
        "don;t": 'do not',
        "didn't": 'did not',
        "doesn't": 'does not',
        "isn't": 'is not',
        "wasn't": 'was not',
        "aren't": 'are not',
        "weren't": 'were not',
        "won't": 'will not',
        "wouldn't": 'would not',
        "hasn't": 'has not',
        "haven't": 'have not',
        "what's": 'what is',
        "that's": 'that is',
    }

    sa = pd.Series(s.str.split()[0])
    sb = sa.map(map_apos).fillna(sa)
    sentence = sb.str.cat(sep=' ')
    s = pd.Series([sentence])

    # step: remove punctuation
    s = s.str.translate(str.maketrans(' ',' ',
                                        string.punctuation))
    # step: remove digits
    s = s.str.translate(str.maketrans(' ', ' ', '\n'))
    s = s.str.translate(str.maketrans(' ', ' ', string.digits))

    # step: remove xx xxx xxx etc
    s = s.str.replace(r'(\sx+)+\s', r' ', regex=True)

    # step: remove stop words
    stop = set(stopwords.words('English'))
    extra_stop_words = ['...']
    stop.update(extra_stop_words) # inplace operation
    s = s.str.split()
    s = s.apply(lambda x: [i for i in x if i not in stop])

    # step: convert word to base form or lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    s = s.apply(lambda lst: [lemmatizer.lemmatize(word) 
                               for word in lst])

    return s.to_numpy()[0]

def add_features(df):
    df[mcl] = df[maincol].apply(process_text)
    df[mc] = df[mcl].str.join(' ')

    return df

print("Creating clean data ...")
df = parallelize_dataframe(df, add_features)

#======================= Text Feature Generation =====
def create_text_features(df):
    # total
    df['total_length'] = df[maincol].apply(len)

    # num of word and sentence
    df['num_words'] = df[maincol].apply(lambda x: len(x.split()))

    df['num_sent']=df[maincol].apply(lambda x: 
                                len(re.findall("\n",str(x)))+1)

    df['num_unique_words'] = df[maincol].apply(
        lambda x: len(set(w for w in x.split())))

    # average
    df["avg_word_len"] = df[maincol].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    df['avg_unique'] = df['num_unique_words'] / df['num_words']
    
    return df

print("Adding Text features ...")
df = parallelize_dataframe(df, create_text_features)


#===================== Save clean data =========================
df.to_csv('../data/complaints_2019_clean.csv.zip',compression='zip', index=False)

time_taken = time.time() - time_start
m,s = divmod(time_taken,60)
print(f"Data cleaning finished in {m} min {s:.2f} sec.")
