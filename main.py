import nltk.tag
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.chunk import ne_chunk

inputText = "man the People Republic of China is so cool!"


def preprosses(text):

   #normalize text
   tokenizer = RegexpTokenizer(r'\w+')
   noPuct = tokenizer.tokenize(text)
   




# tokenize

   senteces = nltk.sent_tokenize(text)

   

   tokens = tokenizer.tokenize(text)

   tokens = [token.lower() for token in tokens]
    
   #processed_words = tokens.lower()
   print(tokens)

   
   return tokens


def pos_tagging(tokens):
    tagged_tokens = pos_tag(tokens)
    print("POS Tagged Tokens:", tagged_tokens)
    return tagged_tokens


def named_entity_recognition(tagged_sentences):
    ner_sentences = [ne_chunk(sent) for sent in tagged_sentences]
    print("NER:",ner_sentences)
    return ner_sentences



def extract_entities(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    
    entities = []
    for subtree in named_entities:
        if isinstance(subtree, nltk.Tree):
            entity = " ".join([word for word, tag in subtree.leaves()])
            entities.append((subtree.label(), entity))
    
    return entities

# Example usage
sentence = preprosses(inputText)
ps = pos_tagging(sentence)
extract_entities(ps)
entities = extract_entities(sentence)
print("Named Entities:", entities)
