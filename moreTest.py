from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer(r'\w+')
print (tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!'))

