from pywsd.lesk import cosine_lesk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from math import ceil


def wsd(text, quary):
    
    sentences =  sent_tokenize(text)
    find_sent = ''
    tag = None

    for sent in sentences:
        if quary in sent:
            find_sent = sent
            break
    synonyms = []

    tag2tag = {
        'NN': 'n',
        'NNS': 'n',
        'RN': 'r',
        'VB': 'v',
        'VBP': 'v',
        'VBD': 'v',
        'VBZ': 'v',
        'VBG': 'v',
        'JJ' : 'a'
    }


    if find_sent != "":
        tags = pos_tag(word_tokenize(find_sent))
        tag = [x[1] for x in tags if x[0]==quary][0]
        try:
            tag = tag2tag[tag]
        except KeyError:
            tag = None
        
        answer = cosine_lesk(find_sent, quary, pos=None, context_is_lemmatized=True, nbest=True)
        for syn in wordnet.synsets(quary):
            for l in syn.lemmas():
                synonyms.append(l.name())
            synonyms = list(set(synonyms))

        print("Synonyms: {}".format(', '.join(synonyms)))
        print("The best definition: {}".format(answer[0][1].definition()))
        print()
        definitions ={}
        for ans in answer:
            definitions[ans[1].definition()] = ceil(ans[0]*100)/100
            #print("Definition: {0}, The similarity is {1}".format(ans[1].definition(), ans[0]))
        return ', '.join(synonyms), answer[0][1].definition(), definitions
    else:
        return ''
