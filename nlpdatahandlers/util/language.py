try:
    from spacy.en import English
except ImportError:
    print '[!] You need to install spaCy! Visit spacy.io/#install'

# Spacy.en provides a faster tokenizer than nltk
nlp = English()

def parse_paragraph(txt):
    """
    Takes a text and returns a list of lists of tokens, where each sublist is a sentence
    """
    sentences = nlp(u'' + txt.decode('ascii', errors='ignore')).sents
    return [[t.text for t in s] for s in sentences]

def tokenize_text(text):
    """
    Gets tokens from a text in English
    """
    if isinstance(text, unicode):
        tokens = [token.lower_ for token in nlp(text)]
    else:
        tokens = [token.lower_ for token in nlp(u'' + text.decode('ascii', errors='ignore'))]

    return tokens