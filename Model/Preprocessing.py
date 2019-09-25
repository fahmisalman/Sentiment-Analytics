import re


def preprocessing(text):
    text = re.sub('\n', '. ', text)
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    return text
