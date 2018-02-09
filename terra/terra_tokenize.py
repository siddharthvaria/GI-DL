__author__ = 'robertk'
import re
import sys
def space_emoji(data):
    if not data:
        return data
    if not isinstance(data, basestring):
        return data
    try:
    # UCS-4
        patt = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
    # UCS-2
        patt = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return patt.sub(lambda m: ' '+m.group()+' ', data)

def tweet_preprocessing(text, stopwords=[]):
	text = re.sub('(::emoji::)|#|', '', text.lower())
	text = re.sub('@[0-9a-zA-Z_]+', 'USER_HANDLE', text)
	text = re.sub('http://[a-zA-Z0-9_\./]*', 'URL', text)
	words = text.split(' ')
	if len(stopwords) > 0:
		words = [word.strip() for word in words if word not in stopwords]
	return text

def main():
    line = sys.stdin.readline()
    while line:
        text = space_emoji(line)
        text = tweet_preprocessing(text)
        sys.stdout.write(text)
        line = sys.stdin.readline()

if __name__ == "__main__":
    main()