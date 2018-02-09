__author__ = 'robertk'


def read_phrase(phrase_path, thresh=0.33):
    if phrase_path is None:
        return None
    phrase_table = dict()
    probs = dict()
    phrase_file = open(phrase_path)
    phrase_lines = phrase_file.readlines()
    for line in phrase_lines:
        data = line.strip().split('\t')
        data[4] = float(data[4])
        if data[0] not in phrase_table:
            if data[4] > thresh:
                phrase_table[data[0]] = data[2]
                probs[data[0]] = data[4]
        else:
            if data[4] > probs[data[0]]:
                phrase_table[data[0]] = data[2]
                probs[data[0]] = data[4]
    return phrase_table


def read_dict(dict_path):
    if dict_path is None:
        return None
    dictionary = dict()
    dict_file = open(dict_path)
    dict_lines = dict_file.readlines()
    for line in dict_lines:
        data = line.strip().split('\t')
        dictionary[data[0]] = data[1]
    return dictionary


def table_translate_sent(line, phrase, dictionary=None):
    out = ''
    tokens = line.split()
    for token in tokens:
        token = token.lower()
        if token in phrase:
            out += phrase[token]
        elif dictionary is not None and token in dictionary:
            out += dictionary[token]
        else:
            out += token
        out += ' '
    return out


def translate(input_path, phrase_path, dict_path=None):
    input_file = open(input_path)
    phrase_table = read_phrase(phrase_path)
    dictionary = read_dict(dict_path)
    input_lines = input_file.readlines()
    output_lines = []
    for line in input_lines:
        out = table_translate_sent(line, phrase_table, dictionary)
        output_lines.append(out)
    return output_lines


def main():
    import sys
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Error: Missing arguments')
        print('Proper use: python2 translate_tweets.py input_path phrase_path [dict_path]')

    file_path = sys.argv[1]
    phrase_path = sys.argv[2]
    dict_path = None
    if len(sys.argv) == 4:
        dict_path = sys.argv[3]
    output = translate(file_path, phrase_path, dict_path)
    for line in output:
        print(line)


if __name__ == "__main__":
    main()