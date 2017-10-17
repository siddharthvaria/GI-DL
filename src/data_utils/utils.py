import csv

def unicode_csv_reader1(utf8_data, **kwargs):
    csv_reader = csv.reader(utf8_data, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

# To represent a unicode string as a string of bytes is known as encoding
# To convert a string of bytes to a unicode string is known as decoding.
# Use unicode('...', encoding) or '...'.decode(encoding)
def unicode_csv_reader2(utf8_data, encoding, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {k:unicode(v, encoding) for k, v in row.iteritems()}

def test_unicode_csv_reader():
    filename = '../data/csv_utf8_test.csv'
    reader = unicode_csv_reader2(open(filename))
    for line in reader:
        print len(line)
        print line
