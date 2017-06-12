import csv

def unicode_csv_reader1(utf8_data, **kwargs):
    csv_reader = csv.reader(utf8_data, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def unicode_csv_reader2(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {k:unicode(v, 'utf-8') for k, v in row.iteritems()}

def test_unicode_csv_reader():
    filename = '../data/csv_utf8_test.csv'
    reader = unicode_csv_reader2(open(filename))
    for line in reader:
        print len(line)
        print line

# test_unicode_csv_reader()
