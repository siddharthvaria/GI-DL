import csv
import cStringIO
import codecs

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

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect = csv.excel, encoding = "utf8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect = dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf8") for s in row])
        # Fetch utf8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
