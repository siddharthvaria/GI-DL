# -*- coding: utf-8 -*-

import csv

# author, content, date, label, url, desc

def unicode_csv_reader2(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {k:unicode(v, 'utf-8') for k, v in row.iteritems()}

fs = {"additional-data.csv":"add.csv"}

#
#for f in fs:
#    lines = open(f, 'r').readlines()
#    out = open(fs[f], 'w')
#    out.write("AUTHOR,CONTENT,DATE,LABEL,URL,DESC\n")
#    for line in lines[1:]:
#        splits = line.strip().split(',')
#        if len(splits) < 5:
#            print splits
#            continue
#        out.write(','.join([splits[2], splits[-3], "DATE", splits[-2], "URL", "DESC"]) + '\n')
#    out.close()
#
#exit(0)

for f in fs:
    i = 0
    
    reader = unicode_csv_reader2(open(f, 'r'), delimiter=',')
    out = open(fs[f], 'w')
    fieldnames = ['AUTHOR', 'CONTENT', 'DATE', 'LABEL', 'URL', 'DESC']
    writer = csv.DictWriter(out, fieldnames)
    writer.writeheader()
    
    for row in reader:
        #tweet_id, user_id, user_name, text, label
        label = row["collapsed_code"].strip().lower()
        l = 'Loss' if 'loss' in label else 'Aggression' if 'aggress' in label else 'Other'
        
        line = {'AUTHOR':row["screen_name"].encode('utf-8'), 'CONTENT':row["text"].encode('utf-8'), 'DATE':row["created_at"].encode('utf-8'), 'LABEL':l.encode('utf-8'), 'URL':row["url"].encode('utf-8'), 'DESC':"DESC".encode('utf-8')}

        writer.writerow(line)

        i += 1

    print i
        
    out.close()
