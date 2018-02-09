import json
import csv

data = open('/proj/nlp/users/terra/streaming_results/twitter_stream_result.json', 'rb')
writer = csv.DictWriter(open('/proj/nlp/users/terra/streaming_results/twitter_stream_result.csv', 'wb'), ['ID', 'AUTHOR', 'CONTENT', 'DATE'])
writer.writeheader()

index = 1
for line in data.readlines():
    tweet_data = json.loads(line)
    print str(index)+') '+tweet_data['text'].encode('utf8')
#    print tweet_data['geo']
#    print
    index += 1

    d = {'ID': tweet_data['id'], 'AUTHOR': tweet_data['user']['screen_name'], 'CONTENT': tweet_data['text'].encode('utf8'), 'DATE': tweet_data['created_at']}
    writer.writerow(d)
