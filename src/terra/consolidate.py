import csv
import re

file_names = ['data/top_ten/conversations/aintyoubecky_conversation.csv', 'data/top_ten/conversations/arrogant_bubba_conversation.csv', 'data/top_ten/conversations/bricksquadmain_conversation.csv', 'data/top_ten/conversations/cashout063_conversation.csv', 'data/top_ten/conversations/dutchiedntshoot_conversation.csv', 'data/top_ten/conversations/fly_girllashea_conversation.csv', 'data/top_ten/conversations/lydiathebest__conversation.csv', 'data/top_ten/conversations/twdro_conversation.csv', 'data/top_ten/conversations/whynotchief_conversation.csv', 'data/top_ten/conversations/younggodumb_conversation.csv']

out_file = 'data/top_ten/conversations/all_no_emojis.csv'

w = open(out_file, 'wb')
writer = csv.DictWriter(w, ['AUTHOR', 'CONTENT', 'URL', 'DATE'])

writer.writeheader()
for file_name in file_names:
   	f = open(file_name, 'rU')
   	reader = csv.DictReader(f)
	for row in reader:
		if len(row['CONTENT']) < 1:
			continue
   		writer.writerow({'AUTHOR': row['AUTHOR'], 'CONTENT':row['CONTENT'], 'URL': row['ARTICLE_URL'], 'DATE': row['PUBLISH_DATE']})
   	f.close()	
   	writer.writerow({'AUTHOR': '', 'CONTENT':'', 'URL': '', 'DATE': ''})
