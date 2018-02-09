__author__ = 'robertk'

def readParsed(filePath, upTo, startAt=0):
    fr = open(filePath, 'r')
    if upTo < 0:
        upTo = 99999999999999999
    parsed = []
    i = startAt
    for j in range(i):
        fr.readline()
    line = fr.readline()
    while line and i < startAt+upTo:
        #print i+1
        lineSplit = line.split()
        tags = []
        for j in range(len(lineSplit)):
            tags.append(lineSplit[j].split('\\')[1])
        parsed.append(tags)
        line = fr.readline()
        i += 1
    return parsed

def get_kappa(test, eval):
    from nltk.metrics.agreement import AnnotationTask
    data = []
    counter = 0
    for i in range(100):
        #print i
        e_sent = eval[i]
        g_sent = test[i]
        for j in range(max(len(e_sent), len(g_sent))):
            e_tag = e_sent[j]
            g_tag = g_sent[j]

            data.append( ('1', counter, e_tag) )
            data.append( ('2', counter, g_tag) )

            counter+=1

    t = AnnotationTask(data=data)
    k = t.kappa()
    return k

def compare(test, eval):
    tags_correct = 0
    tags_incorrect = 0
    lines_correct = 0
    lines_incorrect = 0

    for i in range(len(eval)):
        correctHere = 0
        incorrectHere = 0
        #print i+1
        #print test
        for j in range(max(len(eval[i]), len(test[i]))):
            #if (eval[i][j] != 'E' and test[i][j] != 'E') and (eval[i][j] != 'G' and test[i][j] != 'G') \
            #        and (eval[i][j] != '@' and test[i][j] != '@') and (eval[i][j] != '#' and test[i][j] != '#')\
            #        and (eval[i][j] != 'L' and test[i][j] != 'L') and (eval[i][j] != 'Y' and test[i][j] != 'Y')\
            #        and (eval[i][j] != 'M' and test[i][j] != 'M') and (eval[i][j] != 'T' and test[i][j] != 'T')\
            #        and (eval[i][j] != 'Z' and test[i][j] != 'Z') and (eval[i][j] != 'U' and test[i][j] != 'U'):
            #if (eval[i][j] != 'G' and test[i][j] != 'G'):
            if True:
                if 1 == 1:
                    if test[i][j] == eval[i][j]:
                        correctHere += 1
                    else:
                        incorrectHere += 1
        tags_correct += correctHere
        tags_incorrect += incorrectHere
        if incorrectHere == 0:
            lines_correct += 1
        else:
            lines_incorrect += 1

    return tags_correct, tags_incorrect, lines_correct, lines_incorrect

def testDevAccuracy(numExamples, gold_path, test_path, startAt=0):
    test = readParsed(test_path, numExamples, startAt=startAt)
    mine = readParsed(gold_path, numExamples, startAt=startAt)
    tags_correct, tags_incorrect, lines_correct, lines_incorrect = compare(test, mine)
    k = get_kappa(test, mine)
    print(test_path+" Dev Tags Correct: %.2f%%" % (float(100.0*tags_correct)/float(tags_correct+tags_incorrect)))
    print k

def testTrainAccuracy(numExamples, gold_path, test_path):
    test = readParsed(test_path, numExamples, startAt=100)
    mine = readParsed(gold_path, numExamples, startAt=100)
    tags_correct, tags_incorrect, lines_correct, lines_incorrect = compare(test, mine)
    k = get_kappa(test, mine)
    print(test_path+" Train Tags Correct: %.2f%%" % (float(100.0*tags_correct)/float(tags_correct+tags_incorrect)))
    print k


print
print
print 'CMU'
testDevAccuracy(-1, '../data/gold/cmu_test_gold.txt', '../results/cmu_pos_tagged_cv.txt')
testDevAccuracy(-1, '../data/gold/cmu_test_gold.txt', '../results/cmu_pos_tagged.txt')
testDevAccuracy(-1, '../data/gold/cmu_test_gold.txt', '../results/cmu_pred_cmu.txt')
testDevAccuracy(-1, '../data/gold/cmu_test_gold.txt', '../results/stanford_cmu_tagged.txt')

print
print 'New Dev'
testDevAccuracy(102, '../data/gold/dev_final.txt', '../results/pos_tagged_cv.txt')

testDevAccuracy(102, '../data/gold/dev_final.txt', '../results/pos_tagged.txt')
testDevAccuracy(102, '../data/gold/dev_final.txt', '../results/revised_cmu_simple_emoji.txt')
testDevAccuracy(102, '../data/gold/dev_final.txt', '../results/stanford_tags_simple_emoji.txt')
print
print 'Test Revised'
testDevAccuracy(-1, '../data/gold/test_final_revised.txt', '../results/pos_tagged_test_cv.txt')

testDevAccuracy(-1, '../data/gold/test_final_revised.txt', '../results/pos_tagged_test.txt')
testDevAccuracy(-1, '../data/gold/test_final_revised.txt', '../results/cmu_test_set_revised.txt')
testDevAccuracy(-1, '../data/gold/test_final_revised.txt', '../results/stanford_test_set.txt')
print
print 'My Dev'
testDevAccuracy(102, '../data/gold/simple_gold_revised_emojis.txt', '../results/pos_tagged_cv.txt')

testDevAccuracy(102, '../data/gold/simple_gold_revised_emojis.txt', '../results/pos_tagged.txt')
testDevAccuracy(102, '../data/gold/simple_gold_revised_emojis.txt', '../results/revised_cmu_simple_emoji.txt')
testDevAccuracy(102, '../data/gold/simple_gold_revised_emojis.txt', '../results/stanford_tags_simple_emoji.txt')
print
print 'Agreement'
testDevAccuracy(102, '../data/gold/simple_gold_revised_emojis.txt', '../data/gold/dev_final.txt')
print
print 'CV'
testDevAccuracy(-1, '../data/gold/simple_gold_revised_emojis.txt', '../results/pos_tagged_4_fold_cv.txt')

#testDevAccuracy(100, '../data/gold/test_final.txt', '../results/test_pos_tagged.txt')

#for i in range(1, 101):
#    testDevAccuracy(i, '../data/gold/simple_gold_revised.txt', '../data/gold/terra_POS_content_joined_modified.txt')
