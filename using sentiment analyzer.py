import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

train_dataset = pd.read_csv ('train.csv')
test_dataset = pd.read_csv ('test.csv')

#training set
data=train_dataset['text']
sentiment=train_dataset['sentiment']
processed_text=[]
s_analyser=SentimentIntensityAnalyzer()
for j in range(0,len(data)):
    text = re.sub(r'http\S+', '', str(data.iloc[j]))
    if(sentiment.iloc[j] == "neutral" or len(text.split()) < 2):
        processed_text.append(str(text))
    if(sentiment.iloc[j] == "positive" and len(text.split()) >= 2):
        words=re.split(' ', text)
        array=""
        polar=0
        for word in range(0,len(words)):
            score = s_analyser.polarity_scores(words[word])
            if score['compound'] >polar:
                polar = score['compound']
                array = words[word]
        if len(array) != 0:
            processed_text.append(array)   
        if len(array) == 0:
            processed_text.append(text)
    if(sentiment.iloc[j] == "negative"and len(text.split()) >= 2):
        words= re.split(' ', text)
        array=""
        polar=0
        for word in range(0,len(words)):
            score = s_analyser.polarity_scores(words[word])
            if score['compound'] <polar:
                polar = score['compound']
                array = words[word]
        if len(array) != 0:
            processed_text.append(array)   
        if len(array) == 0:
            processed_text.append(text)

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

average = 0;
for i in range(0,len(data)):
    ja_s = jaccard(str(processed_text[i]),str(data[i]))
    average = ja_s+average
    
print('Training Data accuracey')
print(average/len(processed_text))

#testset
test_data=test_dataset['text']
sentiment=test_dataset['sentiment']
processed_text=[]
s_analyser=SentimentIntensityAnalyzer()
for j in range(0,len(test_data)):
    text = re.sub(r'http\S+', '', str(data.iloc[j]))
    if(sentiment.iloc[j] == "neutral" or len(text.split()) < 2):
        processed_text.append(str(text))
    if(sentiment.iloc[j] == "positive" and len(text.split()) >= 2):
        words=re.split(' ', text)
        array=""
        polar=0
        for word in range(0,len(words)):
            score = s_analyser.polarity_scores(words[word])
            if score['compound'] >polar:
                polar = score['compound']
                array = words[word]
        if len(array) != 0:
            processed_text.append(array)   
        if len(array) == 0:
            processed_text.append(text)
    if(sentiment.iloc[j] == "negative"and len(text.split()) >= 2):
        words= re.split(' ', text)
        array=""
        polar=0
        for word in range(0,len(words)):
            score = s_analyser.polarity_scores(words[word])
            if score['compound'] <polar:
                polar = score['compound']
                array = words[word]
        if len(array) != 0:
            processed_text.append(array)   
        if len(array) == 0:
            processed_text.append(text)
                
textid = test_dataset['textID']
text_id_list = []
for i in range(0,len(textid)):
    text_id_list.append(textid.iloc[i])
write = pd.DataFrame({'textID':text_id_list,'selected_text':processed_text})
write.to_csv('submission.csv',index=False)

