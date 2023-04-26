import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
import json 

with open("../games.json") as f:
    json_data = json.load(f)
data = pd.DataFrame({"Game": [json_data[i]["Title"] for i in range(len(json_data))], 
                     "Review": [json_data[i]["Reviews"] for i in range(len(json_data))]})

print(data["Review"].dtype)
# Function to preprocess Reviews data
def preprocess_reviews_data(data,name):
    # Proprocessing the data
    data[name]=data[name].str.lower()
    # Code to remove the links from the text
    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    # Code to remove the Special characters from the text 
    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Code to substitute the multiple spaces with single spaces
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Code to remove all the single characters in the text
    data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))

# Function to tokenize and remove the stopwords    
def rem_stopwords_tokenize(data,name):
      
    def getting(sen):
        example_sent = sen
        
        filtered_sentence = [] 

        stop_words = set(stopwords.words('english')) 

        word_tokens = word_tokenize(example_sent) 
        
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        
        return filtered_sentence
    # Using "getting(sen)" function to append edited sentence to data
    x=[]
    for i in data[name].values:
        x.append(getting(i))
    data[name]=x

lemmatizer = WordNetLemmatizer()
def Lemmatization(data,name):
    def getting2(sen):
        
        example = sen
        output_sentence =[]
        word_tokens2 = word_tokenize(example)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens2]
        
        # Remove characters which have length less than 2  
        without_single_chr = [word for word in lemmatized_output if len(word) > 2]
        # Remove numbers
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        
        return cleaned_data_title
    # Using "getting2(sen)" function to append edited sentence to data
    x=[]
    for i in data[name].values:
        x.append(getting2(i))
    data[name]=x

def make_sentences(data,name):
    data[name]=data[name].apply(lambda x:' '.join([i+' ' for i in x]))
    # Removing double spaces if created
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))


# Using the preprocessing function to preprocess the hotel data
preprocess_reviews_data(data,'Review')
# # Using tokenizer and removing the stopwords
# rem_stopwords_tokenize(data,'Review_without_stopwords')
# # Converting all the texts back to sentences
# make_sentences(data,'Review_without_stopwords')

# #Edits After Lemmatization
# final_Edit = data['Review_without_stopwords'].copy()
# data["After_lemmatization"] = final_Edit

# # Using the Lemmatization function to lemmatize the hotel data
# Lemmatization(data,'After_lemmatization')
# # Converting all the texts back to sentences
# make_sentences(data,'After_lemmatization')