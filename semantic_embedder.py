from csv import excel_tab
from sentence_transformers import SentenceTransformer
import json
import pdb
import re
import numpy as np
import pickle
from datetime import datetime

def get_embedding(model, text):
    embedding = model.encode(text)
    return embedding

def create_embeddings(news_file_path):
    # Open the JSON file
    data = None
    model = SentenceTransformer('./all-MiniLM-L6-v2')
    with open(news_file_path) as f:
        # Load JSON data from file
        data = json.load(f)
    date_embeddings = {}
    for item in data:
        # date_embeddings["date"] = item[0]
        embeddings =[]
        for heads in item[4]: # already has multiple 
            # for head in heads[3]:
            #     try:
            head = re.sub('[^A-Za-z0-9\s]+', '', heads[1])
            embedding = get_embedding(model, head)
            embeddings.append(embedding)
            # except:
        # date to required format
        date_object = datetime.strptime(item[0], '%d/%m/%Y')
        # pdb.set_trace()
        converted_date = date_object.strftime('%Y-%m-%d')
        date_embeddings[converted_date] = np.array(embeddings)

    # save the data
    with open('date_embeddings.pickle', 'wb') as file:
        pickle.dump(date_embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)
    return date_embeddings

# def main():
#     create_embeddings("./dataset/COVID_India.json")
    
#     # embeddings = model.encode(sentences)

# if __name__=="__main__":
#     main()

