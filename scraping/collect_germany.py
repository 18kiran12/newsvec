from gnews import GNews
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
#import country_list 
import pickle
import json
from utils.tools import daterange, week_range
#import requests
import re

output_path = "german_politics"

start_year = 2016
end_year = 2023

datetime_to_tuple = lambda x: (x.year, x.month, x.day)

weeks = week_range(start_year, end_year)

for start,end in weeks:
    google_news = GNews(language='en', 
                    country="Germany",
                    start_date = datetime_to_tuple(start),
                    end_date = datetime_to_tuple(end),
                    max_results = 15)
    json_resp = google_news.get_news("German Politics")
    text_content = []
    week_str = start.strftime('%Y') +"-"+ start.strftime('%W') 
    print(week_str, len(json_resp))
    count = 0 
    for json_entry in json_resp:
        article = google_news.get_full_article(json_entry['url'])
        try:
            article.nlp()
        except:
            pass
        if article:
            if all([article.text, article.title, article.summary, article.keywords]):
                text_content.append({"source": json_entry['publisher']['title'], 
                                     "date:": json_entry['publishing date'],
                                     "title": article.title, 
                                     "text": article.text, 
                                     "summary": article.summary,
                                     "keywords": article.keywords})
        count += 1
        if count > 10:
            break
    with open(f"{output_path}/{week_str}.json", "w") as outfile:
        json.dump(text_content, outfile)