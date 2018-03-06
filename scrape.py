from requests import get 
from bs4 import BeautifulSoup
import csv

def fetch_summary(movie_id):
    url = "http://www.imdb.com/title/" + movie_id
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    summary = html_soup.find("div", {"class" : "summary_text"})

    if not summary or not summary.text:
        return None
    
    text = summary.text.lstrip().rstrip()
    if text.startswith("Add a Plot"):
        return None
    return text

with open('./subset1.csv', 'r') as f:
    rows = csv.reader(f, delimiter = ",") 
    for row in rows:
        movie_id = row[0]
        summary = fetch_summary(movie_id)
        print(movie_id)
        print(summary)
