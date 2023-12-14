import requests
from bs4 import BeautifulSoup

def scrape_website_text(url):
    try:
        response = requests.get(url)

        # check ce gre kej narobe
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract all text content from the HTML
            text_content = soup.get_text()

            return text_content
        else:
            print(f"Failed to retrieve, status: {response.status_code}")
    except Exception as e:
        print(f"error : {e}")

#primer 
""" website_url = "https://www.huffpost.com/entry/funniest-parenting-tweets_l_632d7d15e4b0d12b5403e479"
result = scrape_website_text(website_url)

if result:
    print(result) """

