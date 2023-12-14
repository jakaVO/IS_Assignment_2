import math
import functions
import json


file_path = 'News_Category_Dataset_IS_course.json'
i = 0
with open(file_path, 'r') as file:
    for line in file:
        #parsamo json
        data = json.loads(line)
              
        link = data.get('link')
        headline = data.get('headline')
        category = data.get('category')
        short_description = data.get('short_description')
        authors = data.get('authors')
        date = data.get('date')

        print()
        print(f"Link: {link}")
        print(f"Headline: {headline}")
        print(f"Category: {category}")
        print(f"Short Description: {short_description}")
        print(f"Authors: {authors}")
        print(f"Date: {date}")
        print("\n")
        i += 1
        if(i == 1):
            break