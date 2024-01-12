# import math
from functions import *
import json


#s tem dobis count kerih categoryjev je najmajn v unem vlkem jsonu. Rezultat: Category with Minimum Count: PARENTS, Count: 3955
""" 
category_counts = {}

with open("News_Category_Dataset_IS_course.json", "r") as file:

    for line in file:
        #parsamo json
        data = json.loads(line)
        category = data.get('category')

        
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1    

#for category, count in category_counts.items():
#   print(f"Category: {category}, Count: {count}")

print()
min_category = min(category_counts, key=category_counts.get)
min_count = category_counts[min_category]
print(f"Category with Minimum Count: {min_category}, Count: {min_count}")
print() """

file_path = 'News_Category_Dataset_IS_course.json'
i = 0

with open(file_path, 'r') as file, open("text_50k.json", "w") as json_file:
    category_counts1 = {}

    for line in file:
        #parsamo json
        data = json.loads(line)
        category = data.get('category')
        if category in category_counts1:
            category_counts1[category] += 1
        else:
            category_counts1[category] = 1

        if(category_counts1[category] < 3955):
            link = data.get('link')
            Scraped_text = scrape_website_text(link)
            text_json = {
            "link": link,
            "text": Scraped_text
            }
            json.dump(text_json, json_file)
            json_file.write('\n')
        
        i += 1
        if(i == 50000):
            break


