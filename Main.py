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
""" 
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
            break """

#koda za prebrat kok je v 50k jsonu kategorij in primerov--------------------------------------
""" file_path1 = 'News_Category_Dataset_IS_course.json'


with open("final_50k.json", "r") as json_file:
    json_data = json_file.readlines()

with open(file_path1, 'r') as file:
    category_counts2 = {}

    for line in json_data:
        # Parse JSON
        data = json.loads(line)
        link = data.get('link')

        for line2 in file:
            data2 = json.loads(line2)
            link2 = data2.get('link')
            
            if link == link2:
                category = data2.get('category')
                if category in category_counts2:
                    category_counts2[category] += 1
                else:
                    category_counts2[category] = 1

                # Break out of the inner loop once a match is found
                break

# Print the results
for category, count in enumerate(category_counts2.items()):
    print(f"Category: {category}, Count: {count}") """

#-------------------------------------------------koda za dolocene kategorije scrapat od nekje napre

""" index = 58000

with open("News_Category_Dataset_IS_course.json", "r" ) as file, open("dodatno.json", "w") as file2 :
    category_counts1 = {}
    for line in file:
        if index <= 0:
            data = json.loads(line)
            category = data.get("category")
            if category in ["PARENTING" , "WELLNESS", "FOOD & DRINK", "STYLE & BEAUTY"]:
                if category in category_counts1:
                    if(category_counts1[category] > 0):
                        category_counts1[category] -= 1
                    else:
                        continue
                    
                    
                else:
                    category_counts1[category] = 300

                link = data.get("link")
                Scraped_text = scrape_website_text(link)
                text_json = {
                    "link": link,
                    "text": Scraped_text
                }
                json.dump(text_json, file2)
                file2.write('\n')

        index -= 1 """

#---------------------------------------------------------- koda za najdit vrstico v tavlkem fileu
""" i = 0
link = "https://www.huffingtonpost.com/entry/better-instructions-for-tattoo-care-could-prevent-infections-doctors-say_us_563d36b6e4b0411d307134fe"
with open("News_Category_Dataset_IS_course.json", "r" ) as file:
    for line in file:
        data = json.loads(line)
        link2 = data.get("link")
        if link == link2 :
            print("TO JE VRSTICA: ", i)
            break
        i += 1 """

#--------------------------------------------------koda za pobrisat prazne strani

with open("allgood.json", "r") as input_file, open("dodatno.json", "w") as output_file:
    for line in input_file:
        data = json.loads(line)
        text = data.get("text")
             
        if text and text.strip() != "":
            
            output_file.write(line)

#-------------------------------------------koda za doloceno kategorijo poscrapat
""" count = 0
with open("News_Category_Dataset_IS_course.json", "r") as file, open("dodatno.json", "w") as file2:
    for line in file:
        data = json.loads(line)
        category = data.get("category")
        if(category =="HOME & LIVING"):
            link = data.get("link")
            Scraped_text = scrape_website_text(link)
            if Scraped_text and Scraped_text.strip() != "":
                text_json = {
                    "link": link,
                    "text": Scraped_text
                }
                json.dump(text_json, file2)
                file2.write('\n')
             """
#----------------------------------------------dodajanje category v json

""" first_json_path = "text_50k.json"
second_json_path = "News_Category_Dataset_IS_course.json"
output_json_path = "training_set_final.json"

# Read the second JSON file into a dictionary with links as keys and categories as values
link_category_mapping = {}
with open(second_json_path, "r") as second_file:
    for line in second_file:
        data = json.loads(line)
        link = data.get("link")
        category = data.get("category")
        if link and category:
            link_category_mapping[link] = category

# Process the first JSON file and add the "category" field based on the link
with open(first_json_path, "r") as first_file, open(output_json_path, "w") as output_file:
    for line in first_file:
        data = json.loads(line)
        link = data.get("link")
        if link in link_category_mapping:
            category = link_category_mapping[link]
            data["category"] = category

        json.dump(data, output_file)
        output_file.write('\n') """

#---------------------------------------------------vsake kategorije ene 400
""" input_file_path = "training_set_final.json"
output_file_path = "allgood.json"
max_instances_per_category = 400

category_counts = {}

with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    for line in input_file:
        data = json.loads(line)
        category = data.get("category")

        if category:
            if category not in category_counts:
                category_counts[category] = 0

            if category_counts[category] < max_instances_per_category:
                json.dump(data, output_file)
                output_file.write('\n')
                
                category_counts[category] += 1 """