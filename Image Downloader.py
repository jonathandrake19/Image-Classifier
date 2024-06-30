import requests
import json
import os
import names

subscription_key = '5fd7ada9f7524e67ae2e6ea25326d0e3'
search_url = 'https://api.bing.microsoft.com/v7.0/images/search'

def search(query, count):
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "count": count}
        
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results

def save_images(image_urls, folder_path, name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, url in enumerate(image_urls):
        try:
            image_data = requests.get(url).content
            with open(os.path.join(folder_path, f'{name}_{i+1}.jpg'), 'wb') as handler:
                handler.write(image_data)
        except Exception as e:
            print(f"Could not save image {i+1} from {url}. Error: {e}")

def main():
    count = 5
    for i in names.words():
        query = i + ' plant'
        results = search(query, count)

        image_urls = []
        for image in results['value']:
            image_urls.append(image['contentUrl'])

        folder_path = '/Users/jonathandrake/Desktop/Python Project/Images'
        save_images(image_urls, folder_path, i)

    print('Done')   

if __name__ == "__main__":
    main()