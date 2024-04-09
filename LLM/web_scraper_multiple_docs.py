import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import concurrent.futures
import time


def scrape_and_clean(url, headers, session):
    try:
        response = session.get(url, headers=headers)
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            wait_time = int(retry_after) if retry_after else 30  # Use 30 seconds if Retry-After is not provided
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            return scrape_and_clean(url, headers, session)  # Retry the request
        elif response.status_code != 200:
            print(f"Failed to retrieve {url}: Status code {response.status_code}")
            return None, None

        soup = BeautifulSoup(response.content, 'html.parser')

        for inline_tag in soup.find_all(['b', 'strong', 'i', 'em', 'span']):
            if inline_tag.string:
                inline_tag.string.replace_with(f" {inline_tag.string} ")

        title = soup.title.string if soup.title else urlparse(url).path.split('/')[-1]
        title = title.replace('/', '_').replace(' ', '_')

        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(" ", strip=True) for p in paragraphs)

        cleaned_text = ' '.join(text.split()).replace('\u200B', '')
        return cleaned_text, title

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None, None


def save_to_text_file(text, title, url, base_dir='./Dataset'):
    filename = f"{title}.txt"
    filepath = os.path.join(base_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"URL: {url}\n\n{text}\n")

    print(f"Saved text to {filepath}")
    return True


def scrape_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    session = requests.Session()
    print(f"Scraping {url}")
    text, title = scrape_and_clean(url, headers, session)
    if text and title:
        if save_to_text_file(text, title, url):
            return True
    else:
        print(f"Failed to scrape text from {url}")
    return False


def main():
    urls_file = "Website_URLS/website_urls.txt"
    with open(urls_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f.readlines()]

    workers = 8
    successful_scrapes = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(scrape_url, urls)
        successful_scrapes = sum(result is True for result in results)

    print(f"Total webpages successfully scraped: {successful_scrapes}")


if __name__ == "__main__":
    main()
