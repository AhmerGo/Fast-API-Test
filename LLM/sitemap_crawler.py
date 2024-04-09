import requests
import xml.etree.ElementTree as ET


def fetch_sitemap_urls(sitemap_url):
    """Fetch and parse URLs from a sitemap or sitemap index."""
    urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(sitemap_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to download sitemap: {sitemap_url} with status code: {response.status_code}")
        return urls

    # Parse the XML content of the sitemap
    sitemap = ET.fromstring(response.content)

    # Namespace is often used in sitemaps
    namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    # Check if this is a sitemap index
    if sitemap.tag == '{http://www.sitemaps.org/schemas/sitemap/0.9}sitemapindex':
        # Fetch URLs from each sitemap listed in the sitemap index
        for sitemap_entry in sitemap.findall('sitemap:sitemap', namespace):
            sitemap_url = sitemap_entry.find('sitemap:loc', namespace).text
            urls.extend(fetch_sitemap_urls(sitemap_url))  # Recursive call for nested sitemaps
    else:
        # Fetch URLs listed directly in the sitemap
        for url_entry in sitemap.findall('sitemap:url', namespace):
            url_loc = url_entry.find('sitemap:loc', namespace).text
            urls.append(url_loc)

    return urls


def filter_urls(urls, keywords):
    """Filter URLs to include only those containing specified keywords."""
    filtered_urls = [url for url in urls if any(keyword in url.lower() for keyword in keywords)]
    return filtered_urls


# Example sitemap URL
sitemap_url = 'https://www.hsutx.edu/sitemap_index.xml'
urls = fetch_sitemap_urls(sitemap_url)

# Specify the keywords for filtering URLs related to admissions and housing
keywords = ['admissions',
            'housing',
            'graduate-programs',
            # 'grad-degree',
            'financial-aid',
            # 'academics',
            # 'fasttrack'
]

# Filter URLs
filtered_urls = filter_urls(urls, keywords)

# Write filtered URLs to a file
with open('./Website_URLS/website_urls.txt', 'w+') as website_urls:
    for url in filtered_urls:
        website_urls.write(url + '\n')

# Optionally, print the number of filtered URLs
print(f"Number of filtered URLs: {len(filtered_urls)}")
