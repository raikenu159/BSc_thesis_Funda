from bs4 import BeautifulSoup
from requests_html import HTMLSession
import re
import time
import numpy as np
from datetime import datetime

def get_last_page_number(url):
    """
    Returns final page number from a Funda result page
    """
    # sends 1 request
    r = session.get(url)
    soup = BeautifulSoup(r.content, "lxml")
    last_page = soup.find("nav", class_="pagination").find_all("a")[-2].get_text()
    last_page = int(re.sub("\W*\D*", "", last_page))

    return int(last_page)

def get_resultpage_numbers_urls(resultpage_url):
    """
    Returns a list containing urls of all Funda result pages.
    """
    # sends 1 request
    last_page = get_last_page_number(resultpage_url)
    urls=[]

    # Add all result page urls to a list in descending order, so that last pages are scraped before they are possibly removed from the website.
    for i in range(last_page, 0, -1):
        urls.append("https://www.funda.nl/koop/heel-nederland/verkocht/woonhuis/sorteer-afmelddatum-op/p" + str(i) + "/")
        
    return urls

def get_listing_urls_from_resultpage_url(resultpage_url):
    """
    Returns a list containing (up to 15) house listing urls from one resultpage
    """
    # sends 1 request
    r = session.get(resultpage_url)
    soup = BeautifulSoup(r.content, "lxml")
    results = soup.find_all("div", class_="search-result__header-title-col")
    
    links = []
    for i in results:
        links.append("https://www.funda.nl" + i.find("a")["href"])

    return links

def write_listings_html_to_file(resultpage_urls, filepath, limit=2):
    """
    Write listing webpage source HTML to text file, with a sleep timer of a random value between 0.5 and 2 seconds between each request to a house listing page.
    limit = amount of resultpages. 0 or smaller is no limit.
    """
    if limit > 0:
        resultpage_urls = resultpage_urls[:limit]
        last = limit
    else:
        last = len(resultpage_urls)

    with open(filepath, "a") as f:
        for i, rurl in enumerate(resultpage_urls):
            try:
                listingurls = get_listing_urls_from_resultpage_url(rurl)
            
                l_lurls = len(listingurls) 
                for j, lurl in enumerate(listingurls):
                    print(f"page {i+1} / {last} | listing {j+1} / {l_lurls}", end="\n") 
                    f.write(f"{session.get(lurl).content}\n")
                    time.sleep(np.random.uniform(0.5, 2))

            except Exception as e:
                logpath = "output/log.txt"
                with open(logpath, "a") as g:
                    g.write(datetime.now(), e, "resultpage url: ", rurl, "listing url: ", lurl)
                continue


if __name__ == "__main__":
 
    session = HTMLSession()
    resultpage_url = "https://www.funda.nl/koop/heel-nederland/verkocht/woonhuis/sorteer-afmelddatum-op/p1/"
    
    filepath = "output/" + input("filename: ") + ".txt"

    resultpageurls = get_resultpage_numbers_urls(resultpage_url)
    print(f"{len(resultpageurls)} pages")
    limit = int(input("Page limit: "))
    
    write_listings_html_to_file(resultpageurls, filepath, limit)
