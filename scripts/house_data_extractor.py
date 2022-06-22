from collections import defaultdict
import pandas as pd
from bs4 import BeautifulSoup
import gzip

def get_data_from_listing(soup):
    """
    Returns a pandas Series containing the found attributes of a house listing webpage.
    """
    datadict = defaultdict(str)
    try:
        datadict["url"] = soup.find("link")["href"]
        datadict["street"] = soup.find("span", class_="object-header__title").text
        datadict["postal_city"] = soup.find("span", class_="object-header__subtitle").text
    except AttributeError:
        pass

    try:
        datadict["sold_price"] = soup.select_one("strong.object-header__price--historic").text
    except AttributeError:
        pass

    try:
        datadict["sale_broker"] = soup.select_one("a.object-contact-aanbieder-link").text
    except AttributeError:
        pass

    area_data_summary = soup.select_one("ul.kenmerken-highlighted__list.fd-flex.fd-list--none.fd-p-none")
    if area_data_summary:
        try:
            datadict["n_sleepingrooms"] = area_data_summary.select_one("span.fd-color-dark-3.fd-m-right-2xs[title=slaapkamers] ~ span.kenmerken-highlighted__value.fd-text--nowrap").text
        except AttributeError:
            pass
    
        
    n_photos = soup.select_one("span.text-dark-3.ml-1.hidden")
    if n_photos:
        datadict["n_photos"] = n_photos.text
    else:
        datadict["n_photos"] = 0
    
    sale_hist_section = soup.select_one("section.object-kenmerken")
    if sale_hist_section:
        sale_hist_data = sale_hist_section.select("dt")
        for sale_data in sale_hist_data:
            try:
                datadict[sale_data.text] =  sale_data.find_next_sibling().text
            except AttributeError:
                pass

    attrs = soup.select_one("section.object-kenmerken.is-expanded")
    if attrs:
        attr_data = attrs.select("dt")
        for attr in attr_data:
            try:
                datadict[attr.text] = attr.next_sibling.select_one("span").text
            except AttributeError:
                pass

    return pd.Series(datadict)

def write_housedata_csv(
    input_filepath, output_filepath, 
    limit, chunksize, progress_steps=10
    ):
    """
    Write house webpage data to a csv file.
    """
    l_series = list()
    with  gzip.open(input_filepath, "r") as f:
        for c, line in enumerate(f):
            if limit > 0 and c > limit:
                break

            line = line.decode("UTF-8").replace("\\r\\n", "")
            l_series.append(
                get_data_from_listing(BeautifulSoup(line, features="lxml"))
                )

            if c % progress_steps == 0 and c != 0:
                print(f"Progress: {c}")

            if c % chunksize == 0 and c != 0:
                header = 1 if c <= chunksize else 0
                mode = "w" if c <= chunksize else "a"
                
                pd.DataFrame(l_series).to_csv(output_filepath, mode=mode, header=header, compression="gzip")

                l_series = list()
    return 

if __name__ == "__main__":
    input_filepath = "output/transfer/raw_full.txt.gz"
    output_filepath = "output/"+ input("output csv filename, without file extensions:\n") + ".csv.gz"

    write_housedata_csv(
        input_filepath=input_filepath,
        output_filepath=output_filepath,
        limit = int(input("House limit, <0 = no limit:\n")) - 1,
        chunksize = int(input("Chunksize in rows. Smaller or equal to limit:\n")) - 1
    )
