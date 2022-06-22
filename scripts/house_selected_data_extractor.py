from collections import defaultdict
import pandas as pd
from bs4 import BeautifulSoup
import gzip

def get_selected_features_from_listing(soup, features_to_find):
    """
    Returns a series containing attributes that were found by house_data_extractor.py. 
    If no value is found for an attribute, either None or 0 is returned, depending on the type of attribute.
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
    
    found_data_rows = soup.select("dt")
    for feature in features_to_find:
        found = False
        for row in found_data_rows:
            if row.text == feature:
                found = 1
                datadict[feature] = row.find_next_sibling().text
                continue
        if not found:
            datadict[feature] = None

    return pd.Series(datadict)


def write_selected_housedata_csv(
    input_filepath, output_filepath, limit, features_to_find, 
    chunksize=10, progress_steps=10
    ):
    """
    Write house data to csv. Features_to_find are the features that were found by using house_data_extractor.py on the first 5000 housesin the raw HTML text file. 
    """
    l_series = list()
    with  gzip.open(input_filepath, "r") as f:
        for c, line in enumerate(f, start=1):
            if limit > 0 and c > limit:
                break


            line = line.decode("UTF-8").replace("\\r\\n", "")
            l_series.append(
                get_selected_features_from_listing(soup=BeautifulSoup(line, features="lxml"), features_to_find=features_to_find)
                )

            if c % progress_steps == 0 and c != 0:
                print(f"Progress: {c}")
            
            # write after chunk amount of rows have been stored in l_series
            if c != 0 and (c % chunksize == 0 or c == limit):
                header = 1 if c <= chunksize else 0
                mode = "w" if c <= chunksize else "a"
                
                pd.DataFrame(l_series).to_csv(output_filepath, mode=mode, header=header, compression="gzip")
                l_series = list()
    pd.DataFrame(l_series).to_csv(output_filepath, mode="a", header=0, compression="gzip")

    return

if __name__ == "__main__":
    input_filepath = "output/transfer/raw_full.txt.gz"
    output_filepath = "output/"+ input("output csv filename, without file extensions:\n") + ".csv.gz"
    
    # features_to_find are ound by using house_data_extractor.py on the first 5000 houses in the raw HTML text file.
    features_to_find = ['Aangeboden sinds', 'Verkoopdatum', 'Looptijd', 'Laatste vraagprijs',
       'Status', 'Soort woonhuis', 'Soort bouw', 'Bouwjaar', 'Wonen',
       'Gebouwgebonden buitenruimte', 'Inhoud', 'Aantal kamers',
       'Aantal woonlagen', 'Voorzieningen', 'Isolatie', 'Verwarming',
       'Warm water', 'Cv-ketel', 'Eigendomssituatie', 'Tuin', 'Achtertuin',
       'Ligging tuin', 'Balkon/dakterras', 'Soort parkeergelegenheid',
       'Soort dak', 'Keurmerken', 'Externe bergruimte', 'Perceel',
       'Aantal badkamers', 'Energielabel', 'Oppervlakte', 'Ligging',
       'Schuur/berging', 'Soort garage', 'Capaciteit', 'Badkamervoorzieningen',
       'Overige inpandige ruimte', 'Oorspronkelijke vraagprijs', 'Plaats',
       'Specifiek', 'Voorlopig energielabel', 'Bouwperiode',
       'Toegankelijkheid', 'Voortuin', 'Lasten', 'Zonneterras', 'Patio/atrium',
       'Zijtuin', 'Servicekosten']

    write_selected_housedata_csv(
        input_filepath=input_filepath,
        output_filepath=output_filepath,
        limit = int(input("House limit, <0 = no limit:\n")),
        features_to_find=features_to_find,
        chunksize = int(input("Chunksize in rows. Smaller or equal to limit:\n")),
        progress_steps = int(input("Progress steps: "))
    )
