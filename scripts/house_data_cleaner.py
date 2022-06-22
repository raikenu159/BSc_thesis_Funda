import re

import dateparser
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def serve_dataset(data_path):
    '''Read only data where status is sold, as some retrieved listings were shifted in columns. Drop duplicates afterwards.'''
    original_columns = list(pd.read_csv(data_path, compression="gzip", nrows=0))
    data = pd.read_csv(data_path, compression="gzip", low_memory=False, usecols=[col for col in original_columns if col != "Unnamed: 0"])
    data = data[data["Status"].str.strip() == "Verkocht"]

    return data.drop_duplicates()

def drop_unused(dataframe):
    # Only keep columns where at least 70% of rows have a (non-nan)value. difference between 70 and 75: energielabel, schuur/berging, achtertuin
# Only keep rows where at least 75% the columns have a (non-nan)value.
    '''
    Drop unused columns from input dataframe and set thresholds on NaN-values for columns and rows. 
    Remaining columns dropped manually and their reason are:
    Laatste Vraagprijs: duplicate of sold_price,
    Eigendomssituatie: not informative because of 'zie akte' values,
    Looptijd: will be calculated separately in days, instead of 
        x amount of days/weeks/months like on the pages,
    Status: was only used to select sold houses
    Oppervlakte: duplicate of Perceel,
    Cv-ketel: unnessecary extra information. 
        Useful information is already contained in heating/warm water,
    Achtertuin: Inconsistent.

    '''
    df = dataframe.copy()
    
    # Only keep columns where at least 70% of rows have a (non-nan)value. difference between 70 and 75: energielabel, schuur/berging, achtertuin
    df = df.dropna(axis=1, thresh=int(0.7 * df.shape[0]))
    # Only keep rows where at least 75% the columns have a (non-nan)value.
    df = df.dropna(thresh=int(0.75 * df.shape[1]))
    df = df.drop(axis=1, labels=["Laatste vraagprijs", "Eigendomssituatie", "Looptijd", "Status", "Oppervlakte", "Cv-ketel", "Achtertuin"])

    return df

def rename_columns(dataframe: pd.DataFrame):
    '''
    Rename columns to names with prefixes, 
    especially useful for isolating categorical (type_) features.
    '''
    df = dataframe.copy()
    new_column_names = {
            "Aangeboden sinds":"date_listed",
            'Verkoopdatum':'date_sale',
            'Bouwjaar':'year_build',
            'Wonen':'area_living',
            'Perceel':'area_plot',
            'Gebouwgebonden buitenruimte':'area_outside',
            'Achtertuin':'area_backyard',
            'Overige inpandige ruimte':'area_misc_indoor',
            'Inhoud':'volume',
            'Aantal kamers':'n_rooms',
            'Aantal woonlagen':'n_floors',
            'Aantal badkamers':'n_bathrooms',
            'Capaciteit':'n_cars_garage',
            'Soort woonhuis': 'type_house',
            'Soort bouw':'type_build',
            'Voorzieningen':'type_facilities',
            'Isolatie':'type_isolation',
            'Verwarming':'type_heating',
            'Warm water':'type_warm_water',
            'Cv-ketel':'type_boiler',
            'Tuin':'type_yard',
            'Ligging tuin':'type_orientation_yard',
            'Balkon/dakterras':'type_balcony_terrace',
            'Soort parkeergelegenheid':'type_parking',
            'Soort dak':'type_roof',
            'Externe bergruimte':'area_external_storage',
            'Energielabel':'type_energy_label',
            'Ligging':'type_locality',
            'Schuur/berging':'type_storage',
            'Soort garage':'type_garage',
            'Badkamervoorzieningen':'type_bathroom_facilities',
            'Specifiek':'type_specific',
        }
    df = df.rename(columns = new_column_names)

    return df 

def reorder_columns(dataframe: pd.DataFrame):
    '''
    Reorder columns so that features not used for modelling are first and
    similar type features are together.
    '''
    df = dataframe.copy()

    columns_ordered = [
        'url', 'street', 'postal_city','sale_broker', 'date_listed', 'date_sale', # not for training
        'sold_price', # only numeric value for training
        'year_build', # only year value for training
        'area_living', 'area_plot','volume', # dimension values
        'n_photos','n_rooms', 'n_sleepingrooms', 'n_floors', # count values
        'type_house','type_build','type_facilities', 'type_isolation', # categorical values
        'type_heating', 'type_warm_water', 'type_yard',
        'type_orientation_yard',
        'type_parking', 'type_roof', 
        'type_energy_label', 'type_locality', 'type_storage',
         'type_bathroom_facilities']
        
    return df[columns_ordered]

def split_postal_city(dataframe: pd.DataFrame):
    '''
    Extract postal code from postal_city and place from url.
    '''
    df = dataframe.copy()
    postal_temp = df["postal_city"].str.split("\s+")
    postal_code = postal_temp.str.slice(start=0, stop=2).str.join(" ")
    place = df["url"].str.extract("verkocht\/(.*)\/huis").iloc[:,0]
   
    df.insert(2, "postal_code", postal_code)
    df.insert(df.columns.get_loc("type_house"), "type_place", place)
    df.drop(labels=["street", "postal_city"], axis=1, inplace=True)

    return df

def extract_digits(raw_value: str):
    '''
    Extract digits from numerical features. Used for area, volume, sold_price, 
    n_rooms, n_floors
    '''
    try:
        raw_value = raw_value.strip().replace(".", "")
    except AttributeError:
        return np.nan

    pattern = "(ac )?(\d+) [mkvwa][^e]"
    try:
        return float(
            re.search(pattern, raw_value).group(2)
        )
    except AttributeError:
        return np.nan

def clean_digit_columns(dataframe: pd.DataFrame):
    '''
    Apply digit extraction to relevant columns
    '''
    df = dataframe.copy()
    columns_to_clean = df.columns.str.startswith(("area_", "volume", "sold", "n_ro", "n_flo"))
    area_columns = df.loc[:,columns_to_clean]
    for col in area_columns:
        df[col] = df[col].apply(extract_digits)
        df[col] = df[col].astype(np.float32)
    df["n_photos"] = df["n_photos"].astype(np.float32)
    df["n_sleepingrooms"] = df["n_sleepingrooms"].astype(np.float32)
    
    return df

def parse_dates(dataframe: pd.DataFrame):
    """
    Convert Dutch written out dates to datetime format.
    """
    df = dataframe.copy()
    date_features = ['date_listed', 'date_sale']
    for feature in date_features:
        df[feature] = df[feature].apply(
            lambda x: dateparser.parse(x, date_formats=['%d %B %Y'], languages=['nl'])
            )
    # year type
    return df

def get_sellingtime(dataframe: pd.DataFrame):
    """
    Get days_to_sell feature from the difference between date_listed and date_sale.
    """
    df = dataframe.copy()
    days_to_sell =  df.apply(
        lambda x:
            int( ((x['date_sale'] - x['date_listed']).days) ), 
        axis=1           
    )
    sale_loc = df.columns.get_loc('date_sale')
    df.insert(loc = sale_loc + 1, column='days_to_sell', value=days_to_sell)
    df.drop(df[df['days_to_sell'] < 0].index, axis=0, inplace=True)
    
        
    return df

def split_type_values(dataframe: pd.DataFrame):
    """
    Split all type_- features by vertical bars ("|") on 'en' and comma's.
    """
    df = dataframe.copy()

    # energy_label, orientation_yard require other processing. Place already processed.
    unwanted_cols = ["type_sale_broker", "type_energy_label", "type_orientation_yard", "place"]
    type_columns = [col for col in data_temp if col.startswith("type_") if col not in unwanted_cols]
    for col in type_columns:
        df[col] = df[col].str.lower().str.strip().str.replace(" ", "_").str.split(r",_|_en_").str.join("|")
        df[col].fillna("unspecified", inplace=True)

    return df

def clean_energy_label(dataframe: pd.DataFrame):
    """
    Extract energy label. Values above A (A+ or more plusses) are regarded as A.
    """
    df = dataframe.copy()
    pattern = "([ABCDEFG])"
    df["type_energy_label"]  = df["type_energy_label"].str.extract(pattern)
    df["type_energy_label"].fillna("unspecified", inplace=True)

    return df

def get_yard_orientation(value: str):
    """
    Extract cardinal direction of yard orientation.
    """
    try:
        value = value.lower()
        pattern = r"noorden|oosten|zuiden|westen|noordoosten|zuidoosten|zuidwesten|noordwesten"
        orientation = re.findall(pattern, value)[0]
        return orientation
    except AttributeError:
        return np.nan

def get_yard_backside_access(value: str):
    """
    Extract prescence of backside access to yard.
    """
    try:
        value = value.lower()
        pattern = r"bereikbaar"
        access = re.findall(pattern, value)
        return 1 if access else 0
    except AttributeError:
        return 0

def clean_yard_orientation(dataframe: pd.DataFrame):
    """
    Replaces type_orientation_yard feature with 2 new features: extracted backyard orientation and prescence of backside access.
    """
    df = dataframe.copy()

    orientation = df["type_orientation_yard"].apply(get_yard_orientation)
    orientation.fillna("unspecified", inplace=True)
    backside_access = df["type_orientation_yard"].apply(get_yard_backside_access)

    idx_yard = df.columns.get_loc("type_orientation_yard")
    df.insert(idx_yard+1, "yard_backside_access", backside_access)
    df.insert(idx_yard+2, "type_yard_orientation", orientation)
    df.drop("type_orientation_yard", axis=1, inplace=True)
    
    return df

def clean_housetype(dataframe: pd.DataFrame):
    """Extract unique housetype values and join the found strings by vertical bars."""
    df = dataframe.copy()
    pattern = "(2-onder-1-kapwoning|bungalow|eengezinswoning|eindwoning|geschakelde_woning|grachtenpand|halfvrijstaande_woning|herenhuis|hoekwoning|landgoed|landhuis|stacaravan|tussenwoning|verspringend|villa|vrijstaande_woning|woonboerderij|woonboot|woonwagen|bedrijfs-_of_dienstwoning|dijkwoning|drive-in_woning|hofjeswoning|kwadrant_woning|patiowoning|semi-bungalow|split-level_woning|waterwoning|wind-_of_watermolen|paalwoning)"


    df["type_house"] = df["type_house"].str.findall(pattern).str.join("|")
    df["type_house"].fillna("unspecified", inplace=True)
    
    return df
    
def clean_bathroom_facilities(dataframe: pd.DataFrame):
    '''Remove counts from facilities'''
    df = dataframe.copy()
    digit_pattern = "\d+_"
    facility_pattern = "douche|toilet|jacuzzi|ligbad|zitbad|stoomcabine|sauna"
    df["type_bathroom_facilities"] = df["type_bathroom_facilities"].str.replace(digit_pattern, "", regex=True).str.findall(facility_pattern).str.join("|")
    df["type_bathroom_facilities"].fillna("unspecified", inplace=True)
    
    return df

def clean_roof_types(dataframe: pd.DataFrame):
    '''Extract separate roof features for making dummies'''
    df = dataframe.copy()
    pattern = "bitumineuze_dakbedekking|lessenaardak|mansarde_dak|plat_dak|dwarskap|schilddak|tentdak|zadeldak|samengesteld_dak|asbest|kunststof|kunststof|leisteen|metaal|pannen|riet|overig"
    df["type_roof"] = df["type_roof"].str.findall(pattern).str.join("|")
    df["type_roof"].fillna("unspecified", inplace=True)

    return df

def parse_buildyear(value: pd.DataFrame):
    """Convert buildyear to an integer. However, because of not found values, the eventual feature will be of datatype float."""
    try:
        return int(value)
    except ValueError:
        return np.nan

def clean_buildyear(dataframe: pd.DataFrame):
    df = dataframe.copy()
    df["year_build"] = df["year_build"].apply(parse_buildyear)
    return df

def handle_nans(dataframe: pd.DataFrame):
    """Drop instances which contain NaN for essential features. Impute other numerical features with median."""
    df = dataframe.copy()
    df = df.dropna(subset=["sold_price", "year_build", "area_living", "area_plot", "n_floors"])
    numerical_columns = df.loc[:,"days_to_sell":"n_floors"].columns
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    return df
def convert_dtypes(dataframe: pd.DataFrame):
    """Convert float64 features to np.float32 and yard_backside_access to np.int8"""
    df = dataframe.copy()

    floatcols = ["days_to_sell", "sold_price", "year_build"]
    for col in floatcols:
        df[col] = df[col].astype(np.float32)
    df["yard_backside_access"] = df["yard_backside_access"].astype(np.int8)

    return df

def categorise_ready_cols(dataframe: pd.DataFrame):
    '''Convert features with single categorical entries as values to the pandas categorical datatype.'''
    df = dataframe.copy()
    unique_cols = ["type_place", "type_build", "type_yard_orientation", "type_energy_label", "type_storage"]
    for col in unique_cols:
        df[col].fillna("unspecified", inplace=True)
        df[col] = df[col].astype("category")

    return df

def drop_outliers_fewplaces(dataframe):
    '''Drop outliers of numerical features and drop instances where its place occurs less than 10 times.'''
    df = dataframe.copy()
    float_cols = df.select_dtypes(np.float32).columns 
    dict_delete_count = dict()
    # Only select rows within 3 standard deviations of mean
    for col in float_cols:
        n_rows_pre = df.shape[0]
        df = df[df[col] <= df[col].mean() + 3*df[col].std()]
        n_rows_post = df.shape[0]
        dict_delete_count[col] = n_rows_pre - n_rows_post

    # Drop rows with places which occur less than 10 times.
    minimum_place_count = 10
    n_rows_pre = df.shape[0]
    df = df[
        df['type_place'].map(
            df['type_place'].value_counts()) >= minimum_place_count
        ]
    n_rows_post = df.shape[0]
    dict_delete_count["type_place"] = n_rows_pre - n_rows_post
    print(dict_delete_count)

    return df

def encode_multilabel_col(dataframe: pd.DataFrame, col: str):
    '''One-hot encode features containing multiple categorical features (multi-label) split by vertical bars.'''
    df = dataframe.copy()
    mlb = MultiLabelBinarizer(sparse_output=False)

    df = df.join(
        other = pd.DataFrame(
            data=mlb.fit_transform(df.pop(col)),
            columns=mlb.classes_,
            index=df.index
            ).add_prefix(f"{col}_").astype(np.int8),
    )

    return df

def encode_categorical_col(dataframe: pd.DataFrame, col: str):
    '''One-hot encode features containing single categorical values (single label)'''
    df = dataframe.copy()
    enc = OneHotEncoder(sparse=False)
    df = df.join(
        pd.DataFrame(
            data=enc.fit_transform(np.array(df.pop(col).values).reshape(-1,1)),
            index=df.index,
        ).add_prefix(f"{col}_").astype(np.int8)
    )
    return df

def sklearn_encode_categoricals(dataframe: pd.DataFrame):
    '''One-hot encode multilabel and single label features.'''
    df = dataframe.loc[:,"days_to_sell":].copy()
    df_slct = df.select_dtypes([float, object])
    multilabel_columns = df_slct.columns[df_slct.columns.str.startswith("type_")]
    for col in multilabel_columns:
        df[col] = df[col].str.split("|").apply(set)
        df = encode_multilabel_col(df, col)

    cat_columns = df.select_dtypes(pd.CategoricalDtype).columns
    for col in cat_columns:
        pass
        df = encode_categorical_col(df, col)
    return df

if __name__ == '__main__':
    data_path = "output/transfer/houses_full_unclean.csv.gz"

    print("serve_dataset...")
    data_temp = serve_dataset(data_path)
    data_temp.to_parquet("output/cleaning_process/1_serve_dataset.parquet", )

    pre_encode_functions = [drop_unused, rename_columns, reorder_columns, split_postal_city,
        clean_digit_columns, parse_dates, get_sellingtime,
        split_type_values, clean_energy_label, clean_yard_orientation, clean_housetype,
        clean_bathroom_facilities, clean_roof_types, clean_buildyear, handle_nans,
        convert_dtypes, categorise_ready_cols, drop_outliers_fewplaces
    ]
    for i, func in enumerate(pre_encode_functions, start=2):
        print(f"{func.__name__}...")
        data_temp = func(data_temp)
        data_temp.to_parquet(f"output/cleaning_process/{i}_{func.__name__}.parquet")

    df_unencoded = data_temp

    print("save unencoded...")
    df_unencoded.to_parquet("output/houses_unencoded.parquet")

    print("sklearn_encode_categoricals...")
    df_encoded_place_skl = sklearn_encode_categoricals(df_unencoded)
    df_encoded_place_skl.to_parquet("output/houses_encoded_place_skl.parquet")

