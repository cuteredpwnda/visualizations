import os
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import numpy as np
import warnings
import time
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import re

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    # expand the "Tatort 2" column into something more useful
    expanded = df["Tatort 2"].str.split(", ", n=1, expand=True)
    expanded.columns = ["address", "poi"]
    expanded["street"] = expanded["address"].str.split(r" (\d+)", expand=True)[0]
    expanded["number"] = expanded["address"].str.split(r"(\s*\d+[a-z]*\-*\/*\d*)", expand=True)[1]
    expanded["number"].fillna("", inplace=True)
    # clean street names
    badwords = ["ggü.", "vor", "neben", "nahe"]
    street_regex = re.compile(r'\s+|\s+'.join(badwords))
    expanded["street"] = expanded["street"].str.split(street_regex, expand=True)[0].str.strip()
    # create a clean address column
    expanded["clean_address"] = (expanded["street"] + " " + expanded["number"]).str.strip()
    expanded["clean_address"] = expanded["clean_address"].str.replace(r"\s+", " ")
    expanded["clean_address"] = expanded["clean_address"].str.replace(r"\/", "-")    
    expanded.drop(columns=["address"], inplace=True)
    # merge the expanded columns back into the original dataframe
    df = df.merge(expanded, left_index=True, right_index=True)
    return df

def batch_geocode(df:pd.DataFrame) -> pd.DataFrame:
    if not "clean_address" in df.columns:
        raise ValueError("The dataframe does not contain a column named 'clean_address'. Please run clean_data() first.")
    
    # check for cached results
    cache_path = "data/cache.pkl"
    if os.path.exists(cache_path):
        # if newer than 1 month, read it
        creation_time = os.path.getmtime(cache_path)
        one_month = 60*60*24*30
        if  creation_time + one_month < time.time():
            print("Cache is older than 1 month. Do you want to reevaluate the addresses? (y/N)")
            if input() == "y":
                # remove cached results
                os.remove(cache_path)
                return _batch_geocoder(df)
                
        cache = pd.read_pickle(cache_path)
        # check if the cache contains all addresses, if so, return the cache
        if cache["clean_address"].isin(df["clean_address"]).all():
            return cache
        # subset the df to the addresses that are not in the cache
        to_geocode = df[~df["Aktenzeichen"].isin(cache["Aktenzeichen"])]
        # geocode the remaining addresses, where the lat and lon are not in the cache
        return _batch_geocoder(to_geocode) if len(to_geocode) > 0 else cache
    else:
        return _batch_geocoder(df)

def _batch_geocoder(df:pd.DataFrame) -> pd.DataFrame:  
    # get unique addresses
    unique = df.drop_duplicates(subset="clean_address")
    # remove empty addresses
    unique.dropna(subset=["clean_address"], inplace=True)
    tqdm.pandas(desc=f"Geocoding {len(unique)} addresses", leave=False)
    geolocator = Nominatim(user_agent="falschparker_aachen")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    unique['query'] = unique["clean_address"] + ', Aachen'
    unique['results'] = unique.progress_apply(lambda x: geocode(x["query"], language='de'), axis=1)
    #unique['results'] = unique.apply(lambda x: geocode(x["query"], language='de'), axis=1)
    unique[['lat', 'lon']] = unique['results'].progress_apply(lambda x: (x.latitude, x.longitude) if x else None).apply(pd.Series)
    df = df.merge(unique[["clean_address", "lat", "lon"]], on="clean_address", how="left", suffixes=("_x", ""))
    if "lat_x" in df.columns:
        df.drop(columns=["lat_x"], inplace=True)
    if "lon_x" in df.columns:
        df.drop(columns=["lon_x"], inplace=True)
    # cache results
    df.to_pickle("data/cache.pkl")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/statistik_test.csv", sep=";")
    df = clean_data(df)
    # testing with a small subset
    parts = np.array_split(df, 2)
    print(f"splitting df into {len(parts)} parts")
    res_df = pd.DataFrame()
    for part in parts:
        temp = part.copy()
        res_df = pd.concat([res_df, batch_geocode(temp)], ignore_index=True).drop_duplicates(subset="Aktenzeichen")
    res_df.to_csv("data/statistik_geocoded.csv", index=False)