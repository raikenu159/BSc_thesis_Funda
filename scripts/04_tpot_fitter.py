import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

print("Reading data...")
data = pd.read_parquet(f"output/houses_encoded_place_skl.parquet")

print("Selecting and splitting data...")
houses = {
    "features": data.iloc[:,1:],
    "target": data.iloc[:,0]}    
X_train, X_test, y_train, y_test = train_test_split(
    houses["features"].values, houses["target"].values, 
    train_size=0.75, test_size=0.25, random_state=42, shuffle=True)
print("Data selected and split!")

n_generations = int(input("n_generations, default=100: "))
pop_size = int(input("population size, default=100: "))

config_selection = str(input("Default or light configuration? d=default, l=light: "))
while config_selection != "d" and config_selection != "l":
    print("Enter d for default or l for light configuration: ")
    config_selection = str(input("Default or light configuration? d/l: "))
if config_selection == "d":
    config = None
else:
    config = "TPOT light"

# Linux/OSX n_jobs > 1 fix mentioned in sklearn FAQ https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
if "win" not in sys.platform:
    print("non-windows platform detected! Continuing with forkserver start_method")
    import multiprocessing
    multiprocessing.set_start_method('forkserver')

tpot = TPOTRegressor(
    generations=n_generations, 
    population_size=pop_size, 
    verbosity=int(input("Verbosity? 1, 2 or 3: ")), 
    random_state=42, 
    config_dict=config,
    n_jobs=int(input("n_jobs: ")),
    warm_start=int(input("Warm start? 1=yes, 0=no: ")),
    periodic_checkpoint_folder="output/model_fit/periodic_checkpoints"
)
tpot.fit(X_train, y_train)

print( tpot.score(X_test, y_test) )
tpot.export("output/model_fit/houses_pipeline.py")
