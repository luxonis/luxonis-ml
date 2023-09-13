import pymongo
import yaml
import os

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["robothub"]
collection = db["configs"]


config_dir = "./configs/"
# get all yaml files in the config directory
config_files = [f for f in os.listdir(config_dir) if os.path.isfile(os.path.join(config_dir, f)) and f.endswith(".yaml")]

# Load YAML files and store in MongoDB
for config_file in config_files:
    with open(config_dir + config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    
    # for any date fields, convert to string
    for key in config_data:
        if "date" in key and config_data[key]:
            config_data[key] = config_data[key].strftime("%Y-%m-%d")

    # insert into MongoDB
    collection.insert_one({"config_name": config_file.split(".")[0], "data": config_data})

