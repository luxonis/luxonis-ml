"""
mongo_config_upload.py

This script uploads configuration files and schemas to a MongoDB instance. 

Usage:
    python3 mongo_config_upload.py --folder [FOLDER_PATH] --configs [CONFIG_FILENAMES] --schema [SCHEMA_PATH/SCHEMA_FILENAME] --mongo_uri [MONGO_URI] --db_name [DATABASE_NAME] --collection_name [COLLECTION_NAME]

Arguments:
    --folder: The directory path containing the configuration files.
    --configs: A list of configuration filenames to upload or special values 'all' or 'none'. 
               'all' uploads all YAML files in the specified folder, while 'none' skips uploading configs.
    --schema:  The filename of the schema to be uploaded.
    
    --mongo_uri: MongoDB connection URI. Default is "mongodb://localhost:27017/".
    --db_name: Name of the MongoDB database. Default is "robothub".
    --collection_name: Name of the MongoDB collection to insert data into. Default is "configs".

Note: Date fields in the configuration files are converted to string format "%Y-%m-%d" before uploading.
"""

import json
import pymongo
import yaml
import os
import argparse

def upload_configs_to_mongo(folder, configs, schema, mongo_uri, db_name, collection_name, schema_collection_name):
    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Determine which configs to upload
    if configs[0] == 'all':
        config_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(".yaml")]
    elif configs == 'none':
        config_files = []
    else:
        config_files = [f for f in configs if os.path.isfile(os.path.join(folder, f)) and f.endswith(".yaml")]

    # Load YAML files and store in MongoDB
    for config_file in config_files:
        with open(os.path.join(folder, config_file), 'r') as file:
            config_data = yaml.safe_load(file)

        # for any date fields, convert to string
        for key in config_data:
            if "date" in key and config_data[key]:
                config_data[key] = config_data[key].strftime("%Y-%m-%d")

        # update the config in MongoDB
        collection.update_one({"config_name": os.path.splitext(os.path.basename(config_file))[0]}, 
                              {"$set": {"data": config_data, "status": "active"}}, upsert=True)
        # mongo shell: db.configs.updateOne({ "config_name": "minimal_rh" }, { $set: { "status": "inactive" }})

    # Load JSON schema and store in MongoDB
    if schema:
        with open(schema, 'r') as file:
            schema_data = json.load(file)
        collection = db[schema_collection_name]
        collection.update_one({"schema_name": os.path.splitext(os.path.basename(schema))[0]}, 
                              {"$set": {"data": schema_data}}, upsert=True)


def main():
    parser = argparse.ArgumentParser(description='Upload configs to MongoDB.')
    parser.add_argument('--folder', required=True, help='Folder containing the config files')
    parser.add_argument('--configs', nargs='+', default=['none'], help='List of config files to upload or "all" or "none"')
    parser.add_argument('--schema', help='Schema file to be uploaded')
    
    parser.add_argument('--mongo_uri', default="mongodb://localhost:27017/", help='MongoDB connection URI')
    parser.add_argument('--db_name', default="robothub", help='Name of the MongoDB database')
    parser.add_argument('--collection_name', default="configs", help='Name of the MongoDB collection to insert data into')
    parser.add_argument('--schema_collection_name', default="schemas", help='Name of the MongoDB collection to insert schemas into')

    args = parser.parse_args()
    upload_configs_to_mongo(args.folder, args.configs, args.schema, args.mongo_uri, args.db_name, args.collection_name, args.schema_collection_name)

if __name__ == '__main__':
    main()

# python3 mongo_config_upload.py --folder ./configs/ --configs simple_rh.yaml full_rh.yaml --schema schema.json
# python3 mongo_config_upload.py --folder ./configs/ --configs all --schema ./configs/config_schema_rh.json
# python3 mongo_config_upload.py --folder ./configs/ --configs all --schema ./configs/config_schema_rh.json --mongo_uri "mongodb://user:pass@localhost:27017/"

