import os
from dotenv import load_dotenv
from config_rh import RHConfig
from filter_rh import RH_Downloader
# from convert_ldf_copy import LDF_Converter

def main():
    # Read the config
    config_path = './simple_rh.yaml'
    schema_path = './config_schema_rh.json'
    rh_config = RHConfig(config_path, schema_path)
    rh_config.pretty_print_config()

    # Get the token
    load_dotenv("./.env")
    rh_token = os.getenv('TOKEN')

    # Download the images
    dest_dir = './tmp/images'
    rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir)
    rh_downloader.run_async_download()

    # Convert to LDF
    # ldf_converter = LDF_Converter()
    # ldf_converter.detections_to_ldf()

if __name__ == "__main__":
    main()