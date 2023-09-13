import os
import argparse
from dotenv import load_dotenv
from luxonis_ml.robothub.config_rh import RHConfig
from luxonis_ml.robothub.filter_rh import RH_Downloader
from luxonis_ml.robothub.convert_ldf import LDF_Converter

def main(args):
    # Read the config
    config_path = args.config_path
    schema_path = args.schema_path
    rh_config = RHConfig(config_path, schema_path)
    # rh_config.pretty_print_config()

    # Get the token
    load_dotenv(args.env_path)
    rh_token = os.getenv('TOKEN')

    # Download the images
    dest_dir = args.dest_dir
    rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir)
    rh_downloader.run_async_download()

    # Convert to LDF
    ldf_converter = LDF_Converter(rh_config, dest_dir)
    ldf_converter.detections_to_ldf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process RobotHub data.")
    parser.add_argument("--config_path", type=str, default='./configs/simple_rh.yaml', help="Path to the config YAML file.")
    parser.add_argument("--schema_path", type=str, default='./configs/config_schema_rh.json', help="Path to the schema JSON file.")
    parser.add_argument("--env_path", type=str, default='./.env', help="Path to the .env file containing the token.")
    parser.add_argument("--dest_dir", type=str, default='./tmp/images', help="Destination directory for downloaded images.")
    args = parser.parse_args()
    main(args)
