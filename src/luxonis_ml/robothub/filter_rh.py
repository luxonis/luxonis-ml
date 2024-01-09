import random
import uuid
import requests
import aiohttp
import aiofiles
import os
import json
import asyncio
from google.cloud import storage
from datetime import datetime, timedelta
from cron_validator import CronValidator
from luxonis_ml.utils import LuxonisFileSystem

class RH_Downloader:
    
    def __init__(self, rh_token, rh_config, dest_dir, lfs:LuxonisFileSystem):
        self.rh_token = rh_token
        self.cfg = rh_config.get_config()
        self.dest_dir = dest_dir # without gs://bucket_name, only prefix
        self.lfs = lfs

    def get_all_robot_ids(self):
        response = requests.get(
            'https://robothub.luxonis.com/api/robot?status=connected',
            headers={'Authorization': f'Bearer {self.rh_token}'},
            timeout=10
        )

        robots = response.json()['robots']
        robot_ids = [robot['id'] for robot in robots]
        return robot_ids
    
    def build_url(self, page_i=1):
        base_url = 'https://robothub.luxonis.com/api/detection/search'
        params = []
        params.append('take=100')
        params.append(f'page={page_i}')

        repeat_every = self.cfg.get("repeat_every")
        if repeat_every:
            # calculate the from_date and to_date
            now = datetime.now()

            # get largest time unit
            td_sec = [61, 3601, 86401, 2592001, 31536001]
            n = repeat_every.split(' ')
            s = [td_sec[i+1] for i in range(len(n)-1) if n[i] != '*'][-1]
            if n[-1] != '*':
                s = 2592001
            start = now - timedelta(seconds=s)
            exe_list = list(CronValidator.get_execution_time(repeat_every, start, now))

            from_date = exe_list[-1]
            from_date = from_date.strftime("%Y-%m-%d")
            params.append(f'from={from_date}')

        robot_ids = self.cfg.get("robots", [])
        if robot_ids == "all":
            robot_ids = self.get_all_robot_ids()

        for robot_id in robot_ids:
            params.append(f'robot={robot_id}')
        

        tags = self.cfg.get("tags", [])
        for tag in tags:
            params.append(f'tag={tag}')

        params.append('hasFrames=true')

        full_url = base_url + '?' + '&'.join(params)
        print("Fetching detections from: ", full_url)
        return full_url
    
    def get_all_detections(self):
        items = []
        page_i = 1
        while True:
            full_url = self.build_url(page_i)
            response = requests.get(
                full_url,
                headers={'Authorization': f'Bearer {self.rh_token}'},
                timeout=10
            )
            if response.status_code != 200:
                print("Error: ", response.status_code)
                print("Error occured while fetching detections.")
                print(response.text)
                return
            
            new_items = response.json()['items']
            items += new_items
            page_i += 1

            if len(items) >= response.json()['pagination']['total'] or len(new_items) == 0:
                if self.cfg.get("tactic") in ["all", "smart"]:
                    break
                elif self.cfg.get("tactic") == "random":
                    items = random.sample(items, self.cfg.get("num_of_imgs"))
                    break
            elif self.cfg.get("tactic") == "limit" and len(items) >= self.cfg.get("num_of_imgs"):
                items = items[:self.cfg.get("num_of_imgs")]
                break
        
        return items
    
    def filter_detections(self, items):
        # filter detections by app_id
        app_ids = self.cfg.get("apps", [])
        if app_ids != "all":
            items = [item for item in items if item['appId'] in app_ids]
        
        # filter detections based on include_unannotated
        include_unannotated = self.cfg.get("include_unannotated", [])
        if "null" in include_unannotated:
            # replace "null" with None
            include_unannotated = [None if x == "null" else x for x in include_unannotated]
        if include_unannotated:
            items = [item for item in items if item['classification'] in include_unannotated]
        
        if self.cfg.get("tactic") == "limit":
            num_of_imgs = self.cfg.get("num_of_imgs")
            if num_of_imgs:
                items = items[:num_of_imgs]
        
        return items

    def save_detections_info(self, items):
        detection_infos = []
        print("Saving detections info...", len(items))
        # Filter and reformat the detections
        for detection in items:
            try:
                # try getting device_id
                device_id = ""
                if "title" in detection:
                    title_name = detection['title'].split(' ')
                    if len(title_name) > 4:
                        device_id = title_name[4].strip('.')
                
                image_data = detection['frames'][0]
                image_name = image_data["path"] if isinstance(image_data, dict) else image_data
                image_uuid = image_name.split('frame__')[-1]
                new_full_path = self.dest_dir + '/images/' + image_uuid + '.png'

                # Create a new dictionary with the required fields
                detection_info = {
                    "id": detection["id"],
                    "createdAt": detection["createdAt"],
                    "teamId": detection["teamId"],
                    "robotId": detection["robotId"],
                    "deviceId": device_id,
                    "appId": detection["appId"],
                    "tags": detection["tags"],
                    "frame": image_name,
                    "framePath": new_full_path,
                    "data": detection["data"],
                    "classification": detection["classification"]
                }

                # Modify tags based on out_tags
                out_tags = self.cfg.get("out_tags", "copy")
                if out_tags == "copy":
                    pass
                elif out_tags == "new":
                    detection_info["tags"] = self.cfg.get("out_tags_new", [])
                elif out_tags == "both":
                    detection_info["tags"].extend(self.cfg.get("out_tags_new", []))
                elif out_tags == "none":
                    detection_info["tags"] = []
                
                detection_infos.append(detection_info)
                
            except:
                print("Error occured while processing detection: ", detection)
                continue
        
        # Save the detections to a json file
        filename = self.dest_dir + '/detections.json'
        data = json.dumps(detection_infos).encode('utf-8')
        self.lfs.put_bytes(data, filename)
        
        print(f"Saved {len(detection_infos)} detections to {filename}")
        return filename

    async def download_frames(self, item_info, session):
        id_ = item_info['id']
        image_name = item_info['frame']
        file_name = item_info['framePath']

        url = f"https://robothub.luxonis.com/api/detection/{id_}/download/{image_name}"

        async with session.get(url) as response:
            image_data = await response.read()
            self.lfs.put_bytes(image_data, file_name)
    
    async def download_images(self, items):        
        # Download the frames
        async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {self.rh_token}'}) as session:
            await asyncio.gather(
                *[self.download_frames(image, session) for image in items]
            )
    
    def run_async_download(self):
        detections = self.get_all_detections()
        filtered_detections = self.filter_detections(detections)
        detections_file = self.save_detections_info(filtered_detections)
        saved_detections_buffer = self.lfs.read_to_byte_buffer(detections_file)
        saved_detections = json.loads(saved_detections_buffer.decode('utf-8'))
        asyncio.run(self.download_images(saved_detections))
        
