import requests
import aiohttp
import aiofiles
import os
import json
import asyncio

from luxonis_ml.robothub.config_rh import RHConfig

class RH_Downloader:
    
    def __init__(self, rh_token, rh_config, dest_dir):
        self.rh_token = rh_token
        self.cfg = rh_config.get_config()
        self.dest_dir = dest_dir

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

        from_date = self.cfg.get("from_date")
        if from_date:
            params.append(f'from={from_date}')
        
        to_date = self.cfg.get("to_date")
        if to_date:
            params.append(f'to={to_date}')
        

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
    
    # def get_detections(self, full_url):        
    #     t = 100 if "take=all" in full_url else 10

    #     response = requests.get(
    #         full_url,
    #         headers={'Authorization': f'Bearer {self.rh_token}'},
    #         timeout=t
    #     )
    #     if response.status_code != 200:
    #         print("Error: ", response.status_code)
    #         print("Error occured while fetching detections.")
    #         print(response.text)
    #         return
        
    #     items = response.json()['items']
    #     return items
    
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
                break
            elif self.cfg.get("tactic") == "limit" and len(items) >= self.cfg.get("num_of_imgs"):
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
                device_id = detection['title'].split(' ')[4].strip('.')
                frame_name = detection['frames'][0]['path']
                id_ = detection['id']
                image_time = detection['createdAt'].replace(":", "-").split(".")[0]
                frame_path = f'{image_time}_{id_}_{frame_name}.png'

                frame_full_path = os.path.abspath(os.path.join(self.dest_dir+'/images', frame_path))

                # Create a new dictionary with the required fields
                detection_info = {
                    "id": detection["id"],
                    "createdAt": detection["createdAt"],
                    "teamId": detection["teamId"],
                    "robotId": detection["robotId"],
                    "deviceId": device_id,
                    "appId": detection["appId"],
                    "tags": detection["tags"],
                    "frame": frame_name,
                    "framePath": frame_full_path,
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
        
        # check if the folder exists
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir) # exist_ok=True)
        
        # Save the filtered detections to a JSON file
        with open(self.dest_dir + '/detections.json', 'w') as f:
            json.dump(detection_infos, f)
        
        print(f"Saved {len(detection_infos)} detections to {self.dest_dir}/detections.json")

    async def download_frames(self, item_info, session):
        id_ = item_info['id']

        image_time = item_info['createdAt'].replace(":", "-").split(".")[0]

        image_data = item_info['frames'][0]
        if isinstance(image_data, dict):
            image_name = image_data["path"]
        else:
            image_name = image_data

        url = f"https://robothub.luxonis.com/api/detection/{id_}/download/{image_name}"
        filename = os.path.join(self.dest_dir, f'{image_time}_{id_}_{image_name}.png')
        async with session.get(url) as response:
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await response.read())
    
    async def download_images(self, items):
        # check if the folder exists
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir) # exist_ok=True)
        
        # Download the frames
        async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {self.rh_token}'}) as session:
            await asyncio.gather(
                *[self.download_frames(image, session) for image in items]
            )
    
    def run_async_download(self):
        # url = self.build_url()
        # detections = self.get_detections(url)
        detections = self.get_all_detections()
        filtered_detections = self.filter_detections(detections)
        asyncio.run(self.download_images(filtered_detections))
        self.save_detections_info(filtered_detections)
