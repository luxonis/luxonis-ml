import requests
import aiohttp
import aiofiles
import os
import json
import asyncio

from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("TOKEN")

def get_all_robot_ids(rh_token):
    response = requests.get(
        'https://robothub.luxonis.com/api/robot?status=connected',
        headers={'Authorization': f'Bearer {rh_token}'},
        timeout=10
    )

    robots = response.json()['robots']
    robot_ids = [robot['id'] for robot in robots]
    return robot_ids

def build_params(rh_token, num_of_imgs, from_date, to_date, robot_ids, tags):
    params = {}
    
    if robot_ids == "all":
        robot_ids = get_all_robot_ids(rh_token)
    
    if num_of_imgs is not None:
        params['take'] = num_of_imgs if num_of_imgs != "all" else "all"
    
    if from_date is not None:
        params['from'] = from_date
    if to_date is not None:
        params['to'] = to_date
    
    params['robot'] = "&robot=".join(robot_ids)
    if tags is not None:
        params['tag'] = "&tag=".join(tags) 
    params['hasFrames'] = 'true'

    return params

def build_url(rh_token, num_of_imgs, from_date, to_date, robot_ids, tags):
    base_url = 'https://robothub.luxonis.com/api/detection/search'
    
    if robot_ids == "all":
        robot_ids = get_all_robot_ids(rh_token)
    
    # Construct the URL with all parameters
    params = []
    if num_of_imgs is not None:
        params.append(f'take={num_of_imgs if num_of_imgs != "all" else "all"}')
    if from_date is not None:
        params.append(f'from={from_date}')
    if to_date is not None:
        params.append(f'to={to_date}')
    for robot_id in robot_ids:
        params.append(f'robot={robot_id}')
    if tags:
        for tag in tags:
            params.append(f'tag={tag}')
    params.append('hasFrames=true')

    full_url = base_url + '?' + '&'.join(params)
    return full_url

def save_detection_info(dest_dir, items):
    detection_infos = []
    # Filter and reformat the detections
    for detection in items:
        device_id = detection['title'].split(' ')[4].strip('.')
        frame_name = detection['frames'][0]['path']
        id_ = detection['id']
        image_time = detection['createdAt'].replace(":", "-").split(".")[0]
        frame_path = f'{image_time}_{id_}_{frame_name}.png'

        frame_full_path = os.path.abspath(os.path.join(dest_dir, frame_path))

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
        detection_infos.append(detection_info)
    
    # Save the filtered detections to a JSON file
    with open(dest_dir + '/detections.json', 'w') as f:
        json.dump(detection_infos, f)

async def download_frames(item_info, dest_dir, session):
    id_ = item_info['id']

    image_time = item_info['createdAt'].replace(":", "-").split(".")[0]

    image_data = item_info['frames'][0]
    if isinstance(image_data, dict):
        image_name = image_data["path"]
    else:
        image_name = image_data

    url = f"https://robothub.luxonis.com/api/detection/{id_}/download/{image_name}"
    filename = os.path.join(dest_dir, f'{image_time}_{id_}_{image_name}.png')
    async with session.get(url) as response:
        async with aiofiles.open(filename, "wb") as f:
            await f.write(await response.read())

async def get_detections(rh_token, dest_dir, full_url, app_ids):
    response = requests.get(
        full_url,
        headers={'Authorization': f'Bearer {rh_token}'},
        timeout=10
    )
    if response.status_code != 200:
        print("Error: ", response.status_code)
        print("Error occured while fetching detections.")
        print(response.text)
        return
    
    items = response.json()['items']

    # filter detections by app_id
    if app_ids != "all":
        items = [item for item in items if item['appId'] in app_ids]

    async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {rh_token}'}) as session:
        await asyncio.gather(
            *[download_frames(image, dest_dir, session) for image in items]
        )
    
    save_detection_info(dest_dir, items)

def run_async_download(rh_token, num_of_imgs, from_date, to_date, robot_ids, app_ids, tags, dest_dir="./tmp"):
    # params = build_params(rh_token, num_of_imgs, from_date, to_date, robot_ids, tags)
    full_url = build_url(rh_token, num_of_imgs, from_date, to_date, robot_ids, tags)
    asyncio.run(get_detections(rh_token, dest_dir, full_url, app_ids))


def main():
    rh_token = TOKEN
    robots_ids = ["a44bbb6d-c582-43bf-9514-033abd05a5eb","8cae84e6-9fe6-4756-b2bf-e6a4d5ee4865"]
    # robots_ids = "all"
    from_date = "2023-09-01"
    to_date = "2023-09-02"
    tags = ["pred_close","low_confidence"]

    dest_dir = f"./tmp/{from_date}_images"
    os.makedirs(dest_dir, exist_ok=True)

    run_async_download(rh_token, None, from_date, to_date, robots_ids, "all", tags, dest_dir)

if __name__ == '__main__':
    main()