#!/usr/bin/env python3

'''
This script generates a sequence of events based on the provided configuration.
It reads the configuration from a file, generates events according to the specified parameters,
and writes the events to an output file.
'''
import argparse
import json
import datetime
import ipaddress
from typing import List, Dict, Any
import csv
import random


'''''
Event structure example:
{
    "event_type": "user_login_event",
    "geoip": {
        "country_code": "IN",
        "continent_code": "AS",
        "region_name": "New Delhi",
        "region_code": "IN",
        "timezone": "Asia/Kolkata",
        "latitude": 28.6139,
        "longitude": 77.2090,
    },
    "client_ip": "103.27.8.123",
    "user_agent": {
        "browser": "Safari",
        "browser_version": "14.0",
        "os": "MacOS",
        "os_version": "11.0",
        "device": "MacBook-Pro.local"
    },
    "id": "event_12345",
    "resource_id": 1,
    "resource_name": "user_1",
    "resource_type": "user",
    "timestamp": "2025-10-01T12:00:00",
    "enterprise_id": "enterprise_3",
    "success": True
}
'''''

REGIONS = {
    "IN": {
        "country_code": "IN",
        "continent_code": "AS",
        "region_name": "New Delhi",
        "region_code": "IN",
        "timezone": "Asia/Kolkata",
        "lat_range": (28.4, 28.9),  # Approximate latitude range for New Delhi
        "long_range": (76.8, 77.4),  # Approximate longitude range for New Delhi
        "ip_range": [
            "103.27.8.0/21",
            "103.248.96.0/22",
            "182.64.0.0/16"
        ],
        "systems": [
            "MacBook-Pro.local",
            "Windows-PC",
            "Linux-Workstation",
            "Office-Desktop"
        ]
    },
    "US": {
        "country_code": "US",
        "continent_code": "NA",
        "region_name": "New York",
        "region_code": "US",
        "timezone": "America/New_York",
        "lat_range": (40.5, 43.0),  # Approximate latitude range for New York State
        "long_range": (-79.8, -71.8),  # Approximate longitude range for New York State
        "ip_range": [
            "104.192.0.0/16",
            "157.131.0.0/16",
            "23.106.0.0/16"
        ],
        "systems": [
            "US-MacBook.local",
            "US-Windows-PC",
            "US-Linux-Box",
            "US-Office-Desktop"
        ]
    },
    "KR": {
        "country_code": "KR",
        "continent_code": "AS",
        "region_name": "Seoul",
        "region_code": "KR",
        "timezone": "Asia/Seoul",
        "lat_range": (37.0, 38.0),  # Approximate latitude range for Seoul
        "long_range": (126.0, 127.5),  # Approximate longitude range for Seoul
        "ip_range": [
            "211.45.0.0/16",
            "175.223.0.0/16",
            "222.99.0.0/16"
        ], 
        "systems": [
            "US-MacBook.local",
            "US-Windows-PC",
            "US-Linux-Box",
            "US-Office-Desktop"
        ]
    },
    "RU": {
        "country_code": "RU",
        "continent_code": "EU",
        "region_name": "Moscow",
        "region_code": "RU",
        "timezone": "Europe/Moscow",
        "lat_range": (55.5, 56.0),  # Approximate latitude range for Moscow
        "long_range": (37.0, 38.0),  # Approximate longitude range for Moscow
        "ip_range": [
            "95.165.0.0/16",
            "178.140.0.0/16",
            "213.87.0.0/16"
        ], 
        "systems": [
            "MacBook-Pro.local",
            "Windows-PC",
            "Linux-Workstation",
            "Office-Desktop"
        ]
    },
    "FR": {
        "country_code": "FR",
        "continent_code": "EU",
        "region_name": "Paris",
        "region_code": "FR",
        "timezone": "Europe/Paris",
        "lat_range": (48.8, 49.0),  # Approximate latitude range for Paris
        "long_range": (2.0, 2.6),  # Approximate longitude range for Paris
        "ip_range": [
            "62.147.0.0/16",
            "90.63.0.0/16",
            "195.154.0.0/16"
        ],
        "systems": [
            "Airport-Terminal",
            "Hotel-PC",
            "Shared-Workstation"
        ]
    },
}

USERAGENT = {
    "MacBook-Pro.local": [
        {"browser": "Safari", "browser_version": "14.0", "os": "MacOS", "os_version": "11.0"},
        {"browser": "Chrome", "browser_version": "114.0", "os": "MacOS", "os_version": "12.0"},
        {"browser": "Firefox", "browser_version": "113.0", "os": "MacOS", "os_version": "13.0"},
    ],
    "Windows-PC": [
        {"browser": "Chrome", "browser_version": "114.0", "os": "Windows", "os_version": "10"},
        {"browser": "Edge", "browser_version": "114.0", "os": "Windows", "os_version": "11"},
        {"browser": "Firefox", "browser_version": "113.0", "os": "Windows", "os_version": "10"},
    ],
    "Linux-Workstation": [
        {"browser": "Chrome", "browser_version": "114.0", "os": "Linux", "os_version": "Ubuntu 22.04"},
        {"browser": "Firefox", "browser_version": "113.0", "os": "Linux", "os_version": "Fedora 38"},
    ],
    "Office-Desktop": [
        {"browser": "Chrome", "browser_version": "114.0", "os": "Windows", "os_version": "10"},
        {"browser": "Firefox", "browser_version": "113.0", "os": "Linux", "os_version": "Ubuntu 20.04"},
        {"browser": "Safari", "browser_version": "14.0", "os": "MacOS", "os_version": "11.0"},
    ]
}

def generate_random_ip(ip_range: str) -> str:
    """
    Generate a random IP address within the given CIDR range.
    """
    network = ipaddress.ip_network(ip_range)
    network_size = network.num_addresses
    host_offset = random.randint(0, network_size - 1)
    random_ip = network.network_address + host_offset
    return str(random_ip)


def generate_event(event_type: str, 
                   region_key: str,
                   resource_id: int, 
                   resource_name: str,
                   resource_type: str, 
                   timestamp: str, 
                   enterprise_id: str, 
                   success: bool = True) -> Dict[str, Any]:
    region_data = REGIONS[region_key]

    #Generate random latitude and longitude within the region's range
    latitude = round(random.uniform(*region_data["lat_range"]), 4)
    longitude = round(random.uniform(*region_data["long_range"]), 4)

    timestamp = generate_timestamp(timestamp)
    # Choose a random IP from the region's range
    ip_range = random.choice(region_data["ip_range"])
    client_ip = generate_random_ip(ip_range)
    # Use a random system name from the region's systems
    system_name = random.choice(region_data["systems"])
    user_agent = random.choice(USERAGENT.get(system_name, USERAGENT["MacBook-Pro.local"])).copy()  # Copy to avoid mutating the original
    user_agent["device"] = system_name
    
    return {
        "event_type": event_type,
        "geoip": {
            "country_code": region_data["country_code"],
            "continent_code": region_data["continent_code"],
            "region_name": region_data["region_name"],
            "region_code": region_data["region_code"],
            "timezone": region_data["timezone"],
            "latitude": latitude,
            "longitude": longitude,
        },
        "client_ip": client_ip,
        "user_agent": user_agent,
        "resource_id": resource_id,
        "resource_name": resource_name,
        "resource_type": resource_type,
        "timestamp": timestamp,
        "enterprise_id": enterprise_id,
        "success": success,  # Placeholder, can be randomized if needed
    }

def generate_timestamp(base_date: datetime.date) -> str:
    hour = random.randint(9, 21)
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    dt = datetime.datetime.combine(base_date, datetime.time(hour, minute, second))
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def main():
    # Parse command line arguments
    print("Generating events for anomaly detection")
    parser = argparse.ArgumentParser(description= 'Generate events for anomaly detection' )
    parser.add_argument('-n', '--num_events', type=int, default=100, help='Number of events to generate(per user)')
    parser.add_argument('-m', '--num_users', type=int, default=10, help='Number of users')
    parser.add_argument('-o', '--output_file', type=str, default='data/events.csv', help='out file name')
    parser.add_argument('--success-rate', type=float, default=0.85, help='Success rate for login attempts (0.0-1.0)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date for events in YYYY-MM-DD format, defaults to today')

    args = parser.parse_args()
    # Parse or default the start date
    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = datetime.datetime.now().date()
    print(f"Generating {args.num_events} events for {args.num_users} users starting from {start_date}")

    # Generate usernames
    usernames = [f"user_{i+1}" for i in range(args.num_users) ]
    userids = [random.randint(1, 1000000) for _ in range(args.num_users)]

    # Assign regions to users (20% in IN, 20% in US, 20% in KR, 20% in RU, 20% in FR)
    num_regions = 5
    users_per_region = args.num_users // num_regions
    remainder = args.num_users % num_regions
    region_keys = ["IN", "US", "KR", "RU", "FR"]
    regions = []
    for i, region in enumerate (region_keys):
        count = users_per_region + (1 if i < remainder else 0)
        regions. extend ( [region] * count)
    random.shuffle(regions)

    # Generate events
    events = []
    for i in range(args.num_users):
        username = usernames[i]
        userid = userids[i]
        region_key = regions[i]
        resource_id = userid
        resource_name = username
        resource_type = "user"
        num_success_events = int(args.num_events * args.success_rate)
        num_failure_events = args.num_events - num_success_events
        success_event_flags = [True] * num_success_events + [False] * num_failure_events

        random.shuffle(success_event_flags)  # Shuffle success flags to mix successes and failures

        for j in range(args.num_events):
            timestamp = start_date + datetime.timedelta(days=j)
            success = success_event_flags[j]
            event = generate_event(
                event_type="user_login_event",
                region_key=region_key,
                resource_id=resource_id,
                resource_name=resource_name,
                resource_type=resource_type,
                timestamp=timestamp,
                enterprise_id=f"enterprise_{random.randint(1, 10)}",
                success=success
            )
            events.append(event)
    # Write events to CSV file
    with open(args.output_file, 'w', newline='') as csvfile:    
        fieldnames = list(events[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            # Serialize geoip and useragent as JSON strings
            event = event.copy()  # Avoid mutating the original event
            event["geoip"] = json.dumps(event["geoip"])
            event["user_agent"] = json.dumps(event["user_agent"])
            writer.writerow(event)

if __name__ == "__main__":
    main()