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
from datetime import datetime, timedelta


'''''
Event structure example:
{
    "event_type": "user_login_event",
    "resource_id": "user_1",
    "resource_name": "user_1",
    "resource_type": "user",
    "enterprise_id": "enterprise_3",
    "timestamp": "2025-05-01T12:00:00",
    "client_ip": "103.27.8.123",
    "geoip": {
        "country_code": "IN",
        "continent_code": "AS",
        "region_name": "New Delhi",
        "region_code": "IN",
        "timezone": "Asia/Kolkata",
        "latitude": 28.6139,
        "longitude": 77.2090
    },
    "user_agent": {
        "browser": "Safari",
        "browser_version": "14.0",
        "os": "MacOS",
        "os_version": "11.0",
        "device": "MacBook-Pro.local"
    },
    "success": "True"
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


def generate_anomalous_events(num_events=1000, anomaly_ratio=0.15):
    """
    Generate events with realistic anomalies:
    - Each user has unique characteristics (home region, preferred devices, browsers, working hours)
    - Most logins from user's home region (randomly assigned)
    - Occasionally, logins from a different region (anomaly)
    - Other anomaly types (timing, device, browser, failed login) are preserved
    """
    events = []
    region_keys = list(REGIONS.keys())

    # Create unique user profiles with personalized characteristics
    normal_users = [f"user_{i}" for i in range(1, 21)]
    user_profiles = {}
    
    for user in normal_users:
        # Assign each user a home region
        home_region = random.choice(region_keys)
        
        # Assign preferred devices (2-3 devices per user)
        available_devices = REGIONS[home_region]["systems"] + ["Mobile-Phone", "Office-Desktop"]
        preferred_devices = random.sample(available_devices, min(3, len(available_devices)))
        
        # Assign preferred browsers (1-2 browsers per user)
        all_browsers = ["Chrome", "Firefox", "Safari", "Edge"]
        preferred_browsers = random.sample(all_browsers, random.randint(1, 2))
        
        # Assign working hours (each user has different patterns)
        work_start = random.randint(7, 10)  # 7-10 AM start
        work_end = random.randint(16, 20)   # 4-8 PM end
        
        # Assign preferred OS
        preferred_os = random.choice(["MacOS", "Windows", "Linux"])
        
        # Assign enterprise (some users work for same company)
        enterprise_id = f"enterprise_{random.randint(1, 10)}"
        
        user_profiles[user] = {
            "home_region": home_region,
            "preferred_devices": preferred_devices,
            "preferred_browsers": preferred_browsers,
            "work_start": work_start,
            "work_end": work_end,
            "preferred_os": preferred_os,
            "enterprise_id": enterprise_id,
            "success_rate": random.uniform(0.85, 0.98)  # Individual success rates
        }
    
    normal_enterprises = [f"enterprise_{i}" for i in range(1, 11)]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    failed_attempts = {}

    for i in range(num_events):
        user = random.choice(normal_users)
        profile = user_profiles[user]
        is_anomaly = random.random() < anomaly_ratio

        # Choose region for this event
        if is_anomaly and random.random() < 0.5:
            # Location anomaly: pick a region that's NOT the user's home
            region_choices = [r for r in region_keys if r != profile["home_region"]]
            region_key = random.choice(region_choices)
        else:
            region_key = profile["home_region"]

        region_data = REGIONS[region_key]

        # Timing - personalized based on user's working hours
        if is_anomaly and random.random() < 0.3:
            # Anomalous timing: outside working hours
            if random.random() < 0.5:
                hour = random.randint(2, 6)  # Night time
            else:
                hour = random.randint(21, 23)  # Late night
        else:
            # Normal timing: within user's working hours
            hour = random.randint(profile["work_start"], profile["work_end"])

        day_offset = random.randint(0, 90)
        event_date = start_date + timedelta(days=day_offset, hours=hour)
        timestamp = event_date.strftime("%Y-%m-%dT%H:%M:%S")

        # IP - based on region
        ip_range = random.choice(region_data["ip_range"])
        client_ip = generate_random_ip(ip_range)

        # Device & browser - personalized based on user preferences
        if is_anomaly and random.random() < 0.3:
            # Anomalous device/browser
            device = random.choice(["Unknown-Device", "Mobile-Phone", "Public-Kiosk", "Internet-Cafe"])
            browser = random.choice(["Unknown-Browser", "Bot-Client", "Script-Engine"])
        else:
            # Normal: use user's preferred devices/browsers (with some variation)
            if random.random() < 0.7:  # 70% chance to use preferred device
                device = random.choice(profile["preferred_devices"])
            else:
                device = random.choice(REGIONS[region_key]["systems"] + ["Mobile-Phone", "Office-Desktop"])
                
            if random.random() < 0.8:  # 80% chance to use preferred browser
                browser = random.choice(profile["preferred_browsers"])
            else:
                browser = random.choice(["Chrome", "Firefox", "Safari", "Edge"])

        # User agent - personalized
        user_agent = {
            "browser": browser,
            "browser_version": f"{random.randint(110, 115)}.0",
            "os": profile["preferred_os"],
            "os_version": f"{random.randint(10, 14)}.0",
            "device": device
        }

        # Success/failure - personalized based on user's success rate
        if is_anomaly:
            # Higher chance of failure for anomalies
            success = "False" if random.random() < 0.6 else "True"
            if success == "False":
                failed_attempts[client_ip] = failed_attempts.get(client_ip, 0) + 1
        else:
            # Normal events: use user's personal success rate
            success = "False" if random.random() < (1 - profile["success_rate"]) else "True"

        event = {
            "event_type": "user_login_event",
            "resource_id": user,
            "resource_name": user,
            "resource_type": "user",
            "enterprise_id": profile["enterprise_id"],
            "timestamp": timestamp,
            "client_ip": client_ip,
            "geoip": json.dumps({
                "country_code": region_data["country_code"],
                "continent_code": region_data["continent_code"],
                "region_name": region_data["region_name"],
                "region_code": region_data["region_code"],
                "timezone": region_data["timezone"],
                "latitude": round(random.uniform(*region_data["lat_range"]), 4),
                "longitude": round(random.uniform(*region_data["long_range"]), 4)
            }),
            "user_agent": json.dumps(user_agent),
            "success": success
        }
        events.append(event)

    # Add burst anomalies (multiple rapid failed attempts)
    num_burst_anomalies = int(num_events * 0.02)
    for _ in range(num_burst_anomalies):
        suspicious_ips = [ip for ip, count in failed_attempts.items() if count >= 3]
        if suspicious_ips:
            ip = random.choice(suspicious_ips)
            for attempt in range(random.randint(3, 5)):
                region_key = random.choice(region_keys)
                region_data = REGIONS[region_key]
                burst_event = {
                    "event_type": "user_login_event",
                    "resource_id": random.choice(normal_users),
                    "resource_name": f"user_{random.randint(1, 20)}",
                    "resource_type": "user",
                    "enterprise_id": random.choice(normal_enterprises),
                    "client_ip": ip,
                    "success": "False",
                    "user_agent": json.dumps({
                        "browser": random.choice(["Unknown-Browser", "Bot-Client", "Script-Engine"]),
                        "browser_version": f"{random.randint(110, 115)}.0",
                        "os": "Unknown-OS",
                        "os_version": "0.0",
                        "device": random.choice(["Unknown-Device", "Mobile-Phone", "Public-Kiosk", "Internet-Cafe"])
                    }),
                    "geoip": json.dumps({
                        "country_code": region_data["country_code"],
                        "continent_code": region_data["continent_code"],
                        "region_name": region_data["region_name"],
                        "region_code": region_data["region_code"],
                        "timezone": region_data["timezone"],
                        "latitude": round(random.uniform(*region_data["lat_range"]), 4),
                        "longitude": round(random.uniform(*region_data["long_range"]), 4)
                    }),
                    "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30), minutes=attempt * random.randint(1, 3))).strftime("%Y-%m-%dT%H:%M:%S")
                }
                events.append(burst_event)

    return events


def save_events_to_csv(events, filename="data/events_with_anomalies.csv"):
    """Save events to CSV file"""
    if not events:
        return
    
    fieldnames = events[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
    
    print(f"Generated {len(events)} events with anomalies saved to {filename}")
    
    # Print statistics
    total_events = len(events)
    failed_events = sum(1 for e in events if e['success'] == 'False')
    
    # Count events from different regions (geoip is now a JSON string)
    region_counts = {}
    for e in events:
        geoip_data = json.loads(e['geoip'])
        country = geoip_data['country_code']
        region_counts[country] = region_counts.get(country, 0) + 1
    
    print(f"\nDataset Statistics:")
    print(f"Total events: {total_events}")
    print(f"Failed logins: {failed_events} ({failed_events/total_events*100:.1f}%)")
    print(f"Events by region:")
    for country, count in sorted(region_counts.items()):
        print(f"  {country}: {count} ({count/total_events*100:.1f}%)")

def main():
    # Parse command line arguments
    print("Generating events for anomaly detection")
    parser = argparse.ArgumentParser(description= 'Generate events for anomaly detection' )
    parser.add_argument('-n', '--num_events', type=int, default=10, help='Number of events to generate(per user)')
    parser.add_argument('-m', '--num_users', type=int, default=100, help='Number of users')
    parser.add_argument('-o', '--output_file', type=str, default='data/events_with_anomalies.csv', help='out file name')
    parser.add_argument('--success-rate', type=float, default=0.85, help='Success rate for login attempts (0.0-1.0)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date for events in YYYY-MM-DD format, defaults to today')
    parser.add_argument('--anomaly-ratio', type=float, default=0.07, help='Ratio of anomalies to total events (0.0-1.0)')

    args = parser.parse_args()
    
    # Calculate total events
    total_events = args.num_events * args.num_users
    print(f"Generating {args.num_events} events for {args.num_users} users = {total_events} total events")
    print(f"Anomaly ratio: {args.anomaly_ratio*100}% ({int(total_events * args.anomaly_ratio)} anomalies)")
    
    # Generate events with anomalies
    events = generate_anomalous_events(num_events=total_events, anomaly_ratio=args.anomaly_ratio)
    save_events_to_csv(events, args.output_file)

if __name__ == "__main__":
    main()