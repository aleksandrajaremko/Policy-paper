#!/usr/bin/env python3
"""
Diagnostic: inspect raw JSON responses from BDL API.
Run this to see the exact field names the API returns,
so we can fix the downloader script.

Usage: python bdl_diagnose.py [--api-key YOUR_KEY]
"""

import requests
import json
import sys
import time

BASE = "https://bdl.stat.gov.pl/api/v1"

headers = {}
if len(sys.argv) > 2 and sys.argv[1] == "--api-key":
    headers["X-ClientId"] = sys.argv[2]

def fetch(endpoint, params=None):
    if params is None:
        params = {}
    params["format"] = "json"
    params["lang"] = "en"
    resp = requests.get(f"{BASE}/{endpoint}", params=params, headers=headers, timeout=15)
    return resp.json()

print("=" * 60)
print("BDL API Diagnostic — Raw JSON Structures")
print("=" * 60)

# 1. Subjects
print("\n[1] SUBJECTS — top level (first 3):")
data = fetch("subjects")
for s in data.get("results", [])[:3]:
    print(json.dumps(s, indent=2, ensure_ascii=False))
    print()

time.sleep(0.3)

# 2. Pick first subject's children
first_k = data.get("results", [{}])[0].get("id", "K3")
print(f"\n[2] SUBJECTS — children of {first_k} (first 3):")
data2 = fetch("subjects", {"parent-id": first_k})
for s in data2.get("results", [])[:3]:
    print(json.dumps(s, indent=2, ensure_ascii=False))
    print()

time.sleep(0.3)

# 3. Variables — search for "population" at level 6
print("\n[3] VARIABLES — search 'population', level=6 (first 3):")
data3 = fetch("variables/search", {"name": "population", "level": 6, "page-size": 3})
print(f"Total records: {data3.get('totalRecords', '?')}")
for v in data3.get("results", [])[:3]:
    print(json.dumps(v, indent=2, ensure_ascii=False))
    print()

time.sleep(0.3)

# 4. Variables — by subject (pick one with hasVariables)
print("\n[4] VARIABLES — by subject-id, level=6 (first 3):")
# Try a known subject
for test_subj in ["P2137", "P3183", "P1834"]:
    data4 = fetch("variables", {"subject-id": test_subj, "level": 6, "page-size": 3})
    results = data4.get("results", [])
    if results:
        print(f"Subject {test_subj}: {data4.get('totalRecords', '?')} vars")
        for v in results[:2]:
            print(json.dumps(v, indent=2, ensure_ascii=False))
            print()
        break
    time.sleep(0.3)

time.sleep(0.3)

# 5. Data — sample download for first variable found
print("\n[5] DATA — by-variable sample (first 2 units):")
vars_found = data3.get("results", [])
if vars_found:
    sample_id = vars_found[0]["id"]
    data5 = fetch(f"data/by-variable/{sample_id}",
                  {"unit-level": 6, "year": 2020, "page-size": 2})
    print(f"Variable: {sample_id}")
    print(f"Total records: {data5.get('totalRecords', '?')}")
    for r in data5.get("results", [])[:2]:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        print()

print("=" * 60)
print("DONE — Copy/paste the output above and share it with Claude")
print("so the downloader script can be fixed to match these fields.")
print("=" * 60)
