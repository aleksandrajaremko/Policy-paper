#!/usr/bin/env python3
"""
Quick-start test: verify BDL API connectivity and explore available data.
Run this BEFORE the main downloader to make sure everything works.

Usage:  python bdl_quickstart_test.py [--api-key YOUR_KEY]
"""

import requests
import json
import sys
import time

BASE_URL = "https://bdl.stat.gov.pl/api/v1"


def test_connection(api_key=None):
    """Test basic API connectivity."""
    print("=" * 60)
    print("BDL API Quick-Start Test")
    print("=" * 60)

    headers = {}
    if api_key:
        headers["X-ClientId"] = api_key
        print(f"✓ Using API key: {api_key[:8]}...")
    else:
        print("⚠ No API key — using anonymous access (lower rate limits)")
        print("  Register at: https://api.stat.gov.pl/Home/BdlApi?lang=en")

    # Test 1: Subjects
    print("\n[Test 1] Fetching top-level subjects...")
    resp = requests.get(
        f"{BASE_URL}/subjects",
        params={"format": "json", "lang": "en"},
        headers=headers,
        timeout=15,
    )
    if resp.status_code == 200:
        subjects = resp.json().get("results", [])
        print(f"  ✓ Found {len(subjects)} top-level subjects:")
        for s in subjects:
            print(f"    {s['id']}: {s['name']}")
    else:
        print(f"  ✗ HTTP {resp.status_code}: {resp.text[:200]}")
        return False

    time.sleep(0.3)

    # Test 2: Units at gmina level (sample)
    print("\n[Test 2] Fetching gmina units (sample, level=6)...")
    resp = requests.get(
        f"{BASE_URL}/units",
        params={"format": "json", "lang": "en", "level": 6,
                "page-size": 5, "page": 0},
        headers=headers,
        timeout=15,
    )
    if resp.status_code == 200:
        data = resp.json()
        total = data.get("totalRecords", 0)
        units = data.get("results", [])
        print(f"  ✓ Total gminas in BDL: {total}")
        print(f"  Sample units:")
        for u in units:
            print(f"    {u['id']}: {u['name']} (level {u.get('level')})")
    else:
        print(f"  ✗ HTTP {resp.status_code}")
        return False

    time.sleep(0.3)

    # Test 3: Search for a variable (population)
    print("\n[Test 3] Searching for 'population' variables at gmina level...")
    resp = requests.get(
        f"{BASE_URL}/variables/search",
        params={"format": "json", "lang": "en",
                "name": "population", "level": 6,
                "page-size": 10},
        headers=headers,
        timeout=15,
    )
    if resp.status_code == 200:
        data = resp.json()
        total = data.get("totalRecords", 0)
        variables = data.get("results", [])
        print(f"  ✓ Found {total} population-related variables at gmina level")
        for v in variables[:5]:
            var_id = v.get("id")
            var_name = v.get("n1", v.get("name", "?"))
            years = v.get("years", [])
            year_range = (
                f"{years[0]}-{years[-1]}" if years else "?"
            )
            print(
                f"    var_id={var_id}: {var_name} "
                f"[{year_range}, {len(years)} years]"
            )
    else:
        print(f"  ✗ HTTP {resp.status_code}")
        return False

    time.sleep(0.3)

    # Test 4: Download sample data
    if variables:
        sample_var = variables[0]["id"]
        print(f"\n[Test 4] Downloading sample data (var={sample_var}, "
              f"gmina level, year=2020)...")
        resp = requests.get(
            f"{BASE_URL}/data/by-variable/{sample_var}",
            params={"format": "json", "lang": "en",
                    "unit-level": 6, "year": 2020,
                    "page-size": 5},
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("totalRecords", 0)
            results = data.get("results", [])
            print(f"  ✓ {total} gminas have data for this variable in 2020")
            for r in results[:3]:
                vals = r.get("values", [])
                val_str = (
                    f"{vals[0].get('val')}" if vals else "N/A"
                )
                print(
                    f"    {r['id']}: {r['name']} → {val_str}"
                )
        else:
            print(f"  ✗ HTTP {resp.status_code}")

    # Test 5: Check aggregates (gmina types)
    print("\n[Test 5] Fetching aggregates (gmina types)...")
    resp = requests.get(
        f"{BASE_URL}/aggregates",
        params={"format": "json", "lang": "en"},
        headers=headers,
        timeout=15,
    )
    if resp.status_code == 200:
        aggs = resp.json()
        print(f"  ✓ Available aggregates:")
        for a in aggs:
            print(f"    {a.get('id')}: {a.get('name')}")
    else:
        print(f"  ✗ HTTP {resp.status_code}")

    print("\n" + "=" * 60)
    print("✓ All tests passed! The API is accessible.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run discovery:  python bdl_gmina_downloader.py --discover")
    print("  2. Review the catalogue: bdl_output/bdl_variable_catalogue_gmina.csv")
    print("  3. Download data:  python bdl_gmina_downloader.py --download")
    return True


if __name__ == "__main__":
    key = None
    if len(sys.argv) > 2 and sys.argv[1] == "--api-key":
        key = sys.argv[2]
    test_connection(key)
