#!/usr/bin/env python3
import requests

try:
    response = requests.get('http://localhost:8000/', timeout=5)
    if response.status_code == 200:
        print('✅ Backend Server: RUNNING')
        data = response.json()
        print(f'   Name: {data.get("name", "Unknown")}')
        print(f'   Status: {data.get("status", "Unknown")}')
    else:
        print(f'⚠️ Backend Server: Error {response.status_code}')
except Exception as e:
    print(f'❌ Backend Server: NOT RUNNING - {e}')
    print('   Need to start: cd ../backend && python -m uvicorn simple_main:app --reload')