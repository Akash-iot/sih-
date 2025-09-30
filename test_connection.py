#!/usr/bin/env python3
"""
Test script to verify ETHEREYE API connection
"""
import requests
import json

def test_api_connection():
    """Test API connection and endpoints"""
    base_url = "http://127.0.0.1:8001"
    
    print("🔍 Testing ETHEREYE API Connection...")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        "/",
        "/health",
        "/api/v1/live/gas",
        "/api/v1/live/prices", 
        "/api/v1/analytics/overview"
    ]
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            print(f"\n📡 Testing: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Status: {response.status_code}")
                data = response.json()
                print(f"📄 Response: {json.dumps(data, indent=2)[:200]}...")
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text}")
                
        except requests.ConnectionError:
            print(f"❌ Connection failed - Server not running on {base_url}")
            return False
        except requests.Timeout:
            print(f"❌ Request timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API connection test completed!")
    return True

if __name__ == "__main__":
    test_api_connection()