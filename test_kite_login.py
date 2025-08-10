#!/usr/bin/env python3
"""
Test script for Kite login functionality
This script tests the ChromeDriver and Selenium setup without running full training.
"""

import os
import sys
from datetime import datetime

# Setup path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
os.chdir(basepath)

def test_chromedriver():
    """Test ChromeDriver compatibility"""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    
    print("🔧 Testing ChromeDriver...")
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    service = Service('./chromedriver')
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print(f"✅ ChromeDriver working: {driver.capabilities['chrome']['chromedriverVersion']}")
        print(f"✅ Chrome version: {driver.capabilities['browserVersion']}")
        
        # Test navigation
        driver.get("https://www.google.com")
        print(f"✅ Navigation test successful: {driver.title}")
        
        driver.quit()
        return True
    except Exception as e:
        print(f"❌ ChromeDriver test failed: {e}")
        return False

def test_kite_login_setup():
    """Test Kite login setup without actual login"""
    print("\n🔐 Testing Kite login setup...")
    try:
        import kitelogin
        
        # Check if we can access the login function
        print("✅ Kite login module imported successfully")
        
        # Test URL generation
        from kiteconnect import KiteConnect
        api_key = "wh7m5jcdtj4g57oh"  # From the code
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        print(f"✅ Login URL generated: {login_url[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Kite login setup failed: {e}")
        return False

def main():
    print("🚀 Testing Kite Login Dependencies")
    print("=" * 50)
    
    success = True
    
    # Test ChromeDriver
    if not test_chromedriver():
        success = False
    
    # Test Kite setup
    if not test_kite_login_setup():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Kite login should work correctly.")
        print("💡 You can now run: python model_trainer.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()