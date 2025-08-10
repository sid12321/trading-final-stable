from lib import *
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pyotp
from kiteconnect import KiteConnect
from parameters import API_KEY, API_SECRET, USER_ID, PASSWORD, TOTP_KEY

def login_to_kite():
    """
    Automates the login process to Zerodha's Kite platform using Selenium.
    
    Returns:
        KiteConnect: An authenticated KiteConnect object
    """
    # Initialize Kite Connect
    kite = KiteConnect(api_key=API_KEY)
    
    # Get the login URL
    login_url = kite.login_url()
    print(f"Login URL: {login_url}")
    
    # Initialize Chrome with Selenium
    chrome_options = Options()
    # Uncomment the line below to run in headless mode (no UI)
    # chrome_options.add_argument("--headless")
    
    # Add Chrome options to avoid snap sandboxing issues
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    time.sleep(5)  # Wait for page to load
    
    # Initialize the Chrome service with local chromedriver
    chromedriver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver")
    service = Service(executable_path=chromedriver_path, log_path=os.devnull)

    time.sleep(5)  # Wait for page to load
    
    # Start the driver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    time.sleep(5)  # Wait for page to load
    
    try:
        # Navigate to the login page
        driver.get(login_url)
        time.sleep(5)  # Wait for page to load
        
        # Enter user ID
        login_id_element = driver.find_element(By.XPATH, '//*[@id="userid"]')
        login_id_element.send_keys(USER_ID)

        time.sleep(1)  # Wait for page to load
        
        # Enter password
        pwd_element = driver.find_element(By.XPATH, '//*[@id="password"]')
        pwd_element.send_keys(PASSWORD)

        time.sleep(1)  # Wait for page to load
        
        # Click on login button
        submit_button = driver.find_element(By.XPATH, '//*[@id="container"]/div/div/div[2]/form/div[4]/button')
        submit_button.click()
        
        # Wait for 2FA page to load
        time.sleep(5)
        
        # Generate TOTP
        totp = pyotp.TOTP(TOTP_KEY)
        auth_key = totp.now()

        #totp = remDr$findElement(using="xpath",'//*[(@id = "userid")]')  
        
        # Enter TOTP
        totp_element = driver.find_element(By.XPATH, '//*[(@id = "userid")]')
        totp_element.send_keys(auth_key)
        
        # Wait for redirection
        time.sleep(5)
        
        # Extract request token from URL
        current_url = driver.current_url
        request_token = current_url.split('request_token=')[1].split('&')[0]

        time.sleep(2)  # Wait for page to load
        
        # Generate access token
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        kite.set_access_token(data["access_token"])

        time.sleep(2)  # Wait for page to load
        
        print("Login successful")
        return kite
        
    except Exception as e:
        print(f"Error during login: {e}")
        return None
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    kite = login_to_kite()
    if kite:
        # Example: get margins
        try:
            margins = kite.margins()
            print(f"Available margins: {margins}")
        except Exception as e:
            print(f"Error fetching margins: {e}")
