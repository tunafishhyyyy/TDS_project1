"""
get_discourse_cookie.py
Automates Google login to Discourse using Playwright and prints the _forum_session cookie for use in API scripts.
User should manually enter their email and password in the browser window.
After login, the cookie is saved to a .env file for use by other scripts.
"""
from playwright.sync_api import sync_playwright

DISCOURSE_LOGIN_URL = "https://discourse.onlinedegree.iitm.ac.in/login"

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(DISCOURSE_LOGIN_URL)
        print("Please log in with Google in the browser window. After login and redirect, press Enter here.")
        input("Press Enter after you are fully logged in and see the Discourse forum page...")
        cookies = page.context.cookies()
        if cookies:
            # Build a single cookie string in browser format
            cookie_str = "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies)
            print("\nYour Discourse browser cookie string:")
            print(cookie_str)
            # Write to .env file for use by other scripts
            with open('.env', 'a') as f:
                f.write(f"DISCOURSE_COOKIE={cookie_str}\n")
            print("[INFO] DISCOURSE_COOKIE value written to .env file in browser format.")
        else:
            print("[ERROR] No cookies found. Login may have failed.")
        browser.close()
