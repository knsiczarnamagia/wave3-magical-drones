import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


# Path to your local file with working API
script_dir = os.path.dirname(os.path.abspath(__file__))
local_file_path = os.path.join(script_dir, "index.html")

# Paths
topo_dir = "topo"
orto_dir = "orto"

os.makedirs(topo_dir, exist_ok=True)
os.makedirs(orto_dir, exist_ok=True)

driver = webdriver.Chrome()
driver.get(local_file_path)

# Initialize wait for 20s max
wait = WebDriverWait(driver, 20)

try:
    # Switch to iframe and hover over "Narzędzia"
    iframe = wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
    driver.switch_to.frame(iframe)

    menu_tools = wait.until(EC.visibility_of_element_located((By.ID, "menu2")))
    driver.execute_script("arguments[0].scrollIntoView(true);", menu_tools)
    ActionChains(driver).move_to_element(menu_tools).perform()

    # Perform search (hard-coded)
    link_search = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Szukaj adresów')]")))
    link_search.click()

    input_field = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@id='wgQueryText']")))
    input_text = "warszawa wojska polskiego 15"
    input_field.send_keys(input_text)

    search_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@gpwidget='wgQuery']")))
    search_button.click()
    print("Search completed successfully.")

    # Allow tiles to load
    time.sleep(5)

    # Save topo map
    topo_img = wait.until(EC.presence_of_element_located((By.XPATH, "//img[contains(@id, 'map')]")))
    topo_img_url = topo_img.get_attribute("src")
    print(f"Topo Image URL: {topo_img_url}")

    topo_img_path = os.path.join(topo_dir, "map1.jpg")
    response = requests.get(topo_img_url)
    if response.status_code == 200:
        with open(topo_img_path, "wb") as file:
            file.write(response.content)
        print(f"Topo map saved: {topo_img_path}")
    else:
        print(f"Failed to download Topo map. HTTP Status Code: {response.status_code}")

    # Switch to "Mapa ORTO" module
    menu_modules = wait.until(EC.presence_of_element_located((By.ID, "menu1")))
    driver.execute_script("arguments[0].scrollIntoView(true);", menu_modules)
    ActionChains(driver).move_to_element(menu_modules).perform()

    orto_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@onclick=\"IMAPLITE.mapMng.changeGpMapFromMenu('gp1');\"]")))
    orto_link.click()
    print("Switched to Mapa ORTO.")

    # Allow tiles to load
    time.sleep(5)

    # Save orto map
    orto_img = wait.until(EC.presence_of_element_located((By.XPATH, "//img[contains(@id, 'map')]")))
    orto_img_url = orto_img.get_attribute("src")
    print(f"Orto Image URL: {orto_img_url}")

    orto_img_path = os.path.join(orto_dir, "map1.jpg")
    response = requests.get(orto_img_url)
    if response.status_code == 200:
        with open(orto_img_path, "wb") as file:
            file.write(response.content)
        print(f"Orto map saved: {orto_img_path}")
    else:
        print(f"Failed to download Orto map. HTTP Status Code: {response.status_code}")


finally:
    driver.quit()
