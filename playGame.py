from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from PIL import Image
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import time
import numpy as np
import os

counter =0 

def screenshot():
    global counter
    counter+=1
    for element in toHide:
        browser.execute_script("arguments[0].style.display = 'none';", element)

    browser.save_screenshot('screenshot'+str(counter)+'.png')

    for element in toHide:
        browser.execute_script("arguments[0].style.display = 'block';", element)

    time.sleep(2)
    # Crop and resize the screenshot
    image = Image.open('screenshot'+str(counter)+".png")
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((256, 256))
    #resized_image.save('cropped_screenshot'+str(counter)+".png")
    return resized_image.convert('RGB')


model = load_model('runs/runFour/model-22.h5', compile=False)

model.compile(optimizer="adam",
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])


chrome_options = Options()
chrome_options.add_argument("user-data-dir=F:\Python\GeoNew\selenium")
browser = webdriver.Chrome(chrome_options=chrome_options)
#browser.get('https://geoguessr.com/join')
browser.get('https://www.geoguessr.com/duels/4e9211ec-5b3c-474e-bee7-762457014013')


window_size = 900  # Desired size for width and height
browser.set_window_size(window_size, window_size)


while True:
    wait = WebDriverWait(browser, 10000000)  # Set timeout to 0 seconds 
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'guess-map')))

    time.sleep(1)
    parent = browser.find_element_by_css_selector('.game_main__sYtZN')
    children = parent.find_elements_by_xpath("./*") 

    toHide = []
    for child in children:
        if child.get_attribute('class')!="game-panorama_panorama__rdhFg":
            toHide.append(child)

    x = screenshot()



    image  = np.array(x)/255

    print(image.shape)

    image = np.expand_dims(image , axis=0)

    predictions = model.predict([image])
    predicted_labels = np.argmax(predictions, axis=1)

    names = [d for d in os.listdir("train") if os.path.isdir(os.path.join("train", d))]

    print(names[predicted_labels[0]])

    wait = WebDriverWait(browser, 10000000)  # Set timeout to 0 seconds round-score_roundNumber__tdukt
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'round-score_roundNumber__tdukt')))
