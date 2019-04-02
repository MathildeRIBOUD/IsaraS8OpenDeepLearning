# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:51:25 2019

@author: Cédric Berteletti

Adapted for Python 3
Original work: https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57
adapted from http://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search

Example usage:
_ by command line : python google_image_search_scraper.py --search "cat" --num_images 10 --directory "dataset/"
_ or directly setting the parameters in the main method

"""

from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import os
import argparse
import json


HTTP_TIMEOUT = 10 #seconds


def get_soup(url, header):
    with urlopen(Request(url, headers=header), timeout=HTTP_TIMEOUT) as url:
        soup = BeautifulSoup(url, "html.parser")
    return soup


def search_and_save(text_to_search, number_of_images, first_position, root_path):
    query = text_to_search.split()
    query = "+".join(query)
    url = "https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
    header = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    soup = get_soup(url, header)

    path = root_path + text_to_search.replace(" ", "_")
    if not os.path.exists(path):
        os.makedirs(path)

    ActualImages = [] # contains the link for Large original images, type of image
    for a in soup.find_all("div",{"class":"rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        ActualImages.append((link,Type))
    for i, (img, Type) in enumerate(ActualImages[first_position:first_position+number_of_images]):
        try:
            req = Request(img, headers=header)
            print("Opening image N°", i, ": ", img)
            with urlopen(req, timeout=HTTP_TIMEOUT) as urlimage:
                raw_img = urlimage.read()
                print("Image read")
            if len(Type) == 0:
                f = open(os.path.join(path , "img" + "_" + str(i) + ".jpg"), "wb")
            else:
                f = open(os.path.join(path , "img" + "_" + str(i) + "." + Type), "wb")
            f.write(raw_img)
            f.close()
        except Exception as e:
            print("could not load : ", img)
            print(e)


def main(args):
    if(len(args) > 1):
        # parse the command line parameters if any
        parser = argparse.ArgumentParser(description="Scrap Google images")
        parser.add_argument("-s", "--search", default="gazelle", type=str, help="Search term")
        parser.add_argument("-n", "--num_images", default=10, type=int, help="Nb images to save")
        parser.add_argument("-f", "--first_index", default=0, type=int, help="First image to save")
        parser.add_argument("-d", "--directory", default="data/", type=str, help="Save directory")
        args = parser.parse_args()
        query = [args.search]
        max_images = args.num_images
        first_image_index = args.first_index
        save_directory = args.directory
    else:
        # if no command line parameter, directly use these parameters:
        query = ["gazelle thomson", "gazelle grant", "monkey", "giraffe",
                   "lion", "leopard", "elephant", "rhinoceros", "hyppopotame"]
        max_images = 100
        first_image_index = 0
        save_directory = "dataset/"

    for text in query:
        search_and_save(text, max_images, first_image_index, save_directory)


if __name__ == "__main__":
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass

