# -*- coding: utf-8 -*-
"""
scrapes website https://www.immowelt.de/ for apartment-expose-data and
collects it in separate pandas-dataframes for buy and rent,
which are saved as csv-files (default-names) scraped_buy.csv, scraped_rent.csv.

First the scraper collects all URLs to apartment-exposes for a given immowelt.de search result -
by default all apartment-exposes for buy and rent, which are offered in german federal-state North Rhine-Westphalia
and have a construction year <= 2021.
Next the scraper requests for each collected URL the apartment-expose-HTML-webpage from immowelt.de
with the requests-module and than parses the HTML-Code with the BeautifulSoup-module.
The embedded JSON-part of the HTML-Code contains the relevant data and gets transformed
to a python-dictionary, which is than used to fill a new dictionary with the relevant apartment-data-parts.
The dictionaries made in this way are collected in a list which is transformed to a pandas-dataframe and
saved as a csv-file in the end - each for buy and rent.

@author: Michael Volk
"""

import generalFunctions as gf
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from collections import defaultdict
import csv
import tqdm #used for progress-measurement of loops
import numpy as np #used for definition of NaN
import pandas as pd


#----------------------------------------------------------------------------------------------------


# Section 1: Define functions for web-scraping-process from website immowelt.de


def urlExposeCollector (urlSearchResultPage, maxSearchResults = 999999):
    """
    Collects for given immowelt.de search result page (urlSearchResultPage), and
    all associated SearchResultPages, all urls linking to the corresponding
    apartment exposes. Collection begins with search result page 1 and 
    runs until all search result pages have been searched (default)
    or if given parameter pagmaxSearchResults has been reached.

    Parameters
    ----------
    urlSearchResultPage : String
        url of an immowelt.de search result page
    maxSearchResults : Integer
        maximum number of search result pages to search for expose-urls.

    Returns
    -------
    urlsExposes : List
        Contains urls to the exposes

    """ 
    
    try:

        #Define empty list for collecting the links to the exposes
        urlsExposes = []
        
        #Get the url base for all search result pages, that means that part of
        #the url which all search result pages have in common
        urlBase = urlSearchResultPage.split("&sp=")[-2] + "&sp="
        
        print(gf.dayTime() + ": urlExposeCollector started")
        #Iterate over the search result pages
        for page in tqdm.tqdm(range(1, 999999)):
            
            #Build search result page url
            url = urlBase + str(page)
            
            #Download website data for given url    
            r = requests.get(url)
            
            #Parse requested Website to HTML
            soup = BeautifulSoup(r.text, 'html.parser')
            
            #Define Counter for counting the number of links to exposes for given
            #search result page
            searchResultPageUrlExposeCounter = 0
            
            #Save expose-urls from search result page in list    
            for link in soup.find_all('a'):
                #Collect only links which contain 'expose'
                if 'expose' in link.get('href'):
                    urlsExposes.append(link.get('href'))
                    searchResultPageUrlExposeCounter += 1
                #Escape inner for loop if maxSearchResults has been reached
                if len(urlsExposes) == maxSearchResults:
                    break
                
            #Escape outer for loop if maxSearchResults has been reached or if
            #searchResultPageUrlExposeCounter = 0
            if len(urlsExposes) == maxSearchResults or searchResultPageUrlExposeCounter == 0:
                print(gf.dayTime() + ": urlExposeCollector finished!")
                print("Number of searched result pages:")
                if len(urlsExposes) == maxSearchResults:
                    print(str(page))
                else:
                    print(str(page - 1))
                print("Number of collected url-expose-links:")
                print(len(urlsExposes))
                break
            
        # Define and run function for deleting redundant urls from urlsExposes while preserving order
        # Thanks to Martin Broadhurst for this compact function: http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html
        def unique(sequence):
            seen = set()
            return [x for x in sequence if not (x in seen or seen.add(x))]
        urlsExposes = unique(urlsExposes)
        print("Number of collected non redundant url-expose-links:")
        print(len(urlsExposes))
        
        return urlsExposes

    except Exception as e:
        print(gf.dayTime() + ': Error occured with urlExposeCollector:\n%s' % e)



def collectAndSaveExposeUrls(urlSearchResultPage, filename):
    """
    Runs function urlExposeCollector() and saves result to csv-file

    Parameters
    ----------
    urlSearchResultPage : String
        url of an immowelt.de search result page
    filename : String
        name of the csv-file which will be saved

    Returns
    -------
    collectedExposeUrls : List
        Contains urls to the exposes
    """
    
    #Collect urls to the exposes
    collectedExposeUrls = urlExposeCollector(urlSearchResultPage)
    
    #Write collectedExposeUrls to csv
    with open(filename + '.csv','w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',')
        for exposeUrl in collectedExposeUrls:
            csv_writer.writerow([exposeUrl]) #Hint: List-Encapsulation of exposeUrl is necessary
    print("""List 'collectedExposeUrls' saved to file: """ + filename + ".csv")
    return collectedExposeUrls



def exposeScraper(urlExpose):
    """
    Scrappes for the given urlExpose the relevant JSON formated String (jsonString)
    from the website, which contains apartment data.
    Than converts it to a Default-Dictionary (jsonDefaultDict).
    Than jsonDefaultDict is used fill a new dictionary (exposeDict), which contains
    only the relevant apartment data.

    Parameters
    ----------
    urlExpose : String
        Link to the expose

    Returns
    -------
    exposeDict : Dictionary
        Contains the relevant apartment data
    """    
    
    try:
        #Create general protocol data
        dateAndTime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        errorTypeOccuredDetail = None
        #Download website data     
        r = requests.get(urlExpose)
        #Parse requested Website to HTML
        soup = BeautifulSoup(r.text, 'html.parser') #Hint: requests seems to import the text as a raw string, so escape characters are ignored => luckly that makes the further steps easier
        
        #Raise Exception and create Detailed Error Message if expose is not available
        if 'Expose nicht verfügbar' in soup.title:
            errorTypeOccuredDetail = """Expose is not available"""
            raise Exception()
        #Navigate to relevant tag object which contains the json-string: <script id="serverApp-state" type="application/json">
        script = soup.find_all(id = "serverApp-state", type = "application/json")
        #Raise Exception and create Detailed Error Message if soup didn't find search key
        if len(script) == 0:
            errorTypeOccuredDetail = """Could not find script with id="serverApp-state" and type = "application/json" """
            raise Exception()
        #determine left and right position for cutting jsonString
        left = len("""<script id="serverApp-state" type="application/json">""")
        right = len("""</script>""")
        jsonString = str(script[0])[left:-right]
        #Replace &q; with "
        jsonStringClean = jsonString.replace('&q;', '"')
        #Parse jsonString Clean from JSON format to dictionary format
        jsonDict = json.loads(jsonStringClean)
    
        #Define recursive function for converting from nested dictionary-list-structure to
        #defaultdictionary-list-structure
        #Thanks to abarnert from https://stackoverflow.com/questions/50013768/how-can-i-convert-nested-dictionary-to-defaultdict
        def defaultify(d):
            if isinstance(d, dict):
                return defaultdict(lambda: None, {k: defaultify(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [defaultify(e) for e in d]
            else:
                return d
        #Convert jsonDict structure to jsonDefaultDict structure inclusive nested
        #dictionaries
        jsonDefaultDict = defaultify(jsonDict)     
        #Creation of own dictionary for putting in all relevant data from jsonDefaultDict
        exposeDict = {}
        #Derive expose-key from urlExpose for accessing the jsonDefaultDict
        exposeKey = 'expose/' + urlExpose.split("/")[-1]
        
        #Fill own structured dictionary with relevant data from the jsonDefaultDict
        exposeDict['ProtocolExposeUrl'] = urlExpose
        exposeDict['ProtocolDatetimeRequestExposeUrl'] = dateAndTime
        exposeDict['ProtocolErrorTypeOccured'] = np.NaN
        exposeDict['ProtocolErrorTypeOccuredDetail'] = np.NaN
        exposeDict['CreateDate'] = jsonDefaultDict[exposeKey]['CreateDate']       
        for equipmentArea in jsonDefaultDict[exposeKey]['EquipmentAreas']:
            for equipment in equipmentArea['Equipments']:
                exposeDict['equipmentArea_' + equipment['Key']] = equipment['Value']
        for key in jsonDefaultDict[exposeKey]['EstateAddress']:
            exposeDict['EstateAddress_' + key] = jsonDefaultDict[exposeKey]['EstateAddress'][key]
        exposeDict['EstateMapData_LocationCoordinates_Latitude'] = jsonDefaultDict[exposeKey]['EstateMapData']['LocationCoordinates']['Latitude']
        exposeDict['EstateMapData_LocationCoordinates_Longitude'] = jsonDefaultDict[exposeKey]['EstateMapData']['LocationCoordinates']['Longitude']
        for key in jsonDefaultDict[exposeKey]['General']:
            exposeDict['General_' + key] = jsonDefaultDict[exposeKey]['General'][key]
        exposeDict['GlobalObjectKey'] = jsonDefaultDict[exposeKey]['GlobalObjectKey']
        for hardfact in jsonDefaultDict[exposeKey]['HardFacts']:
            for key in hardfact:
                if key != 'Key':
                    exposeDict['HardFacts_' + hardfact['Key'] + '_' + key] = hardfact[key]
        if jsonDefaultDict[exposeKey]['MediaItems'] is not None:
            exposeDict['MediaItemsCount'] = len(jsonDefaultDict[exposeKey]['MediaItems']) #number of pictures shown in expose
        #contactData contains sometimes Nonetype, thats why to check for
        if jsonDefaultDict[exposeKey]['Offerer']['contactData'] is not None:
            exposeDict['Offerer_contactData_companyName'] = jsonDefaultDict[exposeKey]['Offerer']['contactData']['companyName']
        exposeDict['Offerer_globalUserId'] = jsonDefaultDict[exposeKey]['Offerer']['globalUserId']
        exposeDict["Offerer_sellerType"] = jsonDefaultDict[exposeKey]['Offerer']['sellerType']
        exposeDict['OnlineId'] = jsonDefaultDict[exposeKey]['OnlineId']
        if jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Commission'] is not None:
            exposeDict['Price_AdditionalInformation_Commission_CommissionType'] = jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Commission']['CommissionType']
            exposeDict['Price_AdditionalInformation_Commission_DisplayValue_Label'] = jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Commission']['DisplayValue']['Label']
            exposeDict['Price_AdditionalInformation_Commission_DisplayValue_NumberValue'] = jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Commission']['DisplayValue']['NumberValue']
            exposeDict['Price_AdditionalInformation_Commission_DisplayValue_StringValue'] = jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Commission']['DisplayValue']['StringValue']
        if jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Foreclosure'] is not None:
            exposeDict['Price_AdditionalInformation_Foreclosure_Key'] = jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Foreclosure']['Key']
            exposeDict['Price_AdditionalInformation_Foreclosure_StringValue'] = jsonDefaultDict[exposeKey]['Price']['AdditionalInformation']['Foreclosure']['StringValue']        
        if jsonDefaultDict[exposeKey]['EnergyPasses'] is not None:
            exposeDict['EnergyPasses_Data_EnergyType'] = jsonDefaultDict[exposeKey]['EnergyPasses'][0]['Data'][0]['EnergyType']
            exposeDict['EnergyPasses_Data_Value'] = jsonDefaultDict[exposeKey]['EnergyPasses'][0]['Data'][0]['Value']
            exposeDict['EnergyPasses_Data_HotWaterIncluded'] = jsonDefaultDict[exposeKey]['EnergyPasses'][0]['Data'][0]['HotWaterIncluded']
        for data in jsonDefaultDict[exposeKey]['Price']['DataTable']:
            for key in data:
                if key != 'Key':
                    exposeDict['Price_DataTable_' + data['Key'] + '_' + key] = data[key]
        exposeDict['DescriptionText'] = ""
        for text in jsonDefaultDict[exposeKey]['Texts']:
            if exposeDict['DescriptionText'] == "":
                exposeDict['DescriptionText'] = text['Title'] + '\n\n' + text['Content']
            else:
                exposeDict['DescriptionText'] = exposeDict['DescriptionText'] + '\n\n' + text['Title'] + '\n\n' + text['Content']
         
        return exposeDict

    except Exception as e:
        print(gf.dayTime() + ': Error with exposeScraper occured for url ' + urlExpose)
        return {
            'ProtocolExposeUrl': urlExpose,
            'ProtocolDatetimeRequestExposeUrl': dateAndTime,
            'ProtocolErrorTypeOccured': "Own defined Error" if len(str(e)) == 0 else str(e),
            'ProtocolErrorTypeOccuredDetail': errorTypeOccuredDetail
            }



def scrapeAndSaveExposes(collectedExposeUrls, filename, maxNumberExposes = 999999):
    """
    Scrapes exposes on basis of urls contained in list collectedExposeUrls until
    maxNumberExposes has been reached.
    
    Parameters
    ----------
    collectedExposeUrls : List
        Contains urls to the exposes
    maxNumberExposes : TYPE, optional
        Maximum number of exposes to be scraped. The default is 999999.
    filename : String
        name of the csv-file which will be saved

    Returns
    -------
    exposesDataframe : Pandas-Dataframe
        Contains data of the exposes, one row per expose
    """
    
    print(gf.dayTime() + ": scrapeAndSaveExposes started")
    #Create list with scraped expose data
    exposesData = []
    for url in tqdm.tqdm(collectedExposeUrls):
        exposesData.append(exposeScraper(url))
        if len(exposesData) == maxNumberExposes:
            break
    print(gf.dayTime() + ": Scraping Exposes from list 'collectedExposeUrls' finished!")
    
    #Create pandas Dataframe from exposesData
    exposesDataframe = pd.DataFrame()
    for expose in exposesData:
        exposesDataframe = pd.concat([exposesDataframe, pd.DataFrame.from_dict([expose])])
    print("""Pandas Dataframe 'exposesDataframe' created and filled""")
    
    #Write exposesDataframe to csv-file
    exposesDataframe.to_csv(filename + '.csv', index = False)
    print("""Pandas Dataframe 'exposesDataframe' saved to file: """ + filename + ".csv")
    print("scrapeAndSaveExposes finished!")
    
    return exposesDataframe


#----------------------------------------------------------------------------------------------------


# Section 2: Define the order of running the above functions.

def run(maxNumberExposes=999999):
    """
    Runs the above functions in defined order.
    The optional parameter maxNumberExposes defines the maximum number of exposes to be scraped.
    """
    #Search and scrape from website immowelt.de for apartments in nordrhein-westfalen to buy with construction year <= 2021
    
    #Define url for a immowelt.de search result page
    urlSearchResultPage_Buying = 'https://www.immowelt.de/liste/bl-nordrhein-westfalen/wohnungen/kaufen?cyma=2021&d=true&sd=DESC&sf=TIMESTAMP&sp=1'
    # Collect and save expose urls to list and csv-file. Than scrape exposes from that list and save them in a pandas-dataframe formated csv-file
    print (gf.dayTime() + ': Started scraping-process for buy')
    scrapeAndSaveExposes(collectAndSaveExposeUrls(urlSearchResultPage_Buying, "urls_buy"), "scraped_buy", maxNumberExposes)
    print (gf.dayTime() + ': Finished scraping-process for buy')
    
    #Search and scrape from website immowelt.de for apartments in nordrhein-westfalen to rent with construction year <= 2021
    
    # Example of an url search result page for apartments in nordrhein-westfalen to rent with construction year <= 2021
    urlSearchResultPage_Renting = 'https://www.immowelt.de/liste/bl-nordrhein-westfalen/wohnungen/mieten?cyma=2021&d=true&sd=DESC&sf=TIMESTAMP&sp=1'
    # Collect and save expose urls to list and csv-file. Than scrape exposes from that list and save them in a pandas-dataframe formated csv-file
    print (gf.dayTime() + ': Started scraping-process for rent')
    scrapeAndSaveExposes(collectAndSaveExposeUrls(urlSearchResultPage_Renting, "urls_rent"), "scraped_rent", maxNumberExposes)
    print (gf.dayTime() + ': Finished scraping-process for rent')

    
# Function run() shall be executed if module is executed directly via console 
if __name__ == "__main__":
    run()