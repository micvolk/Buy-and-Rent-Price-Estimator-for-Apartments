# -*- coding: utf-8 -*-
"""
Cleans the dataframes created by scraping.py and saves them as csv-files cleaned_buy.csv, cleaned_rent.csv.
Cleaning process consists of considering only columns which contain useful information
and of dropping rows out of the dataframes which:
* have outliers for price, area, construction year
* have missing values in important columns
* are identical/very similiar to other apartements
* have defined blacklist-keywords in their description text like 'versteigerung', 'erbbau', 'pflegeimmobilie', ...

@author: Michael Volk
"""

import generalFunctions as gf


#----------------------------------------------------------------------------------------------------


# Section 1: Define functions for the datacleanig-process


def row_dropper(df):    
    """Drops rows from given dataframe, with exposes for buy or rent,
    which do not match certain conditions"""
    
    #Drop rows which do not have the expected value for the defined column
    df.drop(df[df.HardFacts_PRICE_Unit != 'EUR'].index, inplace=True)
    df.drop(df[(df.HardFacts_PRICE_Label == 'Kaufpreis') & ((
        df.HardFacts_PRICE_NumberValue < 40000) | (df.HardFacts_PRICE_NumberValue > 2000000))].index, inplace=True)   
    df.drop(df[(df.equipmentArea_CONSTRUCTIONYEAR < 1850)
               | (df.equipmentArea_CONSTRUCTIONYEAR > 2021)].index, inplace=True)
    df.drop(df[df.HardFacts_AREA_LIVING_Unit != 'SQM'].index, inplace=True)
    df.drop(df[df.HardFacts_AREA_LIVING_Label != 'Wohnfläche'].index, inplace=True)
    df.drop(df[df.HardFacts_ROOMS_Label != 'Zimmer'].index, inplace=True)
    df.drop(df[df.General_EstateTypeKey != 'WOHNUNG'].index, inplace=True)
    
    #Drop rows which are not in defined boundaries to avoid outliers
    df.drop(df[df.HardFacts_AREA_LIVING_NumberValue > 180].index, inplace=True)
    df.drop(df[df.HardFacts_ROOMS_NumberValue > 7].index, inplace=True)
    
    #Drop rows where definied columns have missing values
    df.dropna(axis=0, how='any', subset=['HardFacts_PRICE_NumberValue',
                                        'HardFacts_AREA_LIVING_NumberValue',
                                        'HardFacts_ROOMS_NumberValue',
                                        'equipmentArea_CONSTRUCTIONYEAR',
                                        'DescriptionText',
                                        'EstateAddress_LocationId',
                                        'EstateMapData_LocationCoordinates_Latitude',
                                        'EstateMapData_LocationCoordinates_Longitude',
                                        'MediaItemsCount'], inplace=True)
    
    # Group rows by offerer, construction year and location and consider only the first row of each group
    # to drop out apartments which are offered by the same company and
    # are probably contained in one building and similiar or identical to each other
    # Reason: to avoid masses of similiar apartments from one big building(-project)
    df = df.groupby(['Offerer_globalUserId',
                    'equipmentArea_CONSTRUCTIONYEAR',
                    'EstateAddress_LocationId']).apply(lambda df: df.iloc[0])
    df.reset_index(drop=True, inplace=True)
    
    # Group rows by DescriptionText and consider only the first row of each group
    # to drop out apartments which are identical in their offer text
    df = df.groupby('DescriptionText').apply(lambda df: df.iloc[0])
    df.reset_index(drop=True, inplace=True)

    
    return df

def row_dropper_buy(df):    
    """Drops rows from given dataframe, with exposes for buy only,
    which do not match certain conditions"""
    
    #Drop rows which do not have the expected value for the defined column
    df.drop(df[df.HardFacts_PRICE_Label != 'Kaufpreis'].index, inplace=True)
    df.drop(df[df.General_DistributionTypeKey != 'ZUM_KAUF'].index, inplace=True)
    
    # Drop rows where Headline or DescriptionText contains keywords which indicate an auction or
    # only part-ownership or multiple apartments in one offer (Paketverkauf) or forms of special owership (Erbbaurecht)
    # or special apartments for elderly care (Pflegeimmobilien)
    # Hint: All own created columns, which are appended to the dataframe, will be named with suffix '_OWN'
    blacklistKeywords = ['versteigerung', 'auktion', 'meistbietend', 'höchstbietend',
                        'mindestpreis', 'verkauf gegen angebot', 'miteigentumsanteil',
                        'paketverkauf', 'erbbau', 'pflegeimmobilie']
    df['blacklistKeywordsContained_OWN'] = df.apply(
        lambda row: any([blacklistKeyword in row.DescriptionText.lower() 
                         for blacklistKeyword in blacklistKeywords])
                    or
                    any([blacklistKeyword in row.General_Headline.lower() 
                         for blacklistKeyword in blacklistKeywords]), axis=1)
    df.drop(df[df.blacklistKeywordsContained_OWN].index, inplace=True)

    return df

def row_dropper_rent(df):    
    """Drops rows from given dataframe, with exposes for rent only,
    which do not match certain conditions"""
    
    #Drop rows which do not have the expected value for the defined column
    df.drop(df[df.Price_DataTable_PRICE_RENT_COLD_Label != 'Kaltmiete'].index, inplace=True)
    df.drop(df[df.Price_DataTable_PRICE_RENT_COLD_Unit != 'EUR'].index, inplace=True)
    
    df.drop(df[(df.Price_DataTable_PRICE_RENT_COLD_NumberValue < 100)
               | (df.Price_DataTable_PRICE_RENT_COLD_NumberValue > 5000)].index, inplace=True)    
    
    #Drop rows where definied columns have missing values
    df.dropna(axis=0, how='any', subset=['Price_DataTable_PRICE_RENT_COLD_NumberValue'], inplace=True)
                
    return df

def reduce_to_core_columns_cleaning(df):
    """Reduces the dataframe, with exposes for buy or rent,
    to the columns which contain potential useful information"""
    
    #Define core information columns
    coreColumns = ['HardFacts_PRICE_NumberValue',
                   'Price_DataTable_PRICE_COMMONCHARGE_NumberValue', #only relevant for buy
                   'Price_DataTable_PRICE_RENT_WARM_NumberValue', #only relevant for rent
                   'Price_DataTable_PRICE_RENT_COLD_NumberValue', #only relevant for rent
                   'Price_DataTable_PRICE_ADDITIONALCOSTS_NumberValue', #only relevant for rent
                   'Price_DataTable_PRICE_HEATINGCOSTS_NumberValue', #only relevant for rent
                   'HardFacts_AREA_LIVING_NumberValue',
                   'HardFacts_ROOMS_NumberValue', 'EstateAddress_City',
                   'General_Headline', 'DescriptionText',
                   'equipmentArea_CONSTRUCTIONYEAR',
                   'equipmentArea_CATEGORY', 'equipmentArea_CONDITION',
                   'equipmentArea_ENERGY', 'equipmentArea_FLOOR',
                   'equipmentArea_OUTDOOR',                           
                   'EstateAddress_District', 'EstateAddress_FederalState',
                   'EstateAddress_FederalStateId', 'EstateAddress_LocationId',
                   'EstateAddress_PublishStreet', 'EstateAddress_ZipCode',
                   'EstateMapData_LocationCoordinates_Latitude',
                   'EstateMapData_LocationCoordinates_Longitude',
                   'General_EstateId', 'MediaItemsCount',
                   'EnergyPasses_Data_Value',
                   'Price_AdditionalInformation_Commission_CommissionType', #only relevant for buy                    
                   'Price_AdditionalInformation_Commission_DisplayValue_NumberValue', #only relevant for buy
                   'Price_AdditionalInformation_Commission_DisplayValue_StringValue', #only relevant for buy
                   'Offerer_globalUserId',
                   'CreateDate', 'ProtocolDatetimeRequestExposeUrl',
                   'ProtocolExposeUrl']
    
    # Only consider the coreColumns if they exist in the dataframe
    # (because some columns only exist in the buy or rent dataframe)
    coreColumns_existing = [col for col in coreColumns if col in df.columns]

    # Only consider the existing coreColumns in the dataframe (so implicitly drop all other columns)
    df = df[coreColumns_existing].copy()

    return df

#----------------------------------------------------------------------------------------------------


# Section 2: Define the order of running the above functions.


def run():
    """
    Runs the above functions in defined order.
    """
    
    # Regarding exposes to buy:
    
    # Execute pipeline
    df_buy_cleaned = (
          gf.load_data(filename = "scraped_buy")
          .pipe(row_dropper)
          .pipe(row_dropper_buy)
          .pipe(reduce_to_core_columns_cleaning)
          .pipe(gf.save_data, filename = "cleaned_buy")
    )
    
    # Regarding exposes to rent:
    
    # Execute pipeline
    df_rent_cleaned = (
          gf.load_data(filename = "scraped_rent")
          .pipe(row_dropper)
          .pipe(row_dropper_rent)
          .pipe(reduce_to_core_columns_cleaning)
          .pipe(gf.save_data, filename = "cleaned_rent")
    )


# Function run() shall be executed if module is executed directly via console 
if __name__ == "__main__":
    run()