# -*- coding: utf-8 -*-
"""
Feature engineering based on cleaned dataframe files saved by 'cleaning.py'.
Feature engineering process consists of renaming columns, one-hot-encoding of
categorical columns, create mapping for cityname to central-coordinates of the city.

@author: Michael Volk
"""

import generalFunctions as gf
import geopandas as gpd


#----------------------------------------------------------------------------------------------------


# Section 1: Define functions for the feature-engineering-process

def renameColumns(df):
    """Renames certain columns of df"""
    
    df.rename(columns={'HardFacts_PRICE_NumberValue': 'Price_buy',
                       'Price_DataTable_PRICE_RENT_COLD_NumberValue': 'Price_rent_cold',
                       'Price_DataTable_PRICE_RENT_WARM_NumberValue': 'Price_rent_warm',
                       'HardFacts_AREA_LIVING_NumberValue': 'Area',
                       'HardFacts_ROOMS_NumberValue': 'Rooms',
                       'EstateAddress_District': 'City_district',
                       'EstateMapData_LocationCoordinates_Latitude': 'Latitude',
                       'EstateMapData_LocationCoordinates_Longitude': 'Longitude',
                       'equipmentArea_CONSTRUCTIONYEAR': 'ConstructionYear',
                       'MediaItemsCount': 'Pictures_number',
                       'Offerer_globalUserId': 'Offerer_id',
                       'ProtocolExposeUrl': 'Url'
                       }, inplace=True)
    
    return df
   
def categoricalColumns_mapper(df):
    """Maps values for given dataframe in categorical columns to defined values.
    Creates for every defined value an own column with value 0 or 1 (one-hot-encoding)"""
 
    # equipmentArea_CATEGORY    
    # For missing values set 'unknown' and than create extra one-hot-encoded column 'unknown' for it
    df.equipmentArea_CATEGORY.fillna('unknown', inplace=True)
    df['EQ_CAT_unknown'] = df['equipmentArea_CATEGORY'].apply(
        lambda x: 1 if 'unknown' in x.lower() else 0)
    # Create extra column with 0 or 1 values for every relevant keyword
    df['EQ_CAT_floorApartment'] = df['equipmentArea_CATEGORY'].apply(
        lambda x: 1 if 'etagenwohnung' in x.lower() else 0)
    df['EQ_CAT_apartment'] = df['equipmentArea_CATEGORY'].apply(
        lambda x: 1 if 'apartment' in x.lower() else 0)
    df['EQ_CAT_maisonette'] = df['equipmentArea_CATEGORY'].apply(
        lambda x: 1 if 'maisonette' in x.lower() else 0)
    df['EQ_CAT_penthouse'] = df['equipmentArea_CATEGORY'].apply(
        lambda x: 1 if 'penthouse' in x.lower() else 0)
    df['EQ_CAT_terraceApartment'] = df['equipmentArea_CATEGORY'].apply(
        lambda x: 1 if 'terrassenwohnung' in x.lower() else 0)
    df['EQ_CAT_loft'] = df['equipmentArea_CATEGORY'].apply(
        lambda x: 1 if 'loft' in x.lower() else 0)

    # equipmentArea_CONDITION    
    # For missing values set 'unknown' and than create extra one-hot-encoded column 'unknown' for it
    df.equipmentArea_CONDITION.fillna('unknown', inplace=True)
    df['EQ_CON_unknown'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if 'unknown' in x.lower() else 0)
    # Create extra column with 0 or 1 values for every relevant keyword
    df['EQ_CON_needsRefurbishment'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if ('sanierungsbedürftig' in x.lower() or 'entkernt' in x.lower()) else 0)
    df['EQ_CON_partlyRenovated'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if 'teilsaniert' in x.lower() else 0)
    df['EQ_CON_needsRenovation'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if 'renovierungsbedürftig' in x.lower() else 0)
    df['EQ_CON_upscale'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if 'gehoben' in x.lower() else 0)
    df['EQ_CON_maintained'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if 'gepflegt' in x.lower() else 0)
    df['EQ_CON_refurbished'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if ('saniert' in x.lower() and not 'teilsaniert' in x.lower()) else 0)
    df['EQ_CON_renovated'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if 'renoviert' in x.lower() else 0)
    df['EQ_CON_firstOccupancy'] = df['equipmentArea_CONDITION'].apply(
        lambda x: 1 if 'erstbezug' in x.lower() else 0)
                                
    # equipmentArea_OUTDOOR    
    # For missing values set 'unknown' but create no extra one-hot-encoded column 'unknown' for it
    df.equipmentArea_OUTDOOR.fillna('unknown', inplace=True)
    # Create extra column with 0 or 1 values for every relevant outdoor-keyword
    df['EQ_OUT_balcony'] = df['equipmentArea_OUTDOOR'].apply(
        lambda x: 1 if 'balkon' in x.lower() else 0)
    df['EQ_OUT_garden'] = df['equipmentArea_OUTDOOR'].apply(
        lambda x: 1 if 'garten' in x.lower() else 0)
    df['EQ_OUT_terrace'] = df['equipmentArea_OUTDOOR'].apply(
        lambda x: 1 if 'terrasse' in x.lower() else 0)
    df['EQ_OUT_loggia'] = df['equipmentArea_OUTDOOR'].apply(
        lambda x: 1 if ('loggia' in x.lower() or 'wintergarten' in x.lower()) else 0)    

    return df

def add_Prices_per_Area_buy(df):
    """Adds new column 'Price_buy_perArea'"""
    
    df['Price_buy_perArea'] = df.Price_buy / df.Area
    
    return df

def add_Prices_per_Area_rent(df):
    """Adds new columns 'Price_rent_cold_perArea' and 'Price_rent_warm_perArea'"""
    
    df['Price_rent_warm_perArea'] = df.Price_rent_warm / df.Area
    df['Price_rent_cold_perArea'] = df.Price_rent_cold / df.Area
    
    return df

def convert_to_geopandasDataframe(df):
    """Convertes dataframe to geopandas dataframe and sets the coordinate reference system (CRS)
    to EPSG 4326 (EPSG 4326 corresponds to coordinates in latitude and longitude).
    Adds a new column named 'geometry', which contains the coordinates"""
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
        df.Longitude, df.Latitude))
    df.crs = {'init': 'epsg:4326'}
    
    return df

def load_geopandasMap(filename="shapefiles\dvg2gem_nw.shp"):    
    """Reads in geomap from shp-file and returns it as a geopandas_dataframe.
    Default-shapefile is for german state Nordrhein-Westfalen.
    Source: https://www.opengeodata.nrw.de/produkte/geobasis/vkg/dvg/dvg2/"""
    return gpd.read_file(filename, encoding='ASCII')
        
def convert_epsg(df, epsg=25832):
    """Converts geopandas-dataframe to the epsg used by the geopandasMap
    to be compatible for joint operations"""
    
    df = df.to_crs(epsg=25832)
    
    return df

def replace_citynames(df, df_map):
    """Replaces the scraped_cityname by searching on the basis of coordinates
    for the according district-polygon of the given geopandasMap 'df_map' and
    than using that according district-name."""
    
    def get_cityname(x):
        """Function for finding the district where the coordinates are within.
        In case of coordinates not within any district of df_map,
        the resulting IndexError is handled by returning cityname=UNKNOWN."""
        try:
            return df_map.loc[df_map['geometry'].contains(x), 'GN'].values[0]
        except IndexError:
            return 'UNKNOWN'
            
    # Apply function to dataframe and drop rows with cityname=UNKNOWN
    df['City'] = df['geometry'].apply(get_cityname)
    df.drop(df[df.City == 'UNKNOWN'].index, inplace=True)
    
    return df

def first_n_rows(df, n=10):
    """Returns only the first n rows of the dataframe df.
    Only relevant for testing."""
     
    return df[0:n].copy()

def reduce_to_core_columns_featureEngineering_buy(df):
    """Reduces the dataframe, with exposes for buying,
    to the columns which shall be considered in next step data exploration"""
    
    #Define core information columns
    coreColumns = ['Price_buy',
                   'Price_buy_perArea',
                   'Price_estimate_nearest',
                   'Nearest_price_perArea',
                   'Area',
                   'Rooms',
                   'ConstructionYear',
                   'City',
                   'City_district',
                   'Latitude',
                   'Longitude',
                   'EQ_CAT_unknown',
                   'EQ_CAT_floorApartment',
                   'EQ_CAT_apartment',
                   'EQ_CAT_maisonette',
                   'EQ_CAT_penthouse',
                   'EQ_CAT_terraceApartment',
                   'EQ_CAT_loft',
                   'EQ_CON_unknown',
                   'EQ_CON_firstOccupancy',
                   'EQ_CON_upscale',
                   'EQ_CON_maintained',
                   'EQ_CON_renovated',
                   'EQ_CON_needsRenovation',
                   'EQ_CON_refurbished',
                   'EQ_CON_needsRefurbishment',
                   'EQ_CON_partlyRenovated',
                   'EQ_OUT_balcony',
                   'EQ_OUT_garden',
                   'EQ_OUT_loggia',
                   'EQ_OUT_terrace',
                   'Url']
        
    # Only consider the coreColumns if they exist in the dataframe
    # (because some columns only exist in the buy or rent dataframe)
    coreColumns_existing = [col for col in coreColumns if col in df.columns]

    # Only consider the existing coreColumns in the dataframe (so implicitly drop all other columns)
    df = df[coreColumns_existing].copy()

    return df

def reduce_to_core_columns_featureEngineering_rent(df):
    """Reduces the dataframe, with exposes for renting,
    to the columns which shall be considered in next step data exploration"""
    
    #Define core information columns
    coreColumns = ['Price_rent_cold',
                   'Price_rent_cold_perArea',
                   'Price_rent_warm',
                   'Price_rent_warm_perArea',
                   'Price_estimate_nearest',
                   'Nearest_price_perArea',
                   'Area',
                   'Rooms',
                   'ConstructionYear',
                   'City',
                   'City_district',
                   'Latitude',
                   'Longitude',
                   'EQ_CAT_unknown',
                   'EQ_CAT_floorApartment',
                   'EQ_CAT_apartment',
                   'EQ_CAT_maisonette',
                   'EQ_CAT_penthouse',
                   'EQ_CAT_terraceApartment',
                   'EQ_CAT_loft',
                   'EQ_CON_unknown',
                   'EQ_CON_firstOccupancy',
                   'EQ_CON_upscale',
                   'EQ_CON_maintained',
                   'EQ_CON_renovated',
                   'EQ_CON_needsRenovation',
                   'EQ_CON_refurbished',
                   'EQ_CON_needsRefurbishment',
                   'EQ_CON_partlyRenovated',
                   'EQ_OUT_balcony',
                   'EQ_OUT_garden',
                   'EQ_OUT_loggia',
                   'EQ_OUT_terrace',
                   'Url']
        
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
    df_buy_feature_engineered = (
          gf.load_data(filename = "cleaned_buy")
          # .pipe(first_n_rows, n=100) #only relevant for testing
          .pipe(renameColumns)
          .pipe(convert_to_geopandasDataframe)
          .pipe(convert_epsg)
          .pipe(replace_citynames, load_geopandasMap())
          .pipe(categoricalColumns_mapper)
          .pipe(add_Prices_per_Area_buy)
          .pipe(reduce_to_core_columns_featureEngineering_buy)
          .pipe(gf.save_data, filename = "featureEngineered_buy")
    )
    
    # Regarding exposes to rent:
    
    # Execute pipeline
    df_rent_feature_engineered = (
          gf.load_data(filename = "cleaned_rent")
          # .pipe(first_n_rows, n=100) #only relevant for testing
          .pipe(renameColumns)
          .pipe(convert_to_geopandasDataframe)
          .pipe(convert_epsg)
          .pipe(replace_citynames, load_geopandasMap())
          .pipe(categoricalColumns_mapper)
          .pipe(add_Prices_per_Area_rent)
          .pipe(reduce_to_core_columns_featureEngineering_rent)
          .pipe(gf.save_data, filename = "featureEngineered_rent")
    )
    
    
    # For shapefile regarding german state Nordrhein-WestfalenCreate create mapping
    # for cityname to central-coordinates of the city and save it as file for later using in module app.py
    
    # Read in geomap from shapefile regarding german state North Rhine-Westphalia and return it as a geopandas_dataframe
    df_map = load_geopandasMap().to_crs(epsg=4326)
    # Make central points for the administrative districts out of the polygon
    # Thanks to https://stackoverflow.com/questions/38899190/geopandas-label-polygons
    df_map['central'] = df_map['geometry'].apply(lambda x: x.representative_point().coords[:])
    df_map['central'] = [coords[0] for coords in df_map['central']]
    df_map['Longitude'] = df_map['central'].apply(lambda x: x[0])
    df_map['Latitude'] = df_map['central'].apply(lambda x: x[1])
    df_map.rename(columns={'GN': 'City'}, inplace=True)
    # Save dataframe only with cityname and coordinates to file in current folder and web-application folder
    # HINT: Path of web-application folder is defined for my machine and needs to be adjusted for other machines!
    gf.save_data(df_map[['City', 'Latitude', 'Longitude']].sort_values('City'), filename = 'nrwCityCoordinates')
    gf.save_data(df_map[['City', 'Latitude', 'Longitude']].sort_values('City'), filename = '../Buy-and-Rent-Price-Estimator-for-Apartments_Web-Application/nrwCityCoordinates')
    

# Function run() shall be executed if module is executed directly via console 
if __name__ == "__main__":
    run()