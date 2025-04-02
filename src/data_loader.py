import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def load_meteorite_data(file_path='data/meteorite_landings.csv'):
    """
    Load the meteorite landings dataset from CSV
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Processed meteorite landings data
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Clean the data
        df = clean_meteorite_data(df)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_meteorite_data(df):
    """
    Clean and preprocess the meteorite landings data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw meteorite landings data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned meteorite landings data
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Convert year to numeric, coerce errors to NaN
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Convert mass to numeric, coerce errors to NaN
    if 'mass_(g)' in df.columns:
        df['mass_g'] = pd.to_numeric(df['mass_(g)'], errors='coerce')
        df.drop('mass_(g)', axis=1, inplace=True)
    
    # Drop rows with missing lat/long coordinates
    df = df.dropna(subset=['reclat', 'reclong'])
    
    # Create a proper datetime column if possible
    df['year'] = df['year'].fillna(0).astype(int)
    df['date'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
    
    return df

def filter_meteorites(df, year_range=None, mass_range=None, fell_found=None, recclass=None):
    """
    Filter meteorite data based on criteria
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite landings data
    year_range : tuple
        (min_year, max_year)
    mass_range : tuple
        (min_mass, max_mass)
    fell_found : str
        'Fell' or 'Found'
    recclass : str or list
        Meteorite classification
        
    Returns:
    --------
    pandas.DataFrame
        Filtered meteorite data
    """
    filtered_df = df.copy()
    
    # Filter by year range
    if year_range:
        min_year, max_year = year_range
        filtered_df = filtered_df[(filtered_df['year'] >= min_year) & 
                                  (filtered_df['year'] <= max_year)]
    
    # Filter by mass range
    if mass_range and 'mass_g' in filtered_df.columns:
        min_mass, max_mass = mass_range
        filtered_df = filtered_df[(filtered_df['mass_g'] >= min_mass) & 
                                  (filtered_df['mass_g'] <= max_mass)]
    
    # Filter by fall status
    if fell_found:
        filtered_df = filtered_df[filtered_df['fall'] == fell_found]
    
    # Filter by meteorite classification
    if recclass:
        if isinstance(recclass, list):
            filtered_df = filtered_df[filtered_df['recclass'].isin(recclass)]
        else:
            filtered_df = filtered_df[filtered_df['recclass'] == recclass]
    
    return filtered_df

def convert_to_geodataframe(df):
    """
    Convert a pandas DataFrame to a GeoDataFrame for mapping
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite landings data with lat/long coordinates
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with Point geometry
    """
    # Create geometry column from lat/long
    geometry = [Point(xy) for xy in zip(df['reclong'], df['reclat'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # Set coordinate reference system
    gdf.crs = "EPSG:4326"  # WGS84
    
    return gdf

def get_unique_values(df):
    """
    Get unique values for categorical columns for filter options
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite landings data
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and lists of unique values as values
    """
    unique_values = {}
    
    # Get unique meteorite classes
    if 'recclass' in df.columns:
        unique_values['recclass'] = sorted(df['recclass'].unique().tolist())
    
    # Get unique fall statuses
    if 'fall' in df.columns:
        unique_values['fall'] = sorted(df['fall'].unique().tolist())
    
    # Get year range
    if 'year' in df.columns:
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        unique_values['year_range'] = (min_year, max_year)
    
    # Get mass range
    if 'mass_g' in df.columns:
        min_mass = float(df['mass_g'].min())
        max_mass = float(df['mass_g'].max())
        unique_values['mass_range'] = (min_mass, max_mass)
    
    return unique_values