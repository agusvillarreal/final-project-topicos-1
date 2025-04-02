import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
import hvplot.pandas
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import colorcet as cc

def create_world_map(gdf, color_by=None, size_by=None, zoom_start=2):
    """
    Create an interactive world map of meteorite landings
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with meteorite data and Point geometry
    color_by : str, optional
        Column name to determine marker color
    size_by : str, optional
        Column name to determine marker size
    zoom_start : int
        Initial zoom level for the map
        
    Returns:
    --------
    folium.Map
        Interactive folium map
    """
    # Create a map centered at (0, 0) with default zoom
    m = folium.Map(location=[0, 0], zoom_start=zoom_start, 
                   tiles='CartoDB positron')
    
    # Add a MarkerCluster to handle many points
    marker_cluster = MarkerCluster().add_to(m)
    
    # Create a color map if color_by is specified
    if color_by and color_by in gdf.columns:
        unique_values = gdf[color_by].unique()
        colors = {value: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                  for i, value in enumerate(unique_values)}
    
    # Add points to the map
    for idx, row in gdf.iterrows():
        # Get lat, lon from geometry
        lon, lat = row.geometry.x, row.geometry.y
        
        # Determine marker size
        if size_by and size_by in gdf.columns and pd.notna(row[size_by]):
            try:
                # Normalize size between 5 and 15
                min_val = gdf[size_by].min()
                max_val = gdf[size_by].max()
                norm_size = 5 + (row[size_by] - min_val) / (max_val - min_val) * 10
            except:
                norm_size = 5
        else:
            norm_size = 5
        
        # Determine marker color
        if color_by and color_by in gdf.columns:
            color = colors.get(row[color_by], 'blue')
        else:
            color = 'blue'
        
        # Create popup text
        popup_text = f"<b>{row['name']}</b><br>"
        popup_text += f"Class: {row['recclass']}<br>"
        popup_text += f"Mass: {row.get('mass_g', 'Unknown')} g<br>"
        popup_text += f"Year: {row['year']}<br>"
        popup_text += f"Fall/Found: {row['fall']}"
        
        # Add marker to cluster
        folium.CircleMarker(
            location=[lat, lon],
            radius=norm_size,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(marker_cluster)
    
    return m

def plot_meteorite_distribution(df, column, top_n=10, title=None):
    """
    Create a bar chart showing the distribution of meteorites by a categorical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite data
    column : str
        Column name to plot
    top_n : int
        Number of top categories to show
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Bar chart figure
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe")
        return None
    
    # Count values in the column
    value_counts = df[column].value_counts().nlargest(top_n)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    value_counts.plot(kind='bar', ax=ax, color='skyblue')
    
    # Set labels and title
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel('Count')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Top {top_n} {column.capitalize()} Categories')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_time_series(df, time_column='year', value_column=None, agg_func='count', 
                     bin_size='10Y', title=None):
    """
    Create a time series plot of meteorite landings
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite data
    time_column : str
        Column with time values
    value_column : str, optional
        Column to aggregate, if None, counts records
    agg_func : str
        Aggregation function ('count', 'sum', 'mean', etc.)
    bin_size : str
        Time period for binning (e.g., '1Y', '10Y', '1M')
    title : str, optional
        Plot title
        
    Returns:
    --------
    hvplot object
        Interactive time series plot
    """
    # Convert time column to datetime if it's not already
    if time_column == 'year':
        # Create a datetime series from the year
        time_series = pd.to_datetime(df['year'], format='%Y', errors='coerce')
    else:
        time_series = pd.to_datetime(df[time_column], errors='coerce')
    
    # Create a copy of the dataframe with the proper datetime column
    temp_df = df.copy()
    temp_df['datetime'] = time_series
    
    # Drop rows with invalid dates
    temp_df = temp_df.dropna(subset=['datetime'])
    
    # Group by time periods and aggregate
    if value_column:
        if agg_func == 'count':
            grouped = temp_df.groupby(pd.Grouper(key='datetime', freq=bin_size))[value_column].count()
        elif agg_func == 'sum':
            grouped = temp_df.groupby(pd.Grouper(key='datetime', freq=bin_size))[value_column].sum()
        elif agg_func == 'mean':
            grouped = temp_df.groupby(pd.Grouper(key='datetime', freq=bin_size))[value_column].mean()
        else:
            grouped = temp_df.groupby(pd.Grouper(key='datetime', freq=bin_size))[value_column].agg(agg_func)
    else:
        # Count records by time period
        grouped = temp_df.groupby(pd.Grouper(key='datetime', freq=bin_size)).size()
    
    # Convert to dataframe for plotting
    plot_df = grouped.reset_index()
    plot_df.columns = ['datetime', 'value']
    
    # Create interactive plot with hvplot
    plot = plot_df.hvplot.line(
        x='datetime', 
        y='value', 
        title=title if title else f'{agg_func.capitalize()} of meteorites over time',
        height=400,
        width=700,
        line_width=2,
        color='navy'
    )
    
    return plot

def plot_mass_distribution(df, log_scale=True, bins=50, title=None):
    """
    Create a histogram of meteorite masses
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite data
    log_scale : bool
        Whether to use log scale for mass
    bins : int
        Number of bins for histogram
    title : str, optional
        Plot title
        
    Returns:
    --------
    hvplot object
        Interactive histogram
    """
    # Check if mass column exists
    if 'mass_g' not in df.columns:
        print("Mass column not found in dataframe")
        return None
    
    # Remove rows with missing or zero mass
    plot_df = df[(df['mass_g'].notna()) & (df['mass_g'] > 0)].copy()
    
    # Apply log transformation if requested
    if log_scale:
        plot_df['mass'] = np.log10(plot_df['mass_g'])
        x_label = 'Log10(Mass in grams)'
    else:
        plot_df['mass'] = plot_df['mass_g']
        x_label = 'Mass (grams)'
    
    # Create histogram with hvplot
    plot = plot_df.hvplot.hist(
        'mass',
        bins=bins,
        title=title if title else 'Distribution of Meteorite Masses',
        xlabel=x_label,
        ylabel='Count',
        height=400,
        width=700,
        color='teal'
    )
    
    return plot

def create_heatmap(df, x_column, y_column, agg_function='count', z_column=None, 
                   cmap='viridis', title=None):
    """
    Create a heatmap showing relationship between two variables
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite data
    x_column : str
        Column for x-axis
    y_column : str
        Column for y-axis
    agg_function : str
        Aggregation function ('count', 'mean', 'sum', etc.)
    z_column : str, optional
        Column to aggregate (required for 'mean', 'sum', etc.)
    cmap : str
        Colormap name
    title : str, optional
        Plot title
        
    Returns:
    --------
    hvplot object
        Interactive heatmap
    """
    # Verify columns exist
    if x_column not in df.columns or y_column not in df.columns:
        print(f"Columns {x_column} or {y_column} not found in dataframe")
        return None
    
    if agg_function != 'count' and (z_column is None or z_column not in df.columns):
        print(f"z_column required for aggregation function '{agg_function}'")
        return None
    
    # Create pivot table
    if agg_function == 'count':
        pivot_data = pd.crosstab(df[y_column], df[x_column])
    else:
        pivot_data = pd.pivot_table(
            df, 
            values=z_column,
            index=y_column,
            columns=x_column,
            aggfunc=agg_function
        )
    
    # Create heatmap with hvplot
    plot = pivot_data.hvplot.heatmap(
        title=title if title else f'Heatmap of {y_column} vs {x_column}',
        cmap=cmap,
        height=500,
        width=700,
        colorbar=True,
    )
    
    return plot