# Meteorite Landings Dashboard

# 1. Setup and Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Visualization libraries
import holoviews as hv
import hvplot.pandas
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import colorcet as cc

# Dashboard libraries
import panel as pn
import param

# Set rendering backends
hv.extension('bokeh')
pn.extension('plotly', 'tabulator')

# Configure figure size and styles
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.style.use('ggplot')

# Default color maps
COLORMAP = 'viridis'

# Define helper functions (that would normally be imported from src/)

# Data loader functions
def load_meteorite_data(file_path='data/meteorite_landings.csv'):
    """Load the meteorite landings dataset from CSV"""
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
    """Clean and preprocess the meteorite landings data"""
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
    """Filter meteorite data based on criteria"""
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
    """Convert a pandas DataFrame to a GeoDataFrame for mapping"""
    # Create geometry column from lat/long
    geometry = [gpd.points_from_xy([x], [y])[0] for x, y in zip(df['reclong'], df['reclat'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # Set coordinate reference system
    gdf.crs = "EPSG:4326"  # WGS84
    
    return gdf

def get_unique_values(df):
    """Get unique values for categorical columns for filter options"""
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

# Visualization functions
def create_world_map(gdf, color_by=None, size_by=None, zoom_start=2):
    """Create an interactive world map of meteorite landings"""
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
    """Create a bar chart showing the distribution of meteorites by a categorical column"""
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
                     bin_size='10YE', title=None):  # Changed from '10Y' to '10YE'
    """Create a time series plot of meteorite landings"""
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
    """Create a histogram of meteorite masses"""
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

# Metrics functions
def calculate_summary_statistics(df, column, group_by=None):
    """Calculate summary statistics for a numeric column"""
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe")
        return None
    
    # Check if column has numeric data
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column '{column}' is not numeric")
        return None
    
    # If group_by is provided, calculate statistics for each group
    if group_by and group_by in df.columns:
        stats_df = df.groupby(group_by)[column].agg([
            'count',
            'mean',
            'std',
            'min',
            'median',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            'max'
        ])
        stats_df = stats_df.rename(columns={
            '<lambda_0>': 'q1',
            '<lambda_1>': 'q3'
        })
    else:
        # Calculate statistics for the entire dataset
        stats_dict = {
            'count': df[column].count(),
            'mean': df[column].mean(),
            'std': df[column].std(),
            'min': df[column].min(),
            'median': df[column].median(),
            'q1': df[column].quantile(0.25),
            'q3': df[column].quantile(0.75),
            'max': df[column].max()
        }
        stats_df = pd.DataFrame(stats_dict, index=[0])
    
    return stats_df

def analyze_time_trends(df, time_column='year', value_column=None, 
                        agg_func='count', rolling_window=10):
    """Analyze trends over time with rolling averages"""
    # Create a copy to avoid modifying the original
    temp_df = df.copy()
    
    # Ensure time_column is properly formatted
    if time_column == 'year':
        # Group by year
        temp_df['year'] = pd.to_numeric(temp_df['year'], errors='coerce')
        temp_df = temp_df.dropna(subset=['year'])
        temp_df['year'] = temp_df['year'].astype(int)
        
        # Create groups by year
        if value_column:
            if agg_func == 'count':
                grouped = temp_df.groupby('year')[value_column].count()
            else:
                grouped = temp_df.groupby('year')[value_column].agg(agg_func)
        else:
            grouped = temp_df.groupby('year').size()
    else:
        # Try to use datetime column
        temp_df['datetime'] = pd.to_datetime(temp_df[time_column], errors='coerce')
        temp_df = temp_df.dropna(subset=['datetime'])
        
        # Extract year for grouping
        temp_df['year'] = temp_df['datetime'].dt.year
        
        # Create groups by year
        if value_column:
            if agg_func == 'count':
                grouped = temp_df.groupby('year')[value_column].count()
            else:
                grouped = temp_df.groupby('year')[value_column].agg(agg_func)
        else:
            grouped = temp_df.groupby('year').size()
    
    # Convert to DataFrame
    ts_df = grouped.reset_index()
    ts_df.columns = ['year', 'value']
    
    # Calculate rolling average
    ts_df['rolling_avg'] = ts_df['value'].rolling(window=rolling_window, center=True).mean()
    
    # Create an interactive plot
    plot = ts_df.hvplot.line(
        x='year', 
        y=['value', 'rolling_avg'],
        title=f'Time Trend Analysis with {rolling_window}-Year Rolling Average',
        height=400,
        width=700,
        legend='top',
        line_width=[1, 3],
        value_label='Count/Value',
        color=['skyblue', 'navy']
    )
    
    return ts_df, plot

def correlation_analysis(df, columns=None):
    """Analyze correlations between numeric columns"""
    # Get numeric columns if not specified
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Verify all specified columns exist and are numeric
        numeric_cols = []
        for col in columns:
            if col not in df.columns:
                print(f"Column '{col}' not found in dataframe")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column '{col}' is not numeric")
                continue
            numeric_cols.append(col)
    
    if len(numeric_cols) < 2:
        print("At least 2 numeric columns are required for correlation analysis")
        return None, None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    corr_plot = corr_matrix.hvplot.heatmap(
        title='Correlation Matrix',
        cmap='RdBu_r',
        xaxis=True,
        yaxis=True,
        colorbar=True,
        height=400,
        width=400
    )
    
    return corr_matrix, corr_plot

def geographic_distribution_metrics(gdf, region_column=None):
    """Calculate metrics about geographic distribution of meteorites"""
    metrics = {}
    
    # Calculate centroid of all points
    all_points = gdf.geometry
    try:
        # Use union_all() instead of unary_union (deprecated)
        centroid = all_points.union_all().centroid
        metrics['global_centroid'] = (centroid.y, centroid.x)  # lat, lon
    except:
        metrics['global_centroid'] = (0, 0)
    
    # Calculate geographic median (minimize distance to all points)
    coords = np.array([(p.y, p.x) for p in all_points])
    metrics['geographic_median'] = (np.median(coords[:, 0]), np.median(coords[:, 1]))
    
    # Calculate standard distance (geographic std dev)
    lat_std = coords[:, 0].std()
    lon_std = coords[:, 1].std()
    metrics['standard_distance'] = (lat_std, lon_std)
    
    # Calculate hemispheric distribution
    metrics['northern_hemisphere'] = (coords[:, 0] > 0).sum() / len(coords)
    metrics['southern_hemisphere'] = (coords[:, 0] < 0).sum() / len(coords)
    metrics['eastern_hemisphere'] = (coords[:, 1] > 0).sum() / len(coords)
    metrics['western_hemisphere'] = (coords[:, 1] < 0).sum() / len(coords)
    
    # Calculate by region if provided
    if region_column and region_column in gdf.columns:
        region_counts = gdf[region_column].value_counts()
        metrics['region_distribution'] = region_counts.to_dict()
        
        # Calculate centroids by region
        region_centroids = {}
        for region in gdf[region_column].unique():
            region_points = gdf[gdf[region_column] == region].geometry
            if len(region_points) > 0:
                try:
                    region_centroid = region_points.union_all().centroid  # Fixed here too
                    region_centroids[region] = (region_centroid.y, region_centroid.x)
                except:
                    pass
        
        metrics['region_centroids'] = region_centroids
    
    return metrics

def analyze_mass_distribution(df, log_transform=True):
    """Analyze the distribution of meteorite masses"""
    if 'mass_g' not in df.columns:
        print("Mass column not found in dataframe")
        return None, None
    
    # Remove rows with missing or zero mass
    mass_data = df[(df['mass_g'].notna()) & (df['mass_g'] > 0)]['mass_g']
    
    # Apply log transform if requested
    if log_transform:
        mass_data = np.log10(mass_data)
    
    # Calculate descriptive statistics
    metrics = {
        'count': len(mass_data),
        'mean': mass_data.mean(),
        'median': mass_data.median(),
        'std': mass_data.std(),
        'skewness': stats.skew(mass_data.dropna()),
        'kurtosis': stats.kurtosis(mass_data.dropna()),
        'min': mass_data.min(),
        'max': mass_data.max(),
        'range': mass_data.max() - mass_data.min(),
        'q1': mass_data.quantile(0.25),
        'q3': mass_data.quantile(0.75),
        'iqr': mass_data.quantile(0.75) - mass_data.quantile(0.25),
    }
    
    # Create QQ plot to check for normality
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(mass_data.dropna(), dist="norm", plot=ax)
    
    if log_transform:
        ax.set_title('Q-Q Plot of Log-Transformed Meteorite Masses')
        ax.set_xlabel('Theoretical Quantiles (Normal Distribution)')
        ax.set_ylabel('Log10(Mass) Quantiles')
    else:
        ax.set_title('Q-Q Plot of Meteorite Masses')
        ax.set_xlabel('Theoretical Quantiles (Normal Distribution)')
        ax.set_ylabel('Mass Quantiles')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return metrics, fig

# 2. Data Loading and Exploration
# Load the meteorite landings dataset
# If the file doesn't exist in the data directory, download it
data_file = 'data/meteorite_landings.csv'
if not os.path.exists(data_file):
    import urllib.request
    url = "https://data.nasa.gov/api/views/gh4g-9sfh/rows.csv?accessType=DOWNLOAD"
    os.makedirs('data', exist_ok=True)
    urllib.request.urlretrieve(url, data_file)
    print(f"Downloaded meteorite landings dataset to {data_file}")

# Load and clean the data
df = load_meteorite_data(data_file)

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nDataset columns:")
for col in df.columns:
    print(f"- {col}")

# Show the first few rows
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values by column:")
print(df.isna().sum())

# Basic statistics for numeric columns
print("\nBasic statistics for numeric columns:")
print(df.describe())

# Convert to GeoDataFrame for mapping
gdf = convert_to_geodataframe(df)

# Get unique values for categorical columns to use in filters
unique_values = get_unique_values(df)

# 3. Dashboard Components
# 3.1 Layer 1: Interactive Filters
# Define a parameter class for the filters
class MeteoriteFilters(param.Parameterized):
    # Year range slider
    year_range = param.Range(
        default=(1800, 2020), 
        bounds=(int(df['year'].min()), int(df['year'].max()))
    )
    
    # Mass range slider (log scale)
    mass_range = param.Range(
        default=(0, 1e6), 
        bounds=(0, float(df['mass_g'].max()))
    )
    
    # Fall status selector
    fall_status = param.ObjectSelector(
        default='All',
        objects=['All'] + unique_values.get('fall', [])
    )
    
    # Meteorite class selector (multi-select)
    meteorite_class = param.ListSelector(
        default=[],
        objects=unique_values.get('recclass', [])[:30]  # Limit to top 30 classes for better usability
    )
    
    # Apply filters button
    apply_button = param.Action(lambda x: x.param.trigger('apply_button'), label='Apply Filters')
    
    # Reset filters button
    reset_button = param.Action(lambda x: x.param.trigger('reset_button'), label='Reset Filters')
    
    def filter_data(self):
        """Apply filters to the dataset based on current parameter values"""
        filtered_df = df.copy()
        
        # Apply year filter
        min_year, max_year = self.year_range
        filtered_df = filtered_df[(filtered_df['year'] >= min_year) & 
                                  (filtered_df['year'] <= max_year)]
        
        # Apply mass filter
        min_mass, max_mass = self.mass_range
        filtered_df = filtered_df[(filtered_df['mass_g'] >= min_mass) & 
                                  (filtered_df['mass_g'] <= max_mass)]
        
        # Apply fall status filter
        if self.fall_status != 'All':
            filtered_df = filtered_df[filtered_df['fall'] == self.fall_status]
        
        # Apply meteorite class filter
        if self.meteorite_class:
            filtered_df = filtered_df[filtered_df['recclass'].isin(self.meteorite_class)]
        
        # Convert to GeoDataFrame for mapping
        filtered_gdf = convert_to_geodataframe(filtered_df)
        
        return filtered_df, filtered_gdf

# Create an instance of the filters
filters = MeteoriteFilters()

# Create filter widgets
filter_widgets = pn.Param(
    filters,
    widgets={
        'year_range': pn.widgets.RangeSlider,
        'mass_range': pn.widgets.RangeSlider,
        'fall_status': pn.widgets.Select,
        'meteorite_class': pn.widgets.MultiSelect,
        'apply_button': pn.widgets.Button(button_type='primary'),
        'reset_button': pn.widgets.Button(button_type='danger'),
    },
    name='Filters',
    show_name=False
)

# Initial filtered data
filtered_df, filtered_gdf = filters.filter_data()

# 3.2 Layer 2: Data Visualizations
# Create a reactive function to update visualizations when filters change
@pn.depends(filters.param.apply_button)  # Fixed: using direct parameter reference
def update_visualizations(_):
    # Get filtered data
    filtered_df, filtered_gdf = filters.filter_data()
    
    # Update count display
    count_display.object = f"### Number of meteorites: {len(filtered_df)}"
    
    # Create map visualization
    map_view = create_map_visualization(filtered_gdf)
    
    # Create year distribution plot
    year_plot = create_year_distribution(filtered_df)
    
    # Create mass distribution plot
    mass_plot = create_mass_distribution(filtered_df)
    
    # Create type distribution plot
    type_plot = create_type_distribution(filtered_df)
    
    return pn.Column(
        count_display,
        pn.Tabs(
            ('World Map', map_view),
            ('Year Distribution', year_plot),
            ('Mass Distribution', mass_plot),
            ('Type Distribution', type_plot)
        )
    )

# Helper function to create map visualization
def create_map_visualization(gdf):
    if len(gdf) > 5000:
        # Subsample for better performance if there are too many points
        gdf = gdf.sample(5000)
    
    # Create map
    m = create_world_map(gdf, color_by='fall', size_by='mass_g', zoom_start=2)
    
    # Convert to Panel pane
    map_pane = pn.pane.HTML(m._repr_html_(), height=500)
    
    return map_pane

# Helper function to create year distribution visualization
def create_year_distribution(df):
    # Create time series plot
    time_plot = plot_time_series(df, time_column='year', bin_size='10YE',  # Changed to '10YE'
                                title='Meteorite Discoveries Over Time')
    
    return pn.Column(
        "### Meteorite Discoveries/Falls Over Time",
        time_plot,
        sizing_mode='stretch_width'
    )

# Helper function to create mass distribution visualization
def create_mass_distribution(df):
    # Create mass histogram
    mass_hist = plot_mass_distribution(df, log_scale=True, bins=50,
                                      title='Distribution of Meteorite Masses (Log Scale)')
    
    return pn.Column(
        "### Meteorite Mass Distribution",
        mass_hist,
        sizing_mode='stretch_width'
    )

# Helper function to create type distribution visualization
def create_type_distribution(df):
    # Create bar chart of top meteorite classes
    fig = plot_meteorite_distribution(df, 'recclass', top_n=15,
                                     title='Top 15 Meteorite Classifications')
    
    return pn.Column(
        "### Meteorite Types",
        pn.pane.Matplotlib(fig, tight=True),
        sizing_mode='stretch_width'
    )

# Initialize count display
count_display = pn.pane.Markdown(f"### Number of meteorites: {len(filtered_df)}")

# Initial visualization
visualizations = update_visualizations(None)

# 3.3 Layer 3: Metrics and Analysis
# Create a reactive function to update metrics when filters change
@pn.depends(filters.param.apply_button)  # Fixed: using direct parameter reference
def update_metrics(_):
    # Get filtered data
    filtered_df, filtered_gdf = filters.filter_data()
    
    # Calculate basic statistics
    if len(filtered_df) > 0:
        # Basic statistics for mass
        mass_stats = calculate_summary_statistics(filtered_df, 'mass_g')
        
        # Time trend analysis
        time_trend_df, time_trend_plot = analyze_time_trends(
            filtered_df, time_column='year', rolling_window=10
        )
        
        # Mass distribution analysis
        mass_metrics, qq_plot = analyze_mass_distribution(filtered_df, log_transform=True)
        
        # Geographic distribution metrics
        geo_metrics = geographic_distribution_metrics(filtered_gdf)
        
        # Correlation analysis
        corr_matrix, corr_plot = correlation_analysis(
            filtered_df, columns=['year', 'mass_g', 'reclat', 'reclong']
        )
        
        # Create metrics components
        stats_component = create_stats_component(mass_stats, mass_metrics, geo_metrics)
        trends_component = create_trends_component(time_trend_plot)
        correlation_component = create_correlation_component(corr_matrix, corr_plot)
        
        return pn.Tabs(
            ('Summary Statistics', stats_component),
            ('Time Trends', trends_component),
            ('Correlations', correlation_component)
        )
    else:
        return pn.pane.Markdown("No data available for analysis with current filters.")

# Helper function to create statistics component
def create_stats_component(mass_stats, mass_metrics, geo_metrics):
    # Format mass statistics
    if mass_stats is not None:
        mass_stats_md = "### Basic Statistics (Mass in grams)\n\n"
        mass_stats_md += "| Statistic | Value |\n|-----------|-------|\n"
        for stat, value in mass_stats.iloc[0].items():
            mass_stats_md += f"| {stat.capitalize()} | {value:.2f} |\n"
    else:
        mass_stats_md = "No mass statistics available."
    
    # Format mass distribution metrics
    if mass_metrics:
        mass_distr_md = "### Mass Distribution Metrics (Log10 Scale)\n\n"
        mass_distr_md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in mass_metrics.items():
            if metric in ['count']:
                mass_distr_md += f"| {metric.capitalize()} | {int(value)} |\n"
            else:
                mass_distr_md += f"| {metric.capitalize()} | {value:.4f} |\n"
    else:
        mass_distr_md = "No mass distribution metrics available."
    
    # Format geographic metrics
    if geo_metrics:
        geo_md = "### Geographic Distribution Metrics\n\n"
        geo_md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in geo_metrics.items():
            if metric in ['global_centroid', 'geographic_median', 'standard_distance']:
                if isinstance(value, tuple):
                    geo_md += f"| {metric.replace('_', ' ').capitalize()} | Lat: {value[0]:.4f}, Lon: {value[1]:.4f} |\n"
            elif metric in ['northern_hemisphere', 'southern_hemisphere', 'eastern_hemisphere', 'western_hemisphere']:
                geo_md += f"| {metric.replace('_', ' ').capitalize()} | {value*100:.2f}% |\n"
            elif metric not in ['region_distribution', 'region_centroids']:
                geo_md += f"| {metric.replace('_', ' ').capitalize()} | {value} |\n"
    else:
        geo_md = "No geographic metrics available."
    
    # Combine into a panel layout
    return pn.Column(
        pn.Row(
            pn.pane.Markdown(mass_stats_md),
            pn.pane.Markdown(mass_distr_md),
            sizing_mode='stretch_width'
        ),
        pn.Row(
            pn.pane.Markdown(geo_md),
            sizing_mode='stretch_width'
        ),
        sizing_mode='stretch_width'
    )

# Helper function to create trends component
def create_trends_component(time_trend_plot):
    return pn.Column(
        "### Time Trend Analysis",
        time_trend_plot,
        sizing_mode='stretch_width'
    )

# Helper function to create correlation component
def create_correlation_component(corr_matrix, corr_plot):
    if corr_matrix is not None:
        # Format correlation matrix as a markdown table
        corr_md = "### Correlation Matrix\n\n"
        corr_md += "| | " + " | ".join(corr_matrix.columns) + " |\n"
        corr_md += "|" + "|".join(["-" for _ in range(len(corr_matrix.columns) + 1)]) + "|\n"
        
        for idx, row in corr_matrix.iterrows():
            corr_md += f"| {idx} |"
            for col in corr_matrix.columns:
                corr_md += f" {row[col]:.4f} |"
            corr_md += "\n"
        
        return pn.Column(
            pn.Row(
                pn.pane.Markdown(corr_md),
                pn.pane.HoloViews(corr_plot),
                sizing_mode='stretch_width'
            ),
            pn.pane.Markdown("""
            ### Interpretation
            
            - **Positive correlations** suggest that as one variable increases, the other tends to increase as well.
            - **Negative correlations** suggest that as one variable increases, the other tends to decrease.
            - **Values close to 0** indicate little to no linear relationship between variables.
            """),
            sizing_mode='stretch_width'
        )
    else:
        return pn.pane.Markdown("No correlation analysis available.")

# Reset filters function
def reset_filters(event):
    filters.year_range = (1800, 2020)
    filters.mass_range = (0, 1e6)
    filters.fall_status = 'All'
    filters.meteorite_class = []

# Connect reset button
filters.param.watch(reset_filters, 'reset_button')

# Initial metrics
metrics = update_metrics(None)

# 4. Dashboard Assembly
# Create the dashboard layout
# Create a card for filters with CSS styling
filters_card = pn.Column(
    "# Meteorite Landings Dashboard",
    "## Explore meteorite data from NASA's Open Data Portal",
    "This dashboard provides an interactive exploration of meteorite landings worldwide. Use the filters below to customize the view.",
    pn.layout.Divider(),
    filter_widgets,
    width=300,
    css_classes=['card'],  # Use CSS classes instead of background
    margin=(10, 5),
)

# Create the main layout
dashboard = pn.Column(
    # Add CSS style for cards
    pn.pane.HTML("""
    <style>
        .card {
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, width=0, height=0, margin=0, sizing_mode='fixed'),
    
    pn.Row(
        filters_card,
        pn.Column(
            pn.Tabs(
                ('Visualizations', visualizations),
                ('Metrics & Analysis', metrics)
            ),
            sizing_mode='stretch_both'
        ),
        sizing_mode='stretch_both'
    ),
    pn.layout.Divider(),
    pn.Row(
        "### Data Source: NASA Open Data Portal - Meteorite Landings",
        align='center',
        sizing_mode='stretch_width'
    ),
    sizing_mode='stretch_both'
)

# Display the dashboard
dashboard.servable()

# Uncomment to save the dashboard to a standalone HTML file
# dashboard.save('meteorite_dashboard.html')

# Launch the dashboard in a new browser tab
pn.serve(dashboard)