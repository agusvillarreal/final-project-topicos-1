# Enhanced Meteorite Landings Dashboard

# 1. Setup and Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Visualization libraries
import holoviews as hv
import hvplot.pandas
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, Fullscreen  # Corrected import
import plotly.express as px
import plotly.graph_objects as go
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

# Define helper functions

def load_meteorite_data(file_path='data/meteorite_landings.csv'):  # Fixed case sensitivity in filename
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
    
    # Create decade column for easy grouping
    df['decade'] = (df['year'] // 10) * 10
    
    return df

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

# Enhanced Map Functions
def create_2d_map(gdf, color_by=None, size_by=None, zoom_start=2):
    """Create an interactive 2D world map of meteorite landings with fullscreen option"""
    # Create a map centered at (0, 0) with default zoom
    m = folium.Map(location=[0, 0], zoom_start=zoom_start, 
                  tiles='CartoDB positron')
    
    # Add fullscreen control
    Fullscreen().add_to(m)  # Corrected usage
    
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

def create_3d_globe(df, color_by='fall', size_by='mass_g'):
    """Create an interactive 3D globe visualization of meteorite landings"""
    # Prepare the data
    plot_df = df.copy()
    
    # Process size column
    if size_by and size_by in df.columns:
        # Log transform for better visualization
        plot_df['size'] = np.log1p(plot_df[size_by])
        # Normalize between 3 and 15
        min_size = plot_df['size'].min()
        max_size = plot_df['size'].max()
        plot_df['marker_size'] = 3 + 12 * (plot_df['size'] - min_size) / (max_size - min_size)
    else:
        plot_df['marker_size'] = 5
    
    # Process color column
    if color_by and color_by in df.columns:
        color_col = color_by
    else:
        color_col = None
    
    # Create the 3D globe
    fig = px.scatter_geo(
        plot_df,
        lat='reclat',
        lon='reclong',
        color=color_col,
        size='marker_size',
        hover_name='name',
        hover_data={
            'recclass': True,
            'mass_g': True,
            'year': True,
            'fall': True,
            'marker_size': False,
            'reclat': False,
            'reclong': False
        },
        projection='orthographic',
        title='Global Meteorite Landings',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        opacity=0.8
    )
    
    # Update layout for better appearance
    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(
            showland=True,
            landcolor='rgb(217, 217, 217)',
            coastlinecolor='rgb(80, 80, 80)',
            countrycolor='rgb(217, 217, 217)',
            showocean=True,
            oceancolor='rgb(204, 229, 255)',
            showlakes=False,
            showcountries=True,
            projection_type='orthographic',
            showcoastlines=True
        ),
        legend=dict(
            title=color_by.capitalize() if color_by else None
        )
    )
    
    # Add rotation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ],
                x=0.1,
                y=0,
                xanchor='right',
                yanchor='bottom'
            )
        ]
    )
    
    return fig

# Time Trend Visualizations
def create_time_trend_plots(df):
    """Create multiple time trend visualizations"""
    # Create a copy to avoid modifying the original
    plot_df = df.copy()
    
    # 1. Meteorites found by year
    yearly_counts = plot_df.groupby('year').size().reset_index(name='count')
    yearly_plot = yearly_counts.hvplot.line(
        x='year', 
        y='count', 
        title='Número de meteoritos por año',
        xlabel='Año',
        ylabel='Cantidad',
        height=400,
        line_width=2,
        color='navy'
    )
    
    # 2. Average mass by decade
    plot_df['decade'] = (plot_df['year'] // 10) * 10
    decade_mass = plot_df.groupby('decade')['mass_g'].mean().reset_index()
    decade_plot = decade_mass.hvplot.bar(
        x='decade', 
        y='mass_g', 
        title='Masa promedio por década',
        xlabel='Década',
        ylabel='Masa promedio (g)',
        height=400,
        color='teal'
    )
    
    # 3. Scatter plot of mass vs year with trend line
    if len(plot_df) > 5000:
        scatter_df = plot_df.sample(5000)
    else:
        scatter_df = plot_df
    
    scatter_plot = scatter_df.hvplot.scatter(
        x='year', 
        y='mass_g', 
        title='Relación entre año y masa',
        xlabel='Año',
        ylabel='Masa (g)',
        height=400,
        color='red',
        alpha=0.5,
        logy=True,  # FIXED: Changed from loggy to logy
    )
    
    # 4. Cumulative count over time
    yearly_counts['cumulative'] = yearly_counts['count'].cumsum()
    cumulative_plot = yearly_counts.hvplot.line(
        x='year', 
        y='cumulative', 
        title='Cantidad acumulada de meteoritos a lo largo del tiempo',
        xlabel='Año',
        ylabel='Cantidad acumulada',
        height=400,
        line_width=2,
        color='purple'
    )
    
    return pn.Column(
        "## Tendencias a lo largo del tiempo",
        pn.Tabs(
            ('Meteoritos por año', yearly_plot),
            ('Masa promedio por década', decade_plot),
            ('Masa vs Año', scatter_plot),
            ('Acumulado histórico', cumulative_plot)
        )
    )

# Distribution Visualizations
def create_distribution_plots(df):
    """Create visualizations of variable distributions"""
    # Create a copy to avoid modifying the original
    plot_df = df.copy()
    
    # 1. Mass histogram
    # Remove outliers for better visualization (keep 95% of data)
    mass_q = plot_df['mass_g'].quantile(0.95)
    mass_df = plot_df[plot_df['mass_g'] < mass_q].copy()
    
    mass_hist = mass_df.hvplot.hist(
        'mass_g',
        bins=50,
        title='Distribución de masas de meteoritos',
        xlabel='Masa (g)',
        ylabel='Frecuencia',
        height=400,
        color='teal'
    )
    
    # 2. Log-transformed mass
    plot_df['log_mass'] = np.log1p(plot_df['mass_g'])
    log_mass_hist = plot_df.hvplot.hist(
        'log_mass',
        bins=50,
        title='Distribución de masas (escala logarítmica)',
        xlabel='Log(Masa+1)',
        ylabel='Frecuencia',
        height=400,
        color='navy'
    )
    
    # 3. Bar chart of meteorite classes (top 20)
    class_counts = plot_df['recclass'].value_counts().nlargest(20).reset_index()
    class_counts.columns = ['class', 'count']
    
    class_bar = class_counts.hvplot.bar(
        x='class',
        y='count',
        title='20 clasificaciones más comunes de meteoritos',
        xlabel='Clasificación',
        ylabel='Cantidad',
        height=500,
        width=800,
        color='orange',
        rot=45
    )
    
    # 4. FIXED: Replaced pie chart with bar chart for "Fell vs Found"
    fell_found = plot_df['fall'].value_counts().reset_index()
    fell_found.columns = ['status', 'count']
    
    fell_found_bar = fell_found.hvplot.bar(
        x='status',
        y='count',
        title='Proporción de meteoritos caídos vs encontrados',
        xlabel='Estado',
        ylabel='Cantidad',
        height=400,
        width=400,
        color=['red', 'blue']
    )
    
    return pn.Column(
        "## Distribuciones de variables",
        pn.Tabs(
            ('Distribución de masas', mass_hist),
            ('Masas (escala logarítmica)', log_mass_hist),
            ('Clasificaciones', class_bar),
            ('Caídos vs Encontrados', fell_found_bar)  # FIXED: Updated reference here
        )
    )

# Heat Map Visualizations
def create_heatmap_visualizations(df):
    """Create heatmap visualizations"""
    # Create a copy to avoid modifying the original
    plot_df = df.copy()
    
    # 1. Heatmap of meteorites by decade and fall status
    decade_fall = pd.crosstab(plot_df['decade'], plot_df['fall'])
    decade_fall_heatmap = decade_fall.hvplot.heatmap(
        title='Meteoritos por década y estado de caída',
        xlabel='Estado de caída',
        ylabel='Década',
        height=500,
        width=600,
        cmap='viridis'
    )
    
    # 2. Heatmap of meteorites by decade and top 10 classes
    top_classes = plot_df['recclass'].value_counts().nlargest(10).index.tolist()
    class_df = plot_df[plot_df['recclass'].isin(top_classes)].copy()
    decade_class = pd.crosstab(class_df['decade'], class_df['recclass'])
    decade_class_heatmap = decade_class.hvplot.heatmap(
        title='Meteoritos por década y clasificación (top 10)',
        xlabel='Clasificación',
        ylabel='Década',
        height=600,
        width=800,
        cmap='inferno',
        rot=90
    )
    
    # 3. Geographic heatmap using Plotly
    geo_df = plot_df[['reclat', 'reclong', 'name']].dropna()
    
    # Let's create a simpler geographic heatmap to avoid Mapbox token issues
    geo_fig = px.density_mapbox(
        geo_df, 
        lat='reclat', 
        lon='reclong', 
        z=np.ones(len(geo_df)),  # Uniform weights
        radius=10,
        center=dict(lat=0, lon=0), 
        zoom=1,
        mapbox_style="carto-positron",
        title="Densidad geográfica de meteoritos"
    )
    
    geo_fig.update_layout(height=600)
    
    return pn.Column(
        "## Mapas de calor",
        pn.Tabs(
            ('Década vs Estado de caída', decade_fall_heatmap),
            ('Década vs Clasificación', decade_class_heatmap),
            ('Densidad geográfica', pn.pane.Plotly(geo_fig))
        )
    )

# Enhanced Metrics and Statistics
def calculate_enhanced_statistics(df):
    """Calculate comprehensive statistics for meteorite data"""
    stats = {}
    
    # Basic count statistics
    stats['total_meteorites'] = len(df)
    stats['total_fell'] = df[df['fall'] == 'Fell'].shape[0]
    stats['total_found'] = df[df['fall'] == 'Found'].shape[0]
    stats['unique_classes'] = df['recclass'].nunique()
    
    # Mass statistics (removing NaN values)
    mass_data = df['mass_g'].dropna()
    stats['mass_mean'] = mass_data.mean()
    stats['mass_median'] = mass_data.median()
    stats['mass_std'] = mass_data.std()
    stats['mass_min'] = mass_data.min()
    stats['mass_max'] = mass_data.max()
    stats['mass_q1'] = mass_data.quantile(0.25)
    stats['mass_q3'] = mass_data.quantile(0.75)
    
    # Year statistics
    year_data = df['year'].dropna()
    stats['year_mean'] = year_data.mean()
    stats['year_median'] = year_data.median()
    stats['year_std'] = year_data.std()
    stats['year_min'] = year_data.min()
    stats['year_max'] = year_data.max()
    
    # Top classes
    top_classes = df['recclass'].value_counts().nlargest(5)
    stats['top_classes'] = {name: count for name, count in zip(top_classes.index, top_classes.values)}
    
    # Decade with most meteorites
    decade_counts = df.groupby('decade').size()
    stats['top_decade'] = decade_counts.idxmax()
    stats['top_decade_count'] = decade_counts.max()
    
    # Correlations
    numeric_df = df[['year', 'mass_g', 'reclat', 'reclong']].dropna()
    if len(numeric_df) > 0:
        stats['correlation_matrix'] = numeric_df.corr().to_dict()
    
    return stats

def create_correlation_plot(df):
    """Create scatter plots with trend lines for correlations"""
    # Create copy and drop NaN values
    plot_df = df[['year', 'mass_g', 'reclat', 'reclong']].dropna()
    
    # If there are too many points, sample for better performance
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000)
    
    # Create scatter plot with trend line for year vs mass
    # Log transform mass for better visualization
    plot_df['log_mass'] = np.log1p(plot_df['mass_g'])
    
    # Calculate trend line
    X = plot_df['year'].values.reshape(-1, 1)
    y = plot_df['log_mass'].values
    
    model = LinearRegression()
    try:
        model.fit(X, y)
        plot_df['trend'] = model.predict(X)
        
        # Get r-squared
        r_squared = model.score(X, y)
        
        # Create scatter plot
        scatter = plot_df.hvplot.scatter(
            x='year', 
            y='log_mass',
            title=f'Correlación entre año y masa (R² = {r_squared:.4f})',
            xlabel='Año',
            ylabel='Log(Masa+1)',
            color='blue',
            alpha=0.5,
            height=500,
            width=800
        )
        
        # Create trend line
        trend = plot_df.hvplot.line(
            x='year',
            y='trend',
            color='red',
            line_width=2
        )
        
        return pn.Column(
            "## Análisis de correlación",
            scatter * trend
        )
    except:
        return pn.Column(
            "## Análisis de correlación",
            "No se pudo calcular la línea de tendencia debido a datos insuficientes."
        )

def create_statistics_dashboard(stats):
    """Create a dashboard to display comprehensive statistics"""
    # Format numeric values
    format_num = lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)
    
    # Basic stats card
    basic_stats = pn.pane.Markdown(f"""
    ## Estadísticas generales
    
    | Métrica | Valor |
    | --- | --- |
    | Total de meteoritos | {stats['total_meteorites']:,} |
    | Meteoritos caídos | {stats['total_fell']:,} |
    | Meteoritos encontrados | {stats['total_found']:,} |
    | Clases únicas | {stats['unique_classes']:,} |
    | Década con más meteoritos | {stats['top_decade']} ({stats['top_decade_count']:,} meteoritos) |
    """)
    
    # Mass stats card
    mass_stats = pn.pane.Markdown(f"""
    ## Estadísticas de masa (gramos)
    
    | Métrica | Valor |
    | --- | --- |
    | Media | {format_num(stats['mass_mean'])} |
    | Mediana | {format_num(stats['mass_median'])} |
    | Desviación estándar | {format_num(stats['mass_std'])} |
    | Mínimo | {format_num(stats['mass_min'])} |
    | Máximo | {format_num(stats['mass_max'])} |
    | Primer cuartil (Q1) | {format_num(stats['mass_q1'])} |
    | Tercer cuartil (Q3) | {format_num(stats['mass_q3'])} |
    """)
    
    # Year stats card
    year_stats = pn.pane.Markdown(f"""
    ## Estadísticas de año
    
    | Métrica | Valor |
    | --- | --- |
    | Media | {format_num(stats['year_mean'])} |
    | Mediana | {format_num(stats['year_median'])} |
    | Desviación estándar | {format_num(stats['year_std'])} |
    | Año más antiguo | {int(stats['year_min'])} |
    | Año más reciente | {int(stats['year_max'])} |
    """)
    
    # Top classes card
    top_classes_md = "## Clases más comunes\n\n| Clase | Cantidad |\n| --- | --- |\n"
    for class_name, count in stats['top_classes'].items():
        top_classes_md += f"| {class_name} | {count:,} |\n"
    
    top_classes = pn.pane.Markdown(top_classes_md)
    
    # Correlation matrix
    if 'correlation_matrix' in stats:
        corr_md = "## Matriz de correlación\n\n| | year | mass_g | reclat | reclong |\n| --- | --- | --- | --- | --- |\n"
        
        for var1 in ['year', 'mass_g', 'reclat', 'reclong']:
            corr_md += f"| **{var1}** |"
            for var2 in ['year', 'mass_g', 'reclat', 'reclong']:
                corr_md += f" {stats['correlation_matrix'][var1][var2]:.4f} |"
            corr_md += "\n"
        
        correlation = pn.pane.Markdown(corr_md)
    else:
        correlation = pn.pane.Markdown("## Matriz de correlación\n\nNo hay suficientes datos para calcular correlaciones.")
    
    return pn.Column(
        "# Métricas y estadísticas detalladas",
        pn.Row(
            basic_stats,
            mass_stats
        ),
        pn.Row(
            year_stats,
            top_classes
        ),
        correlation
    )

# 2. Data Loading and Exploration
# Load the meteorite landings dataset
# If the file doesn't exist in the data directory, download it
data_file = 'data/meteorite_landings.csv'  # Fixed case sensitivity in filename
if not os.path.exists(data_file):
    import urllib.request
    url = "https://data.nasa.gov/api/views/gh4g-9sfh/rows.csv?accessType=DOWNLOAD"
    os.makedirs('data', exist_ok=True)
    urllib.request.urlretrieve(url, data_file)
    print(f"Downloaded meteorite landings dataset to {data_file}")

# Load and clean the data
df = load_meteorite_data(data_file)

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
@pn.depends(filters.param.apply_button)
def update_visualizations(_):
    # Get filtered data
    filtered_df, filtered_gdf = filters.filter_data()
    
    # Update count display
    count_display.object = f"### Number of meteorites: {len(filtered_df)}"
    
    # Create 2D map visualization
    map_2d = create_map_visualization(filtered_gdf)
    
    # Create 3D globe visualization
    globe_3d = create_globe_visualization(filtered_df)
    
    # Create time trend visualizations
    time_trends = create_time_trends(filtered_df)
    
    # Create distribution visualizations
    distributions = create_distribution_visualizations(filtered_df)
    
    return pn.Column(
        count_display,
        pn.Tabs(
            ('2D World Map', map_2d),
            ('3D Globe', globe_3d),
            ('Time Trends', time_trends),
            ('Distributions', distributions)
        )
    )

# Helper function to create 2D map visualization
def create_map_visualization(gdf):
    if len(gdf) > 5000:
        # Subsample for better performance if there are too many points
        gdf = gdf.sample(5000)
    
    # Create map
    m = create_2d_map(gdf, color_by='fall', size_by='mass_g', zoom_start=2)
    
    # Convert to Panel pane
    map_pane = pn.pane.HTML(m._repr_html_(), height=600)
    
    return map_pane

# Helper function to create 3D globe visualization
def create_globe_visualization(df):
    if len(df) > 10000:
        # Subsample for 3D globe visualization
        sample_df = df.sample(10000)
    else:
        sample_df = df
    
    # Create 3D globe
    globe = create_3d_globe(sample_df)
    
    # Convert to Panel pane
    globe_pane = pn.pane.Plotly(globe, height=700)
    
    return globe_pane

# Helper function to create time trend visualizations
def create_time_trends(df):
    return create_time_trend_plots(df)

# Helper function to create distribution visualizations
def create_distribution_visualizations(df):
    # Combine distribution plots and heatmaps
    distributions = create_distribution_plots(df)
    heatmaps = create_heatmap_visualizations(df)
    
    return pn.Column(
        distributions,
        pn.layout.Divider(),
        heatmaps
    )

# Initialize count display
count_display = pn.pane.Markdown(f"### Number of meteorites: {len(filtered_df)}")

# Initial visualization
visualizations = update_visualizations(None)

# 3.3 Layer 3: Metrics and Analysis
# Create a reactive function to update metrics when filters change
@pn.depends(filters.param.apply_button)
def update_metrics(_):
    # Get filtered data
    filtered_df, filtered_gdf = filters.filter_data()
    
    # Calculate enhanced statistics
    stats = calculate_enhanced_statistics(filtered_df)
    
    # Create statistics dashboard
    stats_dashboard = create_statistics_dashboard(stats)
    
    # Create correlation plot
    correlation_plot = create_correlation_plot(filtered_df)
    
    return pn.Column(
        stats_dashboard,
        pn.layout.Divider(),
        correlation_plot
    )

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
# Create a card for filters with CSS styling
filters_card = pn.Column(
    "# Meteorite Landings Dashboard",
    "## Explore meteorite data from NASA's Open Data Portal",
    "This dashboard provides an interactive exploration of meteorite landings worldwide. Use the filters below to customize the view.",
    pn.layout.Divider(),
    filter_widgets,
    width=300,
    css_classes=['card'],
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
        .fullscreen-enabled {
            background: white;
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
pn.serve(dashboard)