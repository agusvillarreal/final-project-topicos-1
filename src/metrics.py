import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import hvplot.pandas

def calculate_summary_statistics(df, column, group_by=None):
    """
    Calculate summary statistics for a numeric column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite data
    column : str
        Numeric column to analyze
    group_by : str, optional
        Column to group by
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with summary statistics
    """
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
    """
    Analyze trends over time with rolling averages
    
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
    rolling_window : int
        Window size for rolling average
        
    Returns:
    --------
    pandas.DataFrame, hvplot object
        DataFrame with time series data and interactive plot
    """
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
    """
    Analyze correlations between numeric columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite data
    columns : list, optional
        List of numeric columns to analyze, if None, uses all numeric columns
        
    Returns:
    --------
    pandas.DataFrame, hvplot object
        Correlation matrix and heatmap visualization
    """
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
    """
    Calculate metrics about geographic distribution of meteorites
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with meteorite data and Point geometry
    region_column : str, optional
        Column with region information (continent, country, etc.)
        
    Returns:
    --------
    dict
        Dictionary with geographic distribution metrics
    """
    metrics = {}
    
    # Calculate centroid of all points
    all_points = gdf.geometry
    centroid = all_points.unary_union.centroid
    metrics['global_centroid'] = (centroid.y, centroid.x)  # lat, lon
    
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
                region_centroid = region_points.unary_union.centroid
                region_centroids[region] = (region_centroid.y, region_centroid.x)
        
        metrics['region_centroids'] = region_centroids
    
    return metrics

def analyze_mass_distribution(df, log_transform=True):
    """
    Analyze the distribution of meteorite masses
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Meteorite data
    log_transform : bool
        Whether to analyze log-transformed masses
        
    Returns:
    --------
    dict, matplotlib.figure.Figure
        Dictionary with metrics and QQ plot
    """
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