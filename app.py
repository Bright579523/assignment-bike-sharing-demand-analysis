import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Load data
    df = pd.read_csv('train.csv', parse_dates=['datetime'])
    
    # Feature Engineering (From previous assignments)
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    
    # Create 'day_period' (From Q11)
    bins = [0, 6, 12, 18, 24]
    labels = ['night', 'morning', 'afternoon', 'evening']
    df['day_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)
    
    # Map Season & Weather for better readability
    df['season_label'] = df['season'].map({1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'})
    df['weather_label'] = df['weather'].map({
        1: 'Clear/Cloudy', 
        2: 'Mist/Cloudy', 
        3: 'Light Rain/Snow', 
        4: 'Heavy Rain/Snow'
    })
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'train.csv' not found. Please make sure the dataset is in the same directory.")
    st.stop()

# --- 2. SIDEBAR (WIDGETS) ---
st.sidebar.header("Filter Options")

# Widget 1: Select Year (Multiselect)
selected_years = st.sidebar.multiselect(
    "Select Year(s):",
    options=sorted(df['year'].unique()),
    default=sorted(df['year'].unique())
)

# Widget 2: Select Season (Multiselect)
selected_seasons = st.sidebar.multiselect(
    "Select Season(s):",
    options=df['season_label'].unique(),
    default=df['season_label'].unique()
)

# Widget 3: Working Day Filter (Radio)
day_type_filter = st.sidebar.radio(
    "Filter Day Type:",
    options=["All", "Working Day", "Non-Working Day"]
)

# Apply Filters
filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['season_label'].isin(selected_seasons))
]

if day_type_filter == "Working Day":
    filtered_df = filtered_df[filtered_df['workingday'] == 1]
elif day_type_filter == "Non-Working Day":
    filtered_df = filtered_df[filtered_df['workingday'] == 0]

# --- 3. MAIN DASHBOARD CONTENT ---
st.title("🚴‍♂️ Bike Sharing Demand Analysis Dashboard")
st.markdown("Assignment III: Interactive summary of findings from the Bike Sharing dataset.")

# Top Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", f"{filtered_df['count'].sum():,}")
col2.metric("Average Rentals/Hour", f"{filtered_df['count'].mean():.2f}")
col3.metric("Max Rentals/Hour", f"{filtered_df['count'].max():,}")

st.markdown("---")

# ROW 1: Time Analysis
st.subheader("1. Temporal Patterns (Time & Season)")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Hourly Trend by Day Type**")
    # Plot 1: Hourly Trend
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=filtered_df, x='hour', y='count', hue='workingday', palette='coolwarm', ax=ax1, marker='o')
    ax1.set_title("Average Rentals by Hour (0=Non-Working, 1=Working)")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with c2:
    st.markdown("**Monthly Trend**")
    # Plot 2: Monthly Trend
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    monthly_avg = filtered_df.groupby('month')['count'].mean().reset_index()
    sns.barplot(data=monthly_avg, x='month', y='count', color='skyblue', ax=ax2)
    ax2.set_title("Average Rentals by Month")
    st.pyplot(fig2)

st.markdown("---")

# ROW 2: Weather & Day Period
st.subheader("2. Weather & Period Analysis")
c3, c4 = st.columns(2)

with c3:
    st.markdown("**Impact of Weather Condition**")
    # Plot 3: Weather Boxplot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x='weather_label', y='count', palette='Set2', ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    ax3.set_title("Rentals Distribution by Weather")
    st.pyplot(fig3)

with c4:
    st.markdown("**Rentals by Period of Day**")
    # Plot 4: Day Period Barplot
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    # Widget 4 (Inline): Color Choice
    color_by = st.selectbox("Color bars by:", ["season_label", "workingday", None])
    
    sns.barplot(data=filtered_df, x='day_period', y='count', hue=color_by, palette='viridis', ax=ax4)
    ax4.set_title("Average Rentals by Day Period (Night/Morning/Afternoon/Evening)")
    st.pyplot(fig4)

st.markdown("---")

# ROW 3: Correlation & Raw Data
st.subheader("3. Correlation Analysis")

# Widget 5: Toggle Heatmap
if st.checkbox("Show Correlation Matrix Heatmap", value=True):
    # Plot 5: Heatmap
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    numeric_cols = filtered_df.select_dtypes(include=['number']).columns
    corr = filtered_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)

# Raw Data Expander
with st.expander("See Raw Data"):
    st.dataframe(filtered_df.head(100))