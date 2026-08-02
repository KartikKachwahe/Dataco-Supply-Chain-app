"""
visualization.py
Reusable visualization functions for supply chain analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def plot_delivery_status_distribution(df: pd.DataFrame, save_path: str = None):
    """Bar chart of delivery status counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df['delivery_status'].value_counts()
    colors = sns.color_palette("Blues_r", len(counts))
    counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('Order Distribution by Delivery Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Delivery Status')
    ax.set_ylabel('Number of Orders')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_late_delivery_by_shipping_mode(df: pd.DataFrame, save_path: str = None):
    """Late delivery rate by shipping mode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    late_rate = df.groupby('shipping_mode')['late_delivery_risk'].mean().sort_values(ascending=False) * 100
    late_rate.plot(kind='barh', ax=ax, color=sns.color_palette("Reds_r", len(late_rate)), edgecolor='black')
    ax.set_title('Late Delivery Risk by Shipping Mode', fontsize=14, fontweight='bold')
    ax.set_xlabel('Late Delivery Rate (%)')
    for i, v in enumerate(late_rate.values):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_monthly_order_trend(df: pd.DataFrame, save_path: str = None):
    """Line chart of monthly order volume over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly = df.groupby('order_month').size()
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=4, color='#2196F3')
    ax.fill_between(monthly.index, monthly.values, alpha=0.1, color='#2196F3')
    ax.set_title('Monthly Order Volume Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Orders')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_profit_by_category(df: pd.DataFrame, save_path: str = None):
    """Box plot of profit per order by product category."""
    fig, ax = plt.subplots(figsize=(14, 7))
    top_categories = df['category_name'].value_counts().head(10).index
    subset = df[df['category_name'].isin(top_categories)]
    sns.boxplot(data=subset, x='category_name', y='order_profit_per_order', ax=ax, palette='viridis')
    ax.set_title('Profit Distribution by Top 10 Product Categories', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Profit per Order ($)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sales_by_region(df: pd.DataFrame, save_path: str = None):
    """Horizontal bar chart of total sales by market region."""
    fig, ax = plt.subplots(figsize=(10, 6))
    region_sales = df.groupby('market')['sales'].sum().sort_values()
    region_sales.plot(kind='barh', ax=ax, color=sns.color_palette("coolwarm", len(region_sales)), edgecolor='black')
    ax.set_title('Total Sales Revenue by Market Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Sales ($)')
    for i, v in enumerate(region_sales.values):
        ax.text(v + 1000, i, f'${v:,.0f}', va='center', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Heatmap of correlations between numeric columns."""
    fig, ax = plt.subplots(figsize=(14, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap — Numeric Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_shipping_delay_distribution(df: pd.DataFrame, save_path: str = None):
    """Histogram of shipping delay in days."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['shipping_delay_days'], bins=30, color='#FF7043', edgecolor='black', alpha=0.8)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='On Time')
    ax.set_title('Distribution of Shipping Delay (Days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Delay (days) — Negative = Early, Positive = Late')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
