import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_stock_data(data, title="Stock Price Analysis"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Set the main title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot for price with legends
    ax1.plot(data.index, data['Close'],
             label='Close Price', linewidth=2, color='blue')
    ax1.plot(data.index, data['ma_5'],
             label='5-day MA', alpha=0.7, color='orange')
    ax1.plot(data.index, data['ma_20'],
             label='20-day MA', alpha=0.7, color='red')

    # Add legend to price plot
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Stock Price and Moving Averages', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Volume plot with legend
    ax2.bar(data.index, data['Volume'], alpha=0.6,
            color='orange', label='Daily Volume')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Trading Volume', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Formatting the dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Rotate date labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def create_current_stats_table(stats):
    print("\n" + "="*50)
    print(f"CURRENT STOCK DATA - {stats['symbol']}")
    print("="*50)
    print(f"Current Price:    ${stats['current_price']:.2f}")
    print(
        f"Daily Change:     ${stats['daily_change']:+.2f} ({stats['daily_change_pct']:+.2f}%)")
    print(f"Today's High:     ${stats['high']:.2f}")
    print(f"Today's Low:      ${stats['low']:.2f}")
    print(f"Volume:           {stats['volume']:,}")
    print(f"Last Updated:     {stats['timestamp']}")
    print("="*50)
