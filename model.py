import os
import random

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.svm import SVR
from config import data_base_path
import requests
import retrying

forecast_price = {}

binance_data_path = os.path.join(data_base_path, "market-coin/trump-data")
MAX_DATA_SIZE = 1000  # Giới hạn số lượng dữ liệu tối đa khi lưu trữ


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, start_time=None):
    try:
        url = "{}{}".format(
            "https://clob.polymarket.com/prices-history?market=21742633143463906290569050155826241533067272736897614950488156847949938836455&fidelity=60&startTs=",
            start_time)

        response = requests.get(url)

        data = response.json()

        token_prices = data.get('history', {})

        result = []

        for item in token_prices:
            if item is not None:
                tmp = [item['t'], item['p']]
                result.append(tmp)

        return result
    except Exception as e:
        print(f'Failed to fetch prices for {symbol} from Binance API: {str(e)}')
        raise e


def download_data(token):
    now = datetime.now().utcnow()
    download_path = os.path.join(binance_data_path, token.lower())

    # Đường dẫn file CSV để lưu trữ
    file_path = os.path.join(download_path, f"{token.lower()}_90d_data.csv")
    # file_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")

    after90days = now - timedelta(days=90)

    after90day_unix = int(after90days.timestamp())

    new_data = fetch_prices(token, after90day_unix)

    # Chuyển dữ liệu thành DataFrame
    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "close"
    ])

    # Lưu dữ liệu đã kết hợp vào file CSV
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    new_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(new_df)}")


def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_90d_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)

    # Sử dụng các cột sau (đúng với dữ liệu bạn đã lưu)
    columns_to_use = [
        "start_time", "close"
    ]

    # Kiểm tra nếu tất cả các cột cần thiết tồn tại trong DataFrame
    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "close"
        ]
        df.index = pd.to_datetime(df["start_time"]*1000, unit='ms')
        df.index.name = "date"

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        print(f"Formatted data saved to {output_path}")
    else:
        print(f"Required columns are missing in {file_path}. Skipping this file.")


def train_model(token):
    # Load the token price data
    price_data = pd.read_csv(os.path.join(data_base_path, f"{token.lower()}_price_data.csv"))
    df = pd.DataFrame()

    # Convert 'date' to datetime
    price_data["date"] = pd.to_datetime(price_data["date"])

    # Set the date column as the index for resampling
    price_data.set_index("date", inplace=True)

    # Resample the data to 1D frequency and compute the mean price
    df = price_data.resample('1D').mean()

    # Prepare data for Linear Regression
    df = df.dropna()  # Loại bỏ các giá trị NaN (nếu có)
    X = np.array(range(len(df))).reshape(-1, 1)  # Sử dụng chỉ số thời gian làm đặc trưng
    y = df['close'].values  # Sử dụng giá đóng cửa làm mục tiêu

    # Khởi tạo mô hình Linear Regression
    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(X, y)  # Huấn luyện mô hình

    # Dự đoán giá tiếp theo
    next_time_index = np.array([[len(df)]])  # Giá trị thời gian tiếp theo
    predicted_price = model.predict(next_time_index)[0]  # Dự đoán giá

    # Xác định khoảng dao động xung quanh giá dự đoán
    fluctuation_range = 0.01 * predicted_price  # Lấy 1% của giá dự đoán làm khoảng dao động
    min_price = predicted_price - fluctuation_range
    max_price = predicted_price + fluctuation_range

    # Chọn ngẫu nhiên một giá trị trong khoảng dao động
    price_predict = random.uniform(min_price, max_price)
    forecast_price[token] = price_predict * 100

    print(f"Predicted_price: {predicted_price}, Min_price: {min_price}, Max_price: {max_price}")
    print(f"Forecasted price for {token}: {forecast_price[token]}")


def update_data():
    tokens = ["R"]
    for token in tokens:
        download_data(token)
        format_data(token)
        train_model(token)


if __name__ == "__main__":
    update_data()
