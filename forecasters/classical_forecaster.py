import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

class ClassicalForecaster:
    """
    Класс для прогнозирования временных рядов с использованием классических моделей.

    Поддерживаемые модели:
    - ARIMA (Авторегрессионное интегрированное скользящее среднее)
    - SARIMA (Сезонное ARIMA)
    - ETS (Экспоненциальное сглаживание)

    Атрибуты:
        data (pd.DataFrame | None): Агрегированные данные по всему магазину.
        train_data (pd.DataFrame | None): Обучающая выборка для всего магазина.
        test_data (pd.DataFrame | None): Тестовая выборка для всего магазина.
        models (dict): Словарь для хранения обученных моделей.
        forecasts (dict): Словарь для хранения прогнозов.
        metrics (dict): Словарь для хранения метрик оценки моделей.
        store_id (str | None): Идентификатор магазина, для которого выполнялся анализ.
        items_data (dict): Словарь, где ключи - item_id, значения - DataFrame с данными для этого товара.
        items (list): Список уникальных item_id для выбранного магазина.
    """

    def __init__(self):
        """Инициализация класса ClassicalForecaster."""
        self.data = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.store_id = None
        self.items_data = {}
        self.items = []

    def load_and_preprocess_data(self, sales_path, prices_path, dates_path, store_id='STORE_2'):
        """
        Загружает и предобрабатывает данные, фильтруя по магазину и организуя по товарам.
        Использует оригинальную логику интерполяции пользователя (с заменой 0 на NaN).

        Args:
            sales_path (str): Путь к файлу с данными о продажах (CSV).
            prices_path (str): Путь к файлу с данными о ценах (CSV).
            dates_path (str): Путь к файлу с календарными данными (CSV).
            store_id (str, optional): Идентификатор магазина для анализа. По умолчанию 'STORE_2'.

        Returns:
            dict: Словарь, где ключи - item_id, а значения - pd.DataFrame с обработанными временными рядами для каждого товара.

        Raises:
            FileNotFoundError: Если один из файлов данных не найден.
        """
        self.store_id = store_id
        print(f"Загрузка данных для магазина: {self.store_id}")

        try:
            # Загрузка данных из CSV файлов
            sales = pd.read_csv(sales_path)
            prices = pd.read_csv(prices_path)
            dates = pd.read_csv(dates_path)

        except FileNotFoundError as e:
            print(f"Ошибка: Файл не найден. {e}")
            raise
        # Сохраняем копию данных для дальнейшего использования
        sales = sales[sales['store_id'] == store_id].copy()

        if sales.empty:
            print(f"Предупреждение: Нет данных о продажах для магазина {store_id}")
            self.items = []

        else:
            print(f"Данные отфильтрованы по магазину {store_id}")
            self.items = sales['item_id'].unique()
            print(f"Найдено {len(self.items)} уникальных товаров в магазине {store_id}")
        # Преобразуем даты в формат datetime
        dates['date'] = pd.to_datetime(dates['date'])

        date_cols = ['date_id', 'date', 'wm_yr_wk']
        event_cols = ['event_name_1', 'event_type_1',
                      'event_name_2', 'event_type_2']

        for col in event_cols:
            if col in dates.columns:
                date_cols.append(col)

        cashback_col = f'CASHBACK_{store_id}'
        if cashback_col in dates.columns:
            date_cols.append(cashback_col)
        else:
            print(f"Предупреждение: Колонка кэшбэка {cashback_col} не найдена.")
        # Объединяем данные о продажах, ценах и датах
        merged_data = pd.merge(sales, dates[date_cols], on='date_id', how='left')
        merged_data = pd.merge(
            merged_data,
            prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )
        # Подсчитываем среднюю цену для каждого товара
        price_means = merged_data.groupby('item_id')['sell_price'].transform('mean')
        # Заполняем пропуски в ценах
        merged_data['sell_price'] = merged_data['sell_price'].fillna(price_means)
        merged_data['sell_price'] = merged_data['sell_price'].fillna(method='bfill').fillna(method='ffill')
        if merged_data['sell_price'].isnull().any():
             merged_data['sell_price'] = merged_data['sell_price'].fillna(0)
        # Подсчет выручки
        merged_data['revenue'] = merged_data['cnt'] * merged_data['sell_price']

        # Определяем диапазон дат для интерполяции
        all_dates = pd.date_range(
            start=dates['date'].min(),
            end=dates['date'].max(),
            freq='D'
        )

        print("Обработка данных по товарам...")
        for item_id in self.items:
            item_data = merged_data[merged_data['item_id'] == item_id].copy()
            agg_dict = {
                'cnt': 'sum',
                'revenue': 'sum',
                'sell_price': 'mean'
            }

            for col in event_cols:
                if col in item_data.columns:
                    item_data[col] = item_data[col].fillna('no_event')
                    agg_dict[col] = 'first'

            if cashback_col in item_data.columns:
                item_data[cashback_col] = item_data[cashback_col].fillna(0)
                agg_dict[cashback_col] = 'first'

            # Группируем данные по дате и агрегируем
            daily_data = item_data.groupby('date').agg(agg_dict).reset_index()
            daily_data.set_index('date', inplace=True)
            daily_data = daily_data.reindex(all_dates)

            # Интерполяция пропусков
            for col in ['cnt', 'revenue', 'sell_price']:
                daily_data[col] = daily_data[col].replace(0, np.nan)
                daily_data[col] = daily_data[col].interpolate(method='linear')
                daily_data[col] = daily_data[col].fillna(method='bfill').fillna(method='ffill')
                if daily_data[col].isnull().any():
                     if col == 'sell_price':
                          item_mean_price = merged_data.loc[merged_data['item_id'] == item_id, 'sell_price'].mean()
                          daily_data[col] = daily_data[col].fillna(item_mean_price if pd.notna(item_mean_price) else 0)
                     else:
                          daily_data[col] = daily_data[col].fillna(0)

            for col in agg_dict:
                 if col not in ['cnt', 'revenue', 'sell_price']:
                     daily_data[col] = daily_data[col].fillna(method='ffill') \
                                                      .fillna(method='bfill')
                     if daily_data[col].isnull().any():
                          if 'event' in col:
                               daily_data[col] = daily_data[col].fillna('no_event')
                          elif 'CASHBACK' in col:
                               daily_data[col] = daily_data[col].fillna(0)

            self.items_data[item_id] = daily_data

        print("Агрегация данных по всему магазину...")
        agg_dict_store = {
            'cnt': 'sum',
            'revenue': 'sum'
        }

        for col in event_cols:
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna('no_event')
                agg_dict_store[col] = 'first'

        if cashback_col in merged_data.columns:
            merged_data[cashback_col] = merged_data[cashback_col].fillna(0)
            # Исправлена потенциальная ошибка в оригинале:
            agg_dict_store[cashback_col] = 'first'

        store_daily_data = merged_data.groupby('date') \
                                      .agg(agg_dict_store).reset_index()
        store_daily_data.set_index('date', inplace=True)
        store_daily_data = store_daily_data.reindex(all_dates)

        store_daily_data['cnt'] = store_daily_data['cnt'].replace(0, np.nan).interpolate(method='linear')
        store_daily_data['revenue'] = store_daily_data['revenue'].replace(0, np.nan).interpolate(method='linear')

        for col in agg_dict_store:
             if col not in ['cnt', 'revenue']:
                 store_daily_data[col] = store_daily_data[col].fillna(method='ffill').fillna(method='bfill')
                 if store_daily_data[col].isnull().any():
                      if 'event' in col:
                           store_daily_data[col] = store_daily_data[col].fillna('no_event')
                      elif 'CASHBACK' in col:
                           store_daily_data[col] = store_daily_data[col].fillna(0)

        store_daily_data['cnt'] = store_daily_data['cnt'].fillna(method='bfill').fillna(method='ffill')
        store_daily_data['revenue'] = store_daily_data['revenue'].fillna(method='bfill').fillna(method='ffill')
        store_daily_data['cnt'] = store_daily_data['cnt'].fillna(0)
        store_daily_data['revenue'] = store_daily_data['revenue'].fillna(0)

        self.data = store_daily_data

        print("Данные загружены и предобработаны.")
        if not self.data.empty:
            print(f"Период данных: с {self.data.index.min().date()} по "
                  f"{self.data.index.max().date()}")
        else:
            print("Агрегированные данные по магазину пусты.")

        return self.items_data

    def train_test_split(self, test_size=0.2):
        """
        Разделяет данные на обучающую и тестовую выборки для всех товаров и для агрегированных данных магазина.

        Args:
            test_size (float, optional): Доля тестовой выборки (от 0 до 1). По умолчанию 0.2.

        Returns:
            dict: Словарь, где ключи - item_id, а значения - кортежи (train_data, test_data) для каждого товара.

        Raises:
            ValueError: Если данные по товарам (items_data) не загружены или агрегированные данные (self.data) отсутствуют/некорректны.
        """
        if not self.items_data:
            raise ValueError("Данные по товарам (items_data) не загружены. "
                             "Сначала вызовите метод load_and_preprocess_data")
        if self.data is None or self.data.empty:
             raise ValueError("Агрегированные данные (self.data) отсутствуют или пусты.")

        split_results = {}
        items_to_remove = []

        for item_id, item_data in self.items_data.items():
            if len(item_data) < 5:
                 print(f"Предупреждение: Слишком мало данных ({len(item_data)}) для товара {item_id}. Пропуск разделения.")
                 items_to_remove.append(item_id)
                 continue

            split_idx = int(len(item_data) * (1 - test_size))

            if split_idx < 1 or (len(item_data) - split_idx) < 1:
                 print(f"Предупреждение: Не удалось разделить данные для товара {item_id} "
                       f"(train: {split_idx}, test: {len(item_data) - split_idx}). Пропуск.")
                 items_to_remove.append(item_id)
                 continue

            train_data_item = item_data.iloc[:split_idx].copy()
            test_data_item = item_data.iloc[split_idx:].copy()
            split_results[item_id] = (train_data_item, test_data_item)

        if items_to_remove:
             print(f"Удаление {len(items_to_remove)} товаров из анализа из-за недостатка данных для разделения.")
             for item_id in items_to_remove:
                  if item_id in self.items_data: del self.items_data[item_id]
                  if item_id in split_results: del split_results[item_id]
             self.items = list(self.items_data.keys())

        if len(self.data) < 5:
             raise ValueError(f"Слишком мало агрегированных данных ({len(self.data)}) для разделения.")

        split_idx_store = int(len(self.data) * (1 - test_size))
        if split_idx_store < 1 or (len(self.data) - split_idx_store) < 1:
             raise ValueError(f"Не удалось разделить агрегированные данные "
                              f"(train: {split_idx_store}, test: {len(self.data) - split_idx_store}).")

        self.train_data = self.data.iloc[:split_idx_store].copy()
        self.test_data = self.data.iloc[split_idx_store:].copy()

        print(f"Данные разделены на обучающую и тестовую выборки для {len(split_results)} товаров.")
        print(f"Период обучающих данных (агрег.): с {self.train_data.index.min().date()} "
              f"по {self.train_data.index.max().date()}")
        print(f"Период тестовых данных (агрег.): с {self.test_data.index.min().date()} "
              f"по {self.test_data.index.max().date()}")

        return split_results

    def train_arima_model(self, target_col, order=(2, 1, 2), item_id=None):
        """
        Обучает модель ARIMA для конкретного товара или для всего магазина.

        Args:
            target_col (str): Столбец для прогнозирования.
            order (tuple, optional): Параметры (p, d, q) для ARIMA. По умолчанию (2, 1, 2).
            item_id (str | None, optional): Идентификатор товара. Если None, обучается модель для всего магазина.
                                            По умолчанию None.

        Returns:
            statsmodels.tsa.arima.model.ARIMAResultsWrapper: Обученная модель.

        Raises:
            ValueError: Если данные для товара не найдены или обучающие данные отсутствуют/некорректны.
            Exception: Другие ошибки при обучении модели ARIMA.
        """
        train_data_source = None
        model_key = None
        item_str = ""

        if item_id is None:
            if self.train_data is None:
                 raise ValueError("Агрегированные обучающие данные (self.train_data) отсутствуют.")
            train_data_source = self.train_data
            model_key = f'ARIMA_{target_col}'
            item_str = " всего магазина"
        else:
            if item_id not in self.items_data:
                raise ValueError(f"Данные для товара {item_id} не найдены в self.items_data.")
            if self.train_data is None:
                 raise ValueError("Невозможно определить обучающий период для товара без self.train_data.")
            split_point = len(self.train_data)
            item_full_data = self.items_data[item_id]
            if split_point <= 0 or split_point > len(item_full_data):
                 raise ValueError(f"Некорректная точка разделения ({split_point}) для данных товара {item_id} (длина {len(item_full_data)}).")
            train_data_source = item_full_data.iloc[:split_point]
            model_key = f'ARIMA_{item_id}_{target_col}'
            item_str = f" товара {item_id}"

        print(f"Обучение модели ARIMA {order} для {target_col}{item_str}")

        if target_col not in train_data_source.columns:
             raise ValueError(f"Столбец '{target_col}' не найден в обучающих данных{item_str}.")

        train_series = train_data_source[target_col].copy()
        if train_series.isnull().any():
            mean_val = train_series.mean()
            if pd.isna(mean_val):
                 raise ValueError(f"Невозможно заполнить NaN в '{target_col}', так как все значения NaN.")
            train_series.fillna(mean_val, inplace=True)
            print(f"Предупреждение: Обнаружены и заполнены пропуски в '{target_col}' "
                  f"средним значением ({mean_val:.2f}) перед обучением.")

        try:
            model = ARIMA(train_series, order=order)
            fitted_model = model.fit()
            self.models[model_key] = fitted_model
            self.models[model_key].item_id = item_id
            return fitted_model
        except Exception as e:
            print(f"Ошибка при обучении ARIMA {model_key}: {e}")
            raise

    def train_sarima_model(self, target_col, order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 7), item_id=None):
        """
        Обучает модель SARIMA для конкретного товара или для всего магазина.

        Args:
            target_col (str): Столбец для прогнозирования.
            order (tuple, optional): Параметры (p, d, q) для ARIMA части. По умолчанию (1, 1, 1).
            seasonal_order (tuple, optional): Сезонные параметры (P, D, Q, s). По умолчанию (1, 1, 1, 7).
            item_id (str | None, optional): Идентификатор товара. Если None, обучается модель для всего магазина.
                                            По умолчанию None.

        Returns:
            statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper: Обученная модель.

        Raises:
            ValueError: Если данные для товара не найдены или обучающие данные отсутствуют/некорректны.
            Exception: Другие ошибки при обучении модели SARIMA.
        """
        train_data_source = None
        model_key = None
        item_str = ""

        if item_id is None:
            if self.train_data is None:
                 raise ValueError("Агрегированные обучающие данные (self.train_data) отсутствуют.")
            train_data_source = self.train_data
            model_key = f'SARIMA_{target_col}'
            item_str = " всего магазина"
        else:
            if item_id not in self.items_data:
                raise ValueError(f"Данные для товара {item_id} не найдены в self.items_data.")
            if self.train_data is None:
                 raise ValueError("Невозможно определить обучающий период для товара без self.train_data.")
            split_point = len(self.train_data)
            item_full_data = self.items_data[item_id]
            if split_point <= 0 or split_point > len(item_full_data):
                 raise ValueError(f"Некорректная точка разделения ({split_point}) для данных товара {item_id} (длина {len(item_full_data)}).")
            train_data_source = item_full_data.iloc[:split_point]
            model_key = f'SARIMA_{item_id}_{target_col}'
            item_str = f" товара {item_id}"

        print(f"Обучение модели SARIMA {order} {seasonal_order} для {target_col}{item_str}")

        if target_col not in train_data_source.columns:
             raise ValueError(f"Столбец '{target_col}' не найден в обучающих данных{item_str}.")

        train_series = train_data_source[target_col].copy()
        if train_series.isnull().any():
            mean_val = train_series.mean()
            if pd.isna(mean_val):
                 raise ValueError(f"Невозможно заполнить NaN в '{target_col}', так как все значения NaN.")
            train_series.fillna(mean_val, inplace=True)
            print(f"Предупреждение: Обнаружены и заполнены пропуски в '{target_col}' "
                  f"средним значением ({mean_val:.2f}) перед обучением.")

        try:
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            self.models[model_key] = fitted_model
            self.models[model_key].item_id = item_id
            return fitted_model
        except Exception as e:
            print(f"Ошибка при обучении SARIMA {model_key}: {e}")
            raise

    def train_ets_model(self, target_col, seasonal_periods=7, trend='add',
                        seasonal='add', damped=True, item_id=None):
        """
        Обучает модель экспоненциального сглаживания (ETS).

        Args:
            target_col (str): Столбец для прогнозирования.
            seasonal_periods (int, optional): Период сезонности. По умолчанию 7.
            trend (str | None, optional): Тип тренда ('add', 'mul', None). По умолчанию 'add'.
            seasonal (str | None, optional): Тип сезонности ('add', 'mul', None). По умолчанию 'add'.
            damped (bool, optional): Использовать ли затухающий тренд (damped_trend в statsmodels).
                                     По умолчанию True.
            item_id (str | None, optional): Идентификатор товара. Если None, обучается модель для всего магазина.
                                            По умолчанию None.

        Returns:
            statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper: Обученная модель.

        Raises:
            ValueError: Если данные для товара не найдены или обучающие данные отсутствуют/некорректны.
            Exception: Другие ошибки при обучении модели ETS.
        """
        train_data_source = None
        model_key = None
        item_str = ""

        if item_id is None:
            if self.train_data is None:
                 raise ValueError("Агрегированные обучающие данные (self.train_data) отсутствуют.")
            train_data_source = self.train_data
            model_key = f'ETS_{target_col}'
            item_str = " всего магазина"
        else:
            if item_id not in self.items_data:
                raise ValueError(f"Данные для товара {item_id} не найдены в self.items_data.")
            if self.train_data is None:
                 raise ValueError("Невозможно определить обучающий период для товара без self.train_data.")
            split_point = len(self.train_data)
            item_full_data = self.items_data[item_id]
            if split_point <= 0 or split_point > len(item_full_data):
                 raise ValueError(f"Некорректная точка разделения ({split_point}) для данных товара {item_id} (длина {len(item_full_data)}).")
            train_data_source = item_full_data.iloc[:split_point]
            model_key = f'ETS_{item_id}_{target_col}'
            item_str = f" товара {item_id}"

        print(f"Обучение модели ETS с трендом={trend}, сезонность={seasonal}, "
              f"период={seasonal_periods}, damped={damped} для {target_col}{item_str}")

        if target_col not in train_data_source.columns:
             raise ValueError(f"Столбец '{target_col}' не найден в обучающих данных{item_str}.")

        train_series = train_data_source[target_col].copy()
        if train_series.isnull().any():
            mean_val = train_series.mean()
            if pd.isna(mean_val):
                 raise ValueError(f"Невозможно заполнить NaN в '{target_col}', так как все значения NaN.")
            train_series.fillna(mean_val, inplace=True)
            print(f"Предупреждение: Обнаружены и заполнены пропуски в '{target_col}' "
                  f"средним значением ({mean_val:.2f}) перед обучением.")

        offset = 0
        if (trend == 'mul' or seasonal == 'mul') and (train_series <= 0).any():
            min_val = train_series.min()
            if min_val <= 0:
                offset = abs(min_val) + 0.1
                train_series = train_series + offset
                print(f"Предупреждение: значения временно смещены на {offset:.2f} "
                      "для совместимости с мультипликативной моделью ETS.")

        use_seasonal_periods = None
        if seasonal is not None:
             if seasonal_periods is None or seasonal_periods < 2:
                  print(f"Предупреждение: Некорректный seasonal_periods ({seasonal_periods}). Сезонность отключена.")
                  seasonal = None
             else:
                  use_seasonal_periods = seasonal_periods

        use_damped = False
        if trend is not None:
             use_damped = damped

        try:
            model = ExponentialSmoothing(
                train_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=use_seasonal_periods,
                damped_trend=use_damped
            )
            fitted_model = model.fit()
            self.models[model_key] = fitted_model
            self.models[model_key].item_id = item_id
            return fitted_model
        except Exception as e:
            print(f"Ошибка при обучении ETS {model_key}: {e}")
            raise

    def forecast(self, model_name, steps):
        """
        Генерирует прогноз с использованием обученной модели.

        Args:
            model_name (str): Название модели (ключ в self.models).
            steps (int): Количество шагов (дней) для прогнозирования.

        Returns:
            pd.DataFrame: DataFrame с прогнозом, индексированный по дате.

        Raises:
            ValueError: Если модель с таким именем не найдена, steps некорректно, или не удалось определить последнюю дату обучения.
            Exception: Другие ошибки при прогнозировании.
        """
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена")
        if not isinstance(steps, int) or steps <= 0:
             raise ValueError("Количество шагов (steps) должно быть положительным целым числом.")

        print(f"Прогнозирование с помощью модели {model_name} на {steps} шагов вперед...")
        model = self.models[model_name]
        item_id = getattr(model, 'item_id', None)

        last_train_date = None
        train_data_ref = None
        if item_id is None:
            if self.train_data is not None:
                 train_data_ref = self.train_data
                 last_train_date = self.train_data.index[-1]
        else:
            if self.train_data is not None and item_id in self.items_data:
                 split_point = len(self.train_data)
                 item_full_data = self.items_data[item_id]
                 if split_point > 0 and split_point <= len(item_full_data):
                      train_data_ref = item_full_data.iloc[:split_point]
                      last_train_date = train_data_ref.index[-1]

        if last_train_date is None:
             raise ValueError(f"Не удалось определить последнюю дату обучения для {model_name}.")

        try:
            if 'ARIMA' in model_name or 'SARIMA' in model_name:
                forecast_values = model.forecast(steps)
            else:
                forecast_values = model.forecast(steps)
        except Exception as e:
            print(f"Ошибка при генерации прогноза моделью {model_name}: {e}")
            raise

        parts = model_name.split('_')
        if len(parts) < 2:
             raise ValueError(f"Не удалось извлечь target_col из имени модели {model_name}")
        target_col = parts[-1]

        forecast_index = pd.date_range(
            start=last_train_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        if len(forecast_values) != steps:
             print(f"Предупреждение: Длина прогноза ({len(forecast_values)}) не совпадает со steps ({steps}).")
             min_len = min(len(forecast_values), steps)
             forecast_values = forecast_values[:min_len]
             forecast_index = forecast_index[:min_len]

        forecast_df = pd.DataFrame(
            {target_col: forecast_values},
            index=forecast_index
        )
        self.forecasts[model_name] = forecast_df
        return forecast_df

    def evaluate(self, model_name, horizon='week'):
        """
        Оценивает качество прогноза модели на заданном горизонте.

        Args:
            model_name (str): Название модели (ключ в self.forecasts).
            horizon (str, optional): Горизонт оценки ('week', 'month', 'quarter'). По умолчанию 'week'.

        Returns:
            tuple: Кортеж метрик (mae, rmse, mape, r2).

        Raises:
            ValueError: Если прогноз для модели не найден, горизонт некорректен, или не удалось найти/выровнять тестовые данные.
            Exception: Другие ошибки при расчете метрик.
        """
        if model_name not in self.forecasts:
            raise ValueError(f"Прогноз для модели {model_name} не найден")
        if model_name not in self.models:
             raise ValueError(f"Модель {model_name} не найдена, хотя прогноз существует.")

        horizon_days = {'week': 7, 'month': 30, 'quarter': 90}
        if horizon not in horizon_days:
            raise ValueError("Горизонт должен быть 'week', 'month' или 'quarter'")
        days = horizon_days[horizon]

        model = self.models[model_name]
        item_id = getattr(model, 'item_id', None)

        forecast_all = self.forecasts[model_name]
        if len(forecast_all) < days:
             print(f"Предупреждение: Длина прогноза ({len(forecast_all)}) меньше горизонта ({days}). Оценка на доступных данных.")
             days = len(forecast_all)
        forecast = forecast_all.iloc[:days]

        if forecast.empty:
             raise ValueError(f"Срез прогноза для горизонта {horizon} пуст.")

        actual_data_source = None
        item_str = ""
        target_col = None

        parts = model_name.split('_')
        if len(parts) < 2:
             raise ValueError(f"Не удалось извлечь target_col из имени модели {model_name}")
        target_col = parts[-1]

        if item_id is None:
            if self.test_data is None or self.test_data.empty:
                 raise ValueError("Агрегированные тестовые данные (self.test_data) отсутствуют.")
            actual_data_source = self.test_data
            item_str = " всего магазина"
        else:
            if item_id not in self.items_data:
                 raise ValueError(f"Данные для товара {item_id} не найдены в self.items_data.")
            if self.train_data is None:
                 raise ValueError("Невозможно определить тестовый период для товара без self.train_data.")
            split_point = len(self.train_data)
            item_full_data = self.items_data[item_id]
            if split_point >= len(item_full_data):
                 raise ValueError(f"Точка разделения ({split_point}) не позволяет получить тестовые данные для товара {item_id} (длина {len(item_full_data)}).")
            actual_data_source = item_full_data.iloc[split_point:]
            if actual_data_source.empty:
                 raise ValueError(f"Тестовые данные для товара {item_id} пусты.")
            item_str = f" товара {item_id}"

        if target_col not in actual_data_source.columns:
             raise ValueError(f"Столбец '{target_col}' отсутствует в тестовых данных{item_str}.")

        try:
            actual_aligned = actual_data_source[target_col].reindex(forecast.index)
        except Exception as e:
             raise ValueError(f"Ошибка при выравнивании фактических данных с прогнозом: {e}")

        valid_idx = actual_aligned.dropna().index
        if valid_idx.empty:
             raise ValueError("Нет пересекающихся валидных данных между прогнозом и фактом для оценки.")

        actual = actual_aligned.loc[valid_idx]
        forecast_eval = forecast[target_col].loc[valid_idx]

        try:
            mae = mean_absolute_error(actual, forecast_eval)
            rmse = np.sqrt(mean_squared_error(actual, forecast_eval))

            mask = actual != 0
            if np.sum(mask) > 0:
                 mape = np.mean(np.abs((actual[mask] - forecast_eval[mask]) / actual[mask])) * 100
            else:
                 mape = np.inf

            if len(actual) > 1:
                 r2 = r2_score(actual, forecast_eval)
            else:
                 r2 = np.nan

        except Exception as e:
            print(f"Ошибка при расчете метрик для {model_name}: {e}")
            raise

        metric_key = (model_name, horizon)
        self.metrics[metric_key] = {
            'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2
        }

        print(f"Метрики для модели {model_name}, горизонт {horizon}{item_str}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R2: {r2:.2f}")

        return mae, rmse, mape, r2

    def visualize_forecast(self, model_name, horizon='week'):
        """
        Визуализирует прогноз модели на фоне фактических и обучающих данных.
        (Использует оригинальную логику пользователя для получения данных)

        Args:
            model_name (str): Название модели (ключ в self.forecasts).
            horizon (str, optional): Горизонт визуализации ('week', 'month','quarter'). По умолчанию 'week'.

        Returns:
            tuple: Кортеж (train_data, test_data, forecast) с данными, использованными для графика (как в оригинале).

        Raises:
            ValueError: Если прогноз для модели не найден, горизонт некорректен, или не удалось получить данные для визуализации.
        """
        if model_name not in self.forecasts:
            raise ValueError(f"Прогноз для модели {model_name} не найден")
        if model_name not in self.models:
             raise ValueError(f"Модель {model_name} не найдена.")

        horizon_days = {'week': 7, 'month': 30, 'quarter': 90}
        if horizon not in horizon_days:
            raise ValueError("Горизонт должен быть 'week', 'month' или 'quarter'")
        days = horizon_days[horizon]

        model = self.models[model_name]
        item_id = getattr(model, 'item_id', None)

        forecast_all = self.forecasts[model_name]
        if len(forecast_all) < days:
             days = len(forecast_all)
        forecast = forecast_all.iloc[:days]

        if forecast.empty:
             raise ValueError(f"Срез прогноза для горизонта {horizon} пуст.")

        train_data_vis = None
        test_data_vis = None
        item_str = ""
        target_col = None

        if item_id is not None:
            model_parts = model_name.split('_')
            if len(model_parts) < 2:
                 raise ValueError(f"Не удалось извлечь target_col из имени модели {model_name}")
            target_col = model_parts[-1]

            if item_id not in self.items_data:
                 raise ValueError(f"Данные для товара {item_id} не найдены.")
            item_data = self.items_data[item_id]
            if self.train_data is None:
                 raise ValueError("Невозможно определить период обучения/теста без self.train_data.")

            train_split = len(self.train_data)
            train_data_vis = item_data[target_col].iloc[max(0, train_split - 30):train_split]
            test_data_vis = item_data[target_col].iloc[train_split:train_split + days]
            item_str = f' (товар {item_id})'
        else:
            if '_' not in model_name:
                 raise ValueError(f"Не удалось извлечь target_col из имени модели {model_name}")
            target_col = model_name.split('_', 1)[1]

            if self.train_data is None or self.test_data is None:
                 raise ValueError("Агрегированные обучающие/тестовые данные отсутствуют.")
            train_data_vis = self.train_data[target_col].iloc[-30:]
            test_data_vis = self.test_data[target_col].iloc[:days]
            item_str = ' (агрег.)'

        if target_col not in forecast.columns:
             raise ValueError(f"Столбец '{target_col}' отсутствует в прогнозе {model_name}.")

        plt.figure(figsize=(12, 6))
        if train_data_vis is not None and not train_data_vis.empty:
             plt.plot(train_data_vis.index, train_data_vis, label='Обучающие данные (последние)', color='blue')
        if test_data_vis is not None and not test_data_vis.empty:
             plt.plot(test_data_vis.index, test_data_vis, label='Фактические данные (тест)', color='green')
        if not forecast.empty:
             plt.plot(forecast.index, forecast[target_col], label='Прогноз', color='red', linestyle='--')

        title = f'Прогноз {target_col} моделью {model_name}, горизонт {horizon}{item_str}'

        plt.title(title)
        plt.xlabel('Дата')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return train_data_vis, test_data_vis, forecast

    def find_best_classical_model(self, target_col, horizon='week',
                                  verbose=False, item_id=None):
        """
        Находит лучшую классическую модель (ARIMA, SARIMA, ETS) для прогнозирования на заданном горизонте, сравнивая их по MAE.

        Args:
            target_col (str): Целевой столбец для прогнозирования.
            horizon (str, optional): Горизонт прогнозирования ('week', 'month','quarter'). По умолчанию 'week'.
            verbose (bool, optional): Если True, выводит подробную информацию о процессе тестирования моделей.
                                      По умолчанию False.
            item_id (str | None, optional): Идентификатор товара. Если None, ищет лучшую модель для всего магазина.
                                            По умолчанию None.

        Returns:
            tuple: Кортеж (best_model_name, best_metrics), где
                   best_model_name (str) - название (ключ) лучшей модели,
                   best_metrics (dict) - словарь с метриками лучшей модели.

        Raises:
            ValueError: Если не удалось успешно обучить, спрогнозировать и оценить ни одну модель из списка `models_to_test`.
        """
        item_str = f" для товара {item_id}" if item_id else " для всего магазина"
        if verbose:
            print(f"\nПоиск лучшей модели для прогнозирования {target_col} "
                  f"на горизонте {horizon}{item_str}...")
        else:
            print(f"\nПоиск лучшей модели для {target_col} ({horizon}){item_str}...")

        models_to_test = [
            ('ARIMA', {'order': (2, 1, 2)}),
            ('ARIMA', {'order': (3, 1, 3)}),
            ('SARIMA', {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)}),
            ('SARIMA', {'order': (1, 1, 2), 'seasonal_order': (0, 1, 2, 7)}),
            ('SARIMA', {'order': (2, 1, 2), 'seasonal_order': (2, 1, 2, 7)}),
            ('ETS', {'seasonal_periods': 7, 'trend': 'add', 'seasonal': 'add', 'damped': True}),
            ('ETS', {'seasonal_periods': 7, 'trend': 'mul', 'seasonal': 'mul', 'damped': True}),
        ]

        results = []
        best_model_name = None
        best_mae = float('inf')

        horizon_days = {'week': 7, 'month': 30, 'quarter': 90}
        if horizon not in horizon_days:
             raise ValueError("Горизонт должен быть 'week', 'month' или 'quarter'.")
        steps = horizon_days[horizon]

        for i, (model_type, params) in enumerate(models_to_test):
            model_key = None
            model_desc = "N/A"
            original_stdout = sys.stdout
            devnull = None
            if not verbose:
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                except OSError:
                    print("Предупреждение: Не удалось перенаправить stdout.")
                    verbose = True

            try:
                if model_type == 'ARIMA':
                    model_desc = f"ARIMA{params['order']}"
                    model_key = f'ARIMA_{item_id}_{target_col}' if item_id else f'ARIMA_{target_col}'
                elif model_type == 'SARIMA':
                    model_desc = f"SARIMA{params['order']}{params['seasonal_order']}"
                    model_key = f'SARIMA_{item_id}_{target_col}' if item_id else f'SARIMA_{target_col}'
                else:
                    trend = params.get('trend')
                    seasonal = params.get('seasonal')
                    periods = params.get('seasonal_periods')
                    damped = params.get('damped')
                    model_desc = f"ETS(t={trend},s={seasonal},p={periods},d={damped})"
                    model_key = f'ETS_{item_id}_{target_col}' if item_id else f'ETS_{target_col}'

                if verbose:
                    if devnull: sys.stdout = original_stdout
                    print(f"\nТестирование {model_desc}{item_str}")
                    if devnull: sys.stdout = devnull

                if model_type == 'ARIMA':
                    self.train_arima_model(target_col, item_id=item_id, **params.copy())
                elif model_type == 'SARIMA':
                    self.train_sarima_model(target_col, item_id=item_id, **params.copy())
                elif model_type == 'ETS':
                    self.train_ets_model(target_col, item_id=item_id, **params.copy())

                self.forecast(model_key, steps=steps)
                metrics_tuple = self.evaluate(model_key, horizon=horizon)
                mae, rmse, mape, r2 = metrics_tuple

                if devnull:
                    sys.stdout = original_stdout
                    devnull.close()
                    devnull = None

                results.append({
                    'model_key': model_key,
                    'model': model_desc,
                    'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2
                })

                if mae < best_mae:
                    best_mae = mae
                    best_model_name = model_key

            except Exception as e:
                if devnull:
                    sys.stdout = original_stdout
                    devnull.close()
                    devnull = None
                print(f"Ошибка при тестировании модели {model_desc}{item_str}: {e}")
                if model_key:
                     if model_key in self.models: del self.models[model_key]
                     if model_key in self.forecasts: del self.forecasts[model_key]
                     if (model_key, horizon) in self.metrics: del self.metrics[(model_key, horizon)]
                continue
            finally:
                 if devnull:
                      sys.stdout = original_stdout
                      devnull.close()

        if not results:
            raise ValueError(f"Не удалось успешно протестировать ни одну модель{item_str}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('MAE')

        print(f"\nРезультаты тестирования моделей{item_str}:")
        print(results_df[['model', 'MAE', 'RMSE', 'MAPE', 'R2']].round(2).to_string(index=False))

        best_result_row = results_df.iloc[0]
        print(f"\nЛучшая модель{item_str}: {best_result_row['model']}")
        print(f"  MAE: {best_result_row['MAE']:.2f}, RMSE: {best_result_row['RMSE']:.2f}, "
              f"MAPE: {best_result_row['MAPE']:.2f}%, R2: {best_result_row['R2']:.2f}")

        metrics = self.metrics.get((best_model_name, horizon))
        if metrics is None:
             print(f"Предупреждение: Метрики для {best_model_name} не найдены в self.metrics, берем из таблицы.")
             metrics = best_result_row[['MAE', 'RMSE', 'MAPE', 'R2']].to_dict()

        return best_model_name, metrics

    def predict_with_best_model(self, target_col, horizon='week', steps=None,
                                item_id=None):
        """
        Находит лучшую модель и генерирует прогноз с ее помощью.

        Args:
            target_col (str): Целевой столбец.
            horizon (str, optional): Горизонт для поиска лучшей модели ('week', 'month', 'quarter').
                                     По умолчанию 'week'.
            steps (int | None, optional): Количество шагов для прогноза.
                                          Если None, определяется по горизонту.
                                          По умолчанию None.
            item_id (str | None, optional): Идентификатор товара. Если None, используется модель для всего магазина.
                                            По умолчанию None.

        Returns:
            pd.DataFrame: DataFrame с прогнозом от лучшей модели.

        Raises:
            ValueError: Если не удалось найти лучшую модель или сгенерировать прогноз.
        """
        best_model_name, _ = self.find_best_classical_model(
            target_col, horizon=horizon, item_id=item_id, verbose=False
        )

        if steps is None:
            horizon_days = {'week': 7, 'month': 30, 'quarter': 90}
            if horizon not in horizon_days:
                 raise ValueError(f"Некорректный горизонт '{horizon}' для определения шагов.")
            steps = horizon_days[horizon]

        forecast = self.forecast(best_model_name, steps=steps)
        return forecast

    def find_best_models_for_all_items(self, target_col='cnt', horizon='week',
                                       top_n=5, verbose=False):
        """
        Находит лучшие классические модели для каждого товара в магазине.

        Args:
            target_col (str, optional): Целевой столбец. По умолчанию 'cnt'.
            horizon (str, optional): Горизонт оценки. По умолчанию 'week'.
            top_n (int, optional): Количество лучших товаров для визуализации.
                                   По умолчанию 5.
            verbose (bool, optional): Подробный вывод при поиске моделей для каждого товара. По умолчанию False.

        Returns:
            pd.DataFrame | None: DataFrame с результатами или None при ошибке.
        """
        if not self.items:
            print("Ошибка: Список товаров пуст. Загрузите данные.")
            return None
        if not self.items_data:
             print("Ошибка: Данные по товарам (items_data) отсутствуют.")
             return None

        print(f"\n=== Поиск лучших моделей для всех товаров ({target_col}, {horizon}) ===")
        results = {}
        num_items = len(self.items)

        for i, item_id in enumerate(self.items):
            print(f"\nОбработка товара {item_id} ({i + 1}/{num_items})")
            try:
                best_model_name, metrics = self.find_best_classical_model(
                    target_col,
                    horizon=horizon,
                    verbose=verbose,
                    item_id=item_id
                )
                if metrics is None:
                     print(f"Предупреждение: Метрики для лучшей модели товара {item_id} не получены. Пропуск.")
                     continue

                results[item_id] = {
                    'best_model': best_model_name,
                    'MAE': metrics.get('MAE', np.nan),
                    'RMSE': metrics.get('RMSE', np.nan),
                    'MAPE': metrics.get('MAPE', np.nan),
                    'R2': metrics.get('R2', np.nan)
                }
            except ValueError as e:
                 print(f"Ошибка при обработке товара {item_id}: {e}")
                 continue
            except Exception as e:
                 print(f"Непредвиденная ошибка при обработке товара {item_id}: {e}")
                 continue

        if results:
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df.index.name = 'item_id'
            results_df.reset_index(inplace=True)
            results_df = results_df.sort_values('MAE')

            print("\n=== Итоговые результаты для всех товаров ===")
            print(f"Всего успешно обработано товаров: {len(results_df)}")
            print("\nТоп-5 товаров с лучшими моделями:")
            print(results_df[['item_id', 'best_model', 'MAE', 'RMSE', 'MAPE', 'R2']]
                  .head(top_n).round(2).to_string(index=False))

            print(f"\n=== Визуализация прогнозов для топ-{top_n} товаров ===")
            for i in range(min(top_n, len(results_df))):
                item_id_vis = results_df.iloc[i]['item_id']
                best_model_vis = results_df.iloc[i]['best_model']
                print(f"\nТовар {item_id_vis}, модель {best_model_vis}")
                try:
                    if best_model_vis in self.forecasts:
                         self.visualize_forecast(best_model_vis, horizon=horizon)
                    else:
                         print(f"Предупреждение: Прогноз для модели {best_model_vis} отсутствует.")
                except Exception as e:
                    print(f"Ошибка при визуализации для товара {item_id_vis}: {e}")

            return results_df
        else:
            print("Не удалось успешно обработать ни один товар.")
            return None