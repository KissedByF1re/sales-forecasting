import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, CatBoostError

# Подавляем предупреждения
warnings.filterwarnings("ignore")


class BestForecaster:
    """
    Класс для прогнозирования временных рядов продаж с использованием CatBoost.

    Основная логика:
    1. Пытается загрузить предобученную модель для товара/агрегата.
    2. Если модель не найдена:
        - Разделяет данные на обучающую и тестовую выборки.
        - Обучает новую модель CatBoost.
        - Оценивает модель на тестовой выборке.
        - Сохраняет обученную модель.
    3. Генерирует прогнозы на заданные горизонты ('week', 'month', 'quarter').
    4. Визуализирует результаты прогнозирования (история, факт, прогноз).
    5. Собирает и выводит метрики оценки (если проводилось обучение/оценка).

    Атрибуты:
        models_dir (str): Директория для загрузки/сохранения моделей.
        store_id (str): Идентификатор магазина для обработки.
        items (list): Список ID товаров, доступных для обработки.
        items_data (dict): Словарь {item_id: pd.DataFrame} с предобработанными
                           дневными данными по каждому товару.
        data (pd.DataFrame): Предобработанные агрегированные дневные данные
                             по всему магазину.
        _full_dates_data (pd.DataFrame): Исходные данные календаря (для событий).
        data_split (bool): Флаг, указывающий, были ли данные разделены на
                           обучающую/тестовую выборки в текущей сессии.
        split_results (dict): Результаты разделения {item_id: (train_df, test_df)}.
        train_data (pd.DataFrame): Агрегированная обучающая выборка.
        test_data (pd.DataFrame): Агрегированная тестовая выборка.
        test_split_size (float): Доля данных для тестовой выборки.
        models (dict): Модели, обученные в текущей сессии {model_key: model}.
        loaded_models (dict): Кэш загруженных моделей {model_key: model}.
        forecasts (dict): Сгенерированные прогнозы {forecast_key: pd.DataFrame}.
        metrics (dict): Рассчитанные метрики {(forecast_key, horizon): dict}.
        features (list): Список названий признаков, используемых моделью.
        target (str): Название целевой переменной (обычно 'cnt').
        cat_features (list): Список названий категориальных признаков.
    """

    def __init__(self, models_dir='models'):
        """
        Инициализирует класс BestForecaster.

        Args:
            models_dir (str): Путь к директории, где хранятся или будут
                              сохранены файлы моделей (.cbm). Директория
                              будет создана, если не существует.
        """
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.store_id = None
        self.items = []
        self.items_data = {}
        self.data = None
        self._full_dates_data = None

        self.data_split = False
        self.split_results = {}
        self.train_data = None
        self.test_data = None
        self.test_split_size = 0.2

        self.models = {}
        self.loaded_models = {}
        self.forecasts = {}
        self.metrics = {}

        self.features = None
        self.target = None
        self.cat_features = None

    def load_and_preprocess_data(self, sales_path, prices_path, dates_path,
                                 store_id='STORE_2'):
        """
        Загружает и предобрабатывает данные о продажах, ценах и календаре.

        Выполняет объединение данных, фильтрацию по магазину, обработку пропусков (интерполяция, ffill/bfill),
        агрегацию по дням и создание базовых календарных признаков.
        Сбрасывает состояние предыдущих запусков (модели, прогнозы и т.д.).

        Args:
            sales_path (str): Путь к CSV файлу с данными о продажах.
            prices_path (str): Путь к CSV файлу с данными о ценах.
            dates_path (str): Путь к CSV файлу с календарем и событиями.
            store_id (str): Идентификатор магазина для фильтрации данных.

        Returns:
            dict: Словарь `items_data` с предобработанными данными по товарам.

        Raises:
            FileNotFoundError: Если один из входных файлов не найден.
        """
        self.store_id = store_id
        # Сброс состояния при загрузке новых данных
        self.items = []
        self.items_data = {}
        self.data = None
        self._full_dates_data = None
        self.data_split = False
        self.split_results = {}
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.loaded_models = {}
        self.forecasts = {}
        self.metrics = {}
        self.features = None
        self.target = None
        self.cat_features = None

        try:
            sales = pd.read_csv(sales_path)
            prices = pd.read_csv(prices_path)
            dates = pd.read_csv(dates_path)
        except FileNotFoundError as e:
            print(f"Ошибка загрузки файла данных: {e}")
            raise

        self._full_dates_data = dates.copy()
        self._full_dates_data['date'] = pd.to_datetime(
            self._full_dates_data['date']
        )

        sales = sales[sales['store_id'] == store_id].copy()
        print(f"Данные отфильтрованы по магазину: {store_id}")
        if sales.empty:
            print(f"Предупреждение: Нет данных о продажах для магазина {store_id}.")

        self.items = sales['item_id'].unique()
        print(f"Найдено {len(self.items)} уникальных товаров в магазине {store_id}")

        dates['date'] = pd.to_datetime(dates['date'])

        cashback_col_name = f'CASHBACK_{store_id}'
        date_columns = [
            'date_id', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
        ]
        if cashback_col_name in dates.columns:
            date_columns.append(cashback_col_name)

        merged_data = pd.merge(
            sales, dates[date_columns], on='date_id', how='left'
        )
        merged_data = pd.merge(
            merged_data,
            prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )

        merged_data.sort_values(by=['item_id', 'date'], inplace=True)
        merged_data['sell_price'] = merged_data.groupby('item_id')[
            'sell_price'
        ].ffill()
        merged_data['sell_price'] = merged_data.groupby('item_id')[
            'sell_price'
        ].bfill()

        all_dates = pd.date_range(
            start=dates['date'].min(), end=dates['date'].max(), freq='D'
        )

        self.items_data = {}
        valid_items_post_processing = []
        for item_id in self.items:
            item_data = merged_data[merged_data['item_id'] == item_id].copy()
            if item_data.empty:
                continue

            daily_data = item_data.groupby('date').agg(
                cnt=('cnt', 'sum'), sell_price=('sell_price', 'mean')
            ).reset_index()

            item_data['event_name_1'] = item_data['event_name_1'].fillna('no_event')
            item_data['event_type_1'] = item_data['event_type_1'].fillna('no_event')
            item_data['event_name_2'] = item_data['event_name_2'].fillna('no_event')
            item_data['event_type_2'] = item_data['event_type_2'].fillna('no_event')

            events_daily = item_data.groupby('date')[[
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
            ]].first().reset_index()
            daily_data = pd.merge(
                daily_data, events_daily, on='date', how='left'
            )

            if cashback_col_name in item_data.columns:
                cashback_daily = item_data.groupby('date')[
                    cashback_col_name
                ].first().reset_index()
                daily_data = pd.merge(
                    daily_data, cashback_daily, on='date', how='left'
                )
                if cashback_col_name in daily_data.columns:
                    daily_data[cashback_col_name] = daily_data[
                        cashback_col_name
                    ].fillna(0)

            daily_data.set_index('date', inplace=True)
            daily_data = daily_data.reindex(all_dates)

            for col in ['cnt', 'sell_price']:
                if col in daily_data.columns:
                    daily_data[col] = daily_data[col].interpolate(
                        method='linear', limit_direction='both'
                    )
                    daily_data[col] = daily_data[col].fillna(method='bfill')
                    daily_data[col] = daily_data[col].fillna(method='ffill')
                    daily_data[col] = daily_data[col].fillna(0)

            for col in ['event_name_1', 'event_type_1',
                        'event_name_2', 'event_type_2']:
                if col in daily_data.columns:
                    daily_data[col] = daily_data[col].fillna('no_event')
            if cashback_col_name in daily_data.columns:
                daily_data[cashback_col_name] = daily_data[
                    cashback_col_name
                ].fillna(0)

            daily_data['dayofweek'] = daily_data.index.dayofweek
            daily_data['month'] = daily_data.index.month
            daily_data['year'] = daily_data.index.year
            daily_data['day'] = daily_data.index.day
            daily_data['is_weekend'] = (
                daily_data.index.dayofweek >= 5
            ).astype(int)

            if daily_data.empty or daily_data['cnt'].isnull().all():
                continue

            self.items_data[item_id] = daily_data
            valid_items_post_processing.append(item_id)

        self.items = valid_items_post_processing
        if not self.items:
            print("Предупреждение: Не найдено валидных товаров после предобработки.")

        # Агрегированные данные по магазину
        if not merged_data.empty:
            store_daily_data = merged_data.groupby('date').agg(
                cnt=('cnt', 'sum')
            ).reset_index()

            merged_data['event_name_1'] = merged_data['event_name_1'].fillna('no_event')
            merged_data['event_type_1'] = merged_data['event_type_1'].fillna('no_event')
            merged_data['event_name_2'] = merged_data['event_name_2'].fillna('no_event')
            merged_data['event_type_2'] = merged_data['event_type_2'].fillna('no_event')

            events_daily = merged_data.groupby('date')[[
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
            ]].first().reset_index()
            store_daily_data = pd.merge(
                store_daily_data, events_daily, on='date', how='left'
            )

            if cashback_col_name in merged_data.columns:
                cashback_daily = merged_data.groupby('date')[
                    cashback_col_name
                ].first().reset_index()
                store_daily_data = pd.merge(
                    store_daily_data, cashback_daily, on='date', how='left'
                )
                if cashback_col_name in store_daily_data.columns:
                    store_daily_data[cashback_col_name] = store_daily_data[
                        cashback_col_name
                    ].fillna(0)

            store_daily_data.set_index('date', inplace=True)
            store_daily_data = store_daily_data.reindex(all_dates)

            if 'cnt' in store_daily_data.columns:
                store_daily_data['cnt'] = store_daily_data['cnt'].interpolate(
                    method='linear', limit_direction='both'
                )
                store_daily_data['cnt'] = store_daily_data['cnt'].fillna(method='bfill')
                store_daily_data['cnt'] = store_daily_data['cnt'].fillna(method='ffill')
                store_daily_data['cnt'] = store_daily_data['cnt'].fillna(0)

            for col in ['event_name_1', 'event_type_1',
                        'event_name_2', 'event_type_2']:
                if col in store_daily_data.columns:
                    store_daily_data[col] = store_daily_data[col].fillna('no_event')
            if cashback_col_name in store_daily_data.columns:
                store_daily_data[cashback_col_name] = store_daily_data[
                    cashback_col_name
                ].fillna(0)

            store_daily_data['dayofweek'] = store_daily_data.index.dayofweek
            store_daily_data['month'] = store_daily_data.index.month
            store_daily_data['year'] = store_daily_data.index.year
            store_daily_data['day'] = store_daily_data.index.day
            store_daily_data['is_weekend'] = (
                store_daily_data.index.dayofweek >= 5
            ).astype(int)
            self.data = store_daily_data
        else:
            self.data = pd.DataFrame()

        print(f"Данные загружены и предобработаны для {len(self.items)} товаров.")
        if not all_dates.empty:
            print(f"Период данных: {all_dates.min().date()} по {all_dates.max().date()}")

        return self.items_data

    def add_time_features(self, data, item_id=None):
        """
        Добавляет временные признаки (лаги, скользящие окна, календарные) в DataFrame.

        Предполагается, что входной DataFrame `data` имеет DatetimeIndex и
        столбцы 'cnt' и опционально 'sell_price', 'dayofweek', 'month', 'year',
        'is_weekend', 'event_type_1', 'event_type_2'.

        Args:
            data (pd.DataFrame): Входной DataFrame для добавления признаков.
            item_id (str, optional): ID товара для добавления как признак.

        Returns:
            pd.DataFrame: DataFrame с добавленными признаками, готовый для
                          обучения или прогнозирования (после удаления NaN).
                          Возвращает пустой DataFrame, если входные данные пусты.
        """
        df = data.copy()
        if df.empty or 'cnt' not in df.columns:
            return pd.DataFrame()

        lags_cnt = [1, 2, 3, 7, 14, 21, 28, 35]
        for lag in lags_cnt:
            df[f'cnt_lag_{lag}'] = df['cnt'].shift(lag)

        windows_cnt = [7, 14, 21, 28]
        for window in windows_cnt:
            df[f'cnt_rolling_mean_{window}'] = df['cnt'].rolling(
                window=window
            ).mean()
            df[f'cnt_rolling_std_{window}'] = df['cnt'].rolling(
                window=window
            ).std()

        if ('cnt_rolling_mean_7' in df.columns and
                'cnt_rolling_mean_28' in df.columns):
            df['cnt_rolling_mean_7_28_diff'] = (
                df['cnt_rolling_mean_7'] - df['cnt_rolling_mean_28']
            )

        if 'sell_price' in df.columns:
            df['sell_price_diff_1'] = df['sell_price'].diff(1)
            for lag in [1, 7]:
                df[f'sell_price_lag_{lag}'] = df['sell_price'].shift(lag)

        if 'dayofweek' in df.columns:
            df['dayofweek'] = df['dayofweek'].astype(int)
        if 'month' in df.columns:
            df['month'] = df['month'].astype(int)
        if 'year' in df.columns:
            df['year'] = df['year'].astype(int)
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        df['weekofyear'] = df.index.strftime('%U').astype(int)
        if 'dayofweek' in df.columns:
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        if 'is_weekend' in df.columns and 'month' in df.columns:
            df['weekend_month'] = df['is_weekend'] * df['month']

        if item_id is not None:
            df['item_id'] = item_id

        cols_to_drop = ['event_name_1', 'event_name_2']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)

        if 'sell_price_diff_1' in df.columns:
            df['sell_price_diff_1'] = df['sell_price_diff_1'].fillna(0)
        if 'cnt_rolling_mean_7_28_diff' in df.columns:
            df['cnt_rolling_mean_7_28_diff'] = df[
                'cnt_rolling_mean_7_28_diff'
            ].fillna(0)

        df = df.dropna()

        # Определяем списки признаков при первом успешном вызове
        if self.features is None and not df.empty:
            exclude_columns = ['cnt']
            if 'sell_price' in df.columns:
                exclude_columns.append('sell_price')
            self.features = [
                col for col in df.columns if col not in exclude_columns
            ]

            cat_features_potential = [
                'dayofweek', 'month', 'year', 'quarter', 'weekofyear',
                'is_weekend', 'event_type_1', 'event_type_2'
            ]
            if 'item_id' in self.features:
                cat_features_potential.append('item_id')
            self.cat_features = [
                f for f in cat_features_potential if f in self.features
            ]

        return df

    def _ensure_data_split(self, test_size=0.2):
        """
        Разделяет данные на обучающую и тестовую выборки, если это еще не сделано.

        Выполняет разделение для данных по каждому товару (`items_data`) и
        для агрегированных данных (`data`). Обновляет атрибуты `data_split`,
        `split_results`, `train_data`, `test_data` и `items`.

        Args:
            test_size (float): Доля данных для тестовой выборки.

        Returns:
            bool: True, если данные успешно разделены или уже были разделены,
                  False в случае ошибки (например, нет данных).
        """
        if self.data_split:
            return True

        if not self.items_data and (self.data is None or self.data.empty):
            print("Ошибка: Данные не загружены для разделения.")
            return False

        self.test_split_size = test_size
        self.split_results = {}
        valid_items_for_split = []

        for item_id, item_data in self.items_data.items():
            data_with_features = self.add_time_features(item_data, item_id)
            if data_with_features.empty or len(data_with_features) < 2:
                continue

            split_idx = int(len(data_with_features) * (1 - test_size))
            if split_idx <= 0 or split_idx >= len(data_with_features):
                continue

            train_data_item = data_with_features.iloc[:split_idx].copy()
            test_data_item = data_with_features.iloc[split_idx:].copy()
            self.split_results[item_id] = (train_data_item, test_data_item)
            valid_items_for_split.append(item_id)

        if self.data is not None and not self.data.empty:
            agg_data_with_features = self.add_time_features(self.data)
            if not agg_data_with_features.empty and len(agg_data_with_features) >= 2:
                split_idx_agg = int(len(agg_data_with_features) * (1 - test_size))
                if 0 < split_idx_agg < len(agg_data_with_features):
                    self.train_data = agg_data_with_features.iloc[:split_idx_agg].copy()
                    self.test_data = agg_data_with_features.iloc[split_idx_agg:].copy()
                else:
                    print("Предупреждение: Некорректный индекс разделения для агрегированных данных.")
                    self.train_data, self.test_data = None, None
            else:
                print("Предупреждение: Недостаточно агрегированных данных для разделения.")
                self.train_data, self.test_data = None, None
        else:
            self.train_data, self.test_data = None, None

        original_items = self.items
        self.items = valid_items_for_split
        if len(self.items) < len(original_items):
            print(f"Предупреждение: {len(original_items) - len(self.items)} "
                  f"товаров исключены из-за недостатка данных для разделения.")

        self.data_split = True
        print(f"Разделение данных выполнено (test_size={test_size}). "
              f"Обрабатывается {len(self.items)} товаров.")
        return True

    def _prepare_features_targets(self, target_col='cnt', item_id=None):
        """
        Подготавливает наборы признаков (X) и целевой переменной (y) для обучения/оценки.

        Использует данные из `split_results` (для товара) или `train_data`/
        `test_data` (для агрегата). Списки признаков (`self.features`,
        `self.cat_features`) должны быть определены ранее вызовом
        `add_time_features`.

        Args:
            target_col (str): Название целевой переменной.
            item_id (str, optional): ID товара. Если None, используются
                                     агрегированные данные.

        Returns:
            tuple: Кортеж (X_train, y_train, X_test, y_test, cat_features_list)
                   или (None, None, None, None, None) в случае ошибки.
        """
        if not self.data_split:
            print("Ошибка: Данные не разделены для подготовки признаков.")
            return None, None, None, None, None

        train_data_source, test_data_source = None, None
        if item_id is None:
            train_data_source, test_data_source = self.train_data, self.test_data
        elif item_id in self.split_results:
            train_data_source, test_data_source = self.split_results[item_id]

        if (train_data_source is None or test_data_source is None or
                train_data_source.empty or test_data_source.empty):
            item_str = 'агрегированных данных' if item_id is None else item_id
            print(f"Предупреждение: Отсутствуют данные train/test для {item_str}.")
            return None, None, None, None, None

        if self.features is None or self.cat_features is None:
            print("Ошибка: Списки признаков (features/cat_features) не определены.")
            return None, None, None, None, None

        missing_features = [
            f for f in self.features if f not in train_data_source.columns
        ]
        if missing_features:
            print(f"Ошибка: Отсутствуют признаки {missing_features} в данных "
                  f"train для {item_id}.")
            return None, None, None, None, None

        current_cat_features = [
            f for f in self.cat_features if f in self.features
        ]

        X_train = train_data_source[self.features]
        y_train = train_data_source[target_col]
        X_test = test_data_source[self.features]
        y_test = test_data_source[target_col]

        return X_train, y_train, X_test, y_test, current_cat_features

    def _train_catboost_model(self, target_col='cnt', params=None, item_id=None):
        """
        Обучает модель CatBoostRegressor и сохраняет ее в файл и атрибуты класса.

        Перед обучением гарантирует, что данные разделены на train/test.

        Args:
            target_col (str): Название целевой переменной.
            params (dict, optional): Параметры для CatBoostRegressor.
                                     Если None, используются значения по умолчанию.
            item_id (str, optional): ID товара. Если None, обучается
                                     агрегированная модель.

        Returns:
            catboost.CatBoostRegressor or None: Обученная модель или None в
                                                случае ошибки.
        """
        if not self._ensure_data_split():
            return None

        X_train, y_train, _, _, current_cat_features = \
            self._prepare_features_targets(target_col, item_id)

        if X_train is None or y_train is None or X_train.empty:
            item_str = 'агрегированной модели' if item_id is None else f'модели для {item_id}'
            print(f"Нет данных для обучения {item_str}.")
            return None

        if params is None:
            params = {
                'loss_function': 'MAE', 'iterations': 700,
                'learning_rate': 0.03, 'random_seed': 42, 'verbose': False
            }

        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"
        print(f"Обучение модели CatBoost для {target_col}{item_str}...")

        model = CatBoostRegressor(**params)
        try:
            model.fit(
                X_train, y_train, cat_features=current_cat_features, verbose=False
            )
        except Exception as e:
            print(f"Ошибка во время обучения модели{item_str}: {e}")
            return None

        model_key = f'CatBoost_{item_id}_{target_col}' if item_id else f'CatBoost_{target_col}'
        model_filename_base = f'catboost_{item_id}_{target_col}_model.cbm' if item_id else f'catboost_{target_col}_model.cbm'
        model_path = os.path.join(self.models_dir, model_filename_base)

        try:
            model.save_model(model_path)
            print(f"Модель сохранена: {model_path}")
        except Exception as e:
            print(f"Ошибка сохранения модели {model_path}: {e}")

        self.models[model_key] = model
        self.loaded_models[model_key] = model

        return model

    def _load_model(self, item_id, target_col):
        """
        Загружает сохраненную модель CatBoost из файла, используя кэш.

        Args:
            item_id (str or None): ID товара или None для агрегированной модели.
            target_col (str): Название целевой переменной.

        Returns:
            catboost.CatBoostRegressor or None: Загруженная модель или None,
                                                если файл не найден или
                                                произошла ошибка загрузки.
        """
        model_key = f'CatBoost_{item_id}_{target_col}' if item_id else f'CatBoost_{target_col}'
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        model_filename_base = f'catboost_{item_id}_{target_col}_model.cbm' if item_id else f'catboost_{target_col}_model.cbm'
        model_path = os.path.join(self.models_dir, model_filename_base)

        if not os.path.exists(model_path):
            return None

        try:
            model = CatBoostRegressor()
            model.load_model(model_path)
            print(f"Модель успешно загружена: {model_path}")
            self.loaded_models[model_key] = model
            return model
        except Exception as e:
            print(f"Ошибка загрузки модели {model_path}: {e}")
            return None

    def _prepare_for_prediction(self, target_col='cnt', item_id=None):
        """
        Подготавливает все необходимое для запуска итеративного прогнозирования.

        Загружает или обучает модель, определяет базовую историю (train set
        или все данные) и дату начала прогноза.

        Args:
            target_col (str): Название целевой переменной.
            item_id (str, optional): ID товара или None для агрегата.

        Returns:
            tuple: Кортеж вида:
                   (model, base_history_with_features, start_forecast_date,
                   trained_new_model)
                   В случае ошибки возвращает (None, None, None, False) или
                   (model, None, None, trained_flag) если модель есть,
                   но историю подготовить не удалось.
        """
        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"
        model = self._load_model(item_id, target_col)
        trained_new_model = False
        base_history_with_features = None
        start_forecast_date = None
        raw_history_source = None

        if model is None:
            print(f"Предобученная модель{item_str} не найдена. Запуск обучения...")
            model = self._train_catboost_model(target_col=target_col, item_id=item_id)
            if model is None:
                print(f"Не удалось обучить модель{item_str}. Подготовка к прогнозу невозможна.")
                return None, None, None, False
            trained_new_model = True
            # Если обучили, базовая история - это train set
            if item_id is None:
                raw_history_source = self.train_data
            elif item_id in self.split_results:
                raw_history_source = self.split_results[item_id][0]
        else:
            print(f"Использование предобученной модели{item_str}.")
            # Если загрузили, базовая история - это все данные
            if item_id is None:
                raw_history_source = self.data
            elif item_id in self.items_data:
                raw_history_source = self.items_data[item_id]

        if raw_history_source is None or raw_history_source.empty:
            print(f"Ошибка: Отсутствуют исторические данные для подготовки к прогнозированию{item_str}.")
            if trained_new_model:
                model = None
            return model, None, None, trained_new_model

        # Применяем генерацию признаков к стартовой истории ОДИН РАЗ
        base_history_with_features = self.add_time_features(
            raw_history_source.copy(), item_id
        )

        if base_history_with_features is None or base_history_with_features.empty:
            print(f"Предупреждение: Нет исторических данных с признаками для старта прогноза{item_str}.")
            if trained_new_model:
                model = None
            return model, None, None, trained_new_model

        # Определяем точку старта прогноза ОДИН РАЗ
        last_hist_date = base_history_with_features.index[-1]
        start_forecast_date = last_hist_date + pd.Timedelta(days=1)
        print(f"Определена точка старта прогноза: {start_forecast_date.date()}")

        return model, base_history_with_features, start_forecast_date, trained_new_model

    def predict(self, model, base_history_with_features, start_forecast_date,
                target_col='cnt', horizon='week', item_id=None):
        """
        Генерирует прогнозы итеративно на заданный горизонт.

        Использует переданную модель, базовую историю и дату старта. Не выполняет
        загрузку/обучение модели или определение истории.

        Args:
            model (CatBoostRegressor): Обученная или загруженная модель.
            base_history_with_features (pd.DataFrame): История с признаками,
                                                       заканчивающаяся перед
                                                       `start_forecast_date`.
            start_forecast_date (pd.Timestamp): Дата, с которой начинается прогноз.
            target_col (str): Название целевой переменной.
            horizon (str): Горизонт прогнозирования ('week', 'month', 'quarter').
            item_id (str, optional): ID товара или None для агрегированных данных.

        Returns:
            pd.DataFrame or None: DataFrame с прогнозами (индекс - дата, столбец -
                                  `target_col`) или None в случае ошибки.
        """
        if model is None or base_history_with_features is None or start_forecast_date is None:
            print("Ошибка: Не переданы необходимые данные (модель/история/дата старта) в predict.")
            return None

        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"

        days_map = {'week': 7, 'month': 30, 'quarter': 90}
        if horizon not in days_map:
            print(f"Ошибка: Некорректный горизонт '{horizon}' в predict.")
            return None
        days = days_map[horizon]

        print(f"Генерация прогноза на {horizon} ({days} дней) для {target_col}{item_str} "
              f"(старт: {start_forecast_date.date()})...")

        try:
            model_features = model.feature_names_
            model_cat_feature_indices = model.get_cat_feature_indices()
            model_cat_features_names = [
                model_features[i] for i in model_cat_feature_indices
            ]
        except Exception as e:
            print(f"Предупреждение: Не удалось получить признаки из модели{item_str}: {e}. "
                  f"Используем self.features.")
            if self.features is None or self.cat_features is None:
                print("Ошибка: Списки признаков не определены. Прогноз невозможен.")
                return None
            model_features = self.features
            model_cat_features_names = self.cat_features

        forecast_dates = pd.date_range(
            start=start_forecast_date, periods=days, freq='D'
        )
        forecast_df = pd.DataFrame(index=forecast_dates)
        forecast_df[target_col] = np.nan

        future_info_lookup = self._full_dates_data.set_index('date') \
            if self._full_dates_data is not None else None
        cashback_col = f'CASHBACK_{self.store_id}'
        item_full_data = self.items_data.get(item_id, None)

        iterative_history = base_history_with_features.copy()

        lags_cnt = [1, 2, 3, 7, 14, 21, 28, 35]
        windows_cnt = [7, 14, 21, 28]
        lags_price = [1, 7]

        for current_date in forecast_dates:
            current_features_df = pd.DataFrame(index=[current_date])

            # Генерация признаков для current_date
            current_features_df['dayofweek'] = current_date.dayofweek
            current_features_df['month'] = current_date.month
            current_features_df['year'] = current_date.year
            current_features_df['day'] = current_date.day
            current_features_df['is_weekend'] = 1 if current_date.dayofweek >= 5 else 0
            current_features_df['dayofyear'] = current_date.dayofyear
            current_features_df['quarter'] = current_date.quarter
            current_features_df['weekofyear'] = int(current_date.strftime('%U'))
            current_features_df['dayofweek_sin'] = np.sin(2 * np.pi * current_features_df['dayofweek'] / 7)
            current_features_df['dayofweek_cos'] = np.cos(2 * np.pi * current_features_df['dayofweek'] / 7)
            current_features_df['month_sin'] = np.sin(2 * np.pi * current_features_df['month'] / 12)
            current_features_df['month_cos'] = np.cos(2 * np.pi * current_features_df['month'] / 12)
            current_features_df['weekend_month'] = current_features_df['is_weekend'] * current_features_df['month']

            if item_id is not None and 'item_id' in model_features:
                current_features_df['item_id'] = item_id

            event_type_1_val, event_type_2_val, cashback_value = 'no_event', 'no_event', 0
            if future_info_lookup is not None and current_date in future_info_lookup.index:
                date_info = future_info_lookup.loc[current_date]
                event_type_1_val = date_info.get('event_type_1', 'no_event')
                event_type_1_val = 'no_event' if pd.isna(event_type_1_val) else event_type_1_val
                event_type_2_val = date_info.get('event_type_2', 'no_event')
                event_type_2_val = 'no_event' if pd.isna(event_type_2_val) else event_type_2_val
                if cashback_col in date_info:
                    cashback_value = date_info[cashback_col]

            if 'event_type_1' in model_features:
                current_features_df['event_type_1'] = event_type_1_val
            if 'event_type_2' in model_features:
                current_features_df['event_type_2'] = event_type_2_val
            if cashback_col in model_features:
                current_features_df[cashback_col] = cashback_value

            current_sell_price = np.nan
            if item_full_data is not None and current_date in item_full_data.index:
                current_sell_price = item_full_data.loc[current_date, 'sell_price']
            if pd.isna(current_sell_price) and 'sell_price' in iterative_history.columns and not iterative_history.empty:
                current_sell_price = iterative_history['sell_price'].iloc[-1]
            current_sell_price = 0 if pd.isna(current_sell_price) else current_sell_price

            if 'sell_price_diff_1' in model_features:
                prev_date = current_date - pd.Timedelta(days=1)
                prev_sell_price = 0
                if prev_date in iterative_history.index and 'sell_price' in iterative_history:
                    prev_sell_price = iterative_history.loc[prev_date, 'sell_price']
                    prev_sell_price = 0 if pd.isna(prev_sell_price) else prev_sell_price
                current_features_df['sell_price_diff_1'] = current_sell_price - prev_sell_price

            if 'sell_price' in iterative_history:
                temp_series_price = iterative_history['sell_price'].fillna(0)
                for lag in lags_price:
                    feature_name = f'sell_price_lag_{lag}'
                    if feature_name in model_features:
                        current_features_df[feature_name] = temp_series_price.iloc[-lag] if len(temp_series_price) >= lag else 0

            temp_series_cnt = iterative_history[target_col].fillna(0)
            for lag in lags_cnt:
                feature_name = f'cnt_lag_{lag}'
                if feature_name in model_features:
                    current_features_df[feature_name] = temp_series_cnt.iloc[-lag] if len(temp_series_cnt) >= lag else 0

            rolling_means = {}
            for window in windows_cnt:
                feature_name_mean = f'cnt_rolling_mean_{window}'
                feature_name_std = f'cnt_rolling_std_{window}'
                hist_window = temp_series_cnt.iloc[-window:]
                current_mean = hist_window.mean() if len(hist_window) > 0 else 0
                current_std = hist_window.std() if len(hist_window) > 0 else 0
                if feature_name_mean in model_features:
                    current_features_df[feature_name_mean] = pd.Series(current_mean).fillna(0).iloc[0]
                    rolling_means[window] = current_features_df[feature_name_mean].iloc[0]
                if feature_name_std in model_features:
                    current_features_df[feature_name_std] = pd.Series(current_std).fillna(0).iloc[0]

            if 'cnt_rolling_mean_7_28_diff' in model_features:
                mean7, mean28 = rolling_means.get(7, 0), rolling_means.get(28, 0)
                current_features_df['cnt_rolling_mean_7_28_diff'] = mean7 - mean28

            # Подготовка данных для модели
            final_features_for_pred = pd.DataFrame(
                columns=model_features, index=[current_date]
            )
            for feature in model_features:
                if feature in current_features_df.columns:
                    final_features_for_pred[feature] = current_features_df[feature]
                else:
                    is_categorical = feature in model_cat_features_names
                    final_features_for_pred[feature] = 'no_event' if is_categorical else 0

            num_cols = final_features_for_pred.select_dtypes(include=np.number).columns
            final_features_for_pred[num_cols] = final_features_for_pred[num_cols].fillna(0)
            cat_cols_in_final = [
                col for col in model_cat_features_names
                if col in final_features_for_pred.columns
            ]
            final_features_for_pred[cat_cols_in_final] = final_features_for_pred[
                cat_cols_in_final
            ].fillna('no_event').astype(str)

            # Предсказание
            try:
                prediction = model.predict(final_features_for_pred[model_features])[0]
                prediction = max(0, prediction)
            except Exception as e:
                print(f"Ошибка предсказания на {current_date} для {item_id}: {e}")
                prediction = 0

            forecast_df.loc[current_date, target_col] = prediction

            # Обновление истории
            new_history_row_data = {}
            for col in iterative_history.columns:
                if col == target_col:
                    new_history_row_data[col] = prediction
                elif col == 'sell_price':
                    new_history_row_data[col] = current_sell_price
                elif col in final_features_for_pred.columns:
                    new_history_row_data[col] = final_features_for_pred.iloc[0][col]
                elif col in current_features_df.columns:
                    new_history_row_data[col] = current_features_df.iloc[0][col]
                else:
                    dtype = iterative_history[col].dtype
                    new_history_row_data[col] = 0 if pd.api.types.is_numeric_dtype(dtype) else 'no_event'

            new_history_row_df = pd.DataFrame(
                [new_history_row_data], index=[current_date]
            )
            new_history_row_df = new_history_row_df.reindex(
                columns=iterative_history.columns
            )
            for col in new_history_row_df.columns:
                if new_history_row_df[col].isnull().any():
                    dtype = iterative_history[col].dtype
                    fill_value = 0 if pd.api.types.is_numeric_dtype(dtype) else 'no_event'
                    new_history_row_df[col] = new_history_row_df[col].fillna(fill_value)

            iterative_history = pd.concat([iterative_history, new_history_row_df])

        forecast_key = f'CatBoost_{item_id}_{target_col}_{horizon}' if item_id else f'CatBoost_{target_col}_{horizon}'
        self.forecasts[forecast_key] = forecast_df

        return forecast_df

    def evaluate(self, target_col='cnt', horizon='week', item_id=None):
        """
        Оценивает качество прогноза на тестовой выборке.

        Сравнивает сохраненный прогноз (`self.forecasts`) с соответствующей
        частью тестовой выборки (`self.test_data` или `self.split_results`).
        Расчет возможен только если данные были разделены (`self.data_split`).

        Args:
            target_col (str): Название целевой переменной.
            horizon (str): Горизонт прогноза ('week', 'month', 'quarter'),
                           для которого была сгенерирована и сохранена модель.
            item_id (str, optional): ID товара или None для агрегированных данных.

        Returns:
            tuple: Кортеж метрик (mae, rmse, mape, r2). Возвращает (np.nan, ...)
                   если оценка невозможна (нет прогноза, нет тестовых данных,
                   данные не разделены).
        """
        forecast_key = f'CatBoost_{item_id}_{target_col}_{horizon}' if item_id else f'CatBoost_{target_col}_{horizon}'

        if forecast_key not in self.forecasts:
            print(f"Прогноз {forecast_key} не найден для оценки.")
            return np.nan, np.nan, np.nan, np.nan

        if not self.data_split:
            print(f"Данные не были разделены, оценка для {forecast_key} невозможна.")
            return np.nan, np.nan, np.nan, np.nan

        test_data_source = None
        if item_id is None:
            test_data_source = self.test_data
        elif item_id in self.split_results:
            _, test_data_source = self.split_results[item_id]

        if test_data_source is None or test_data_source.empty:
            print(f"Тестовые данные для оценки {forecast_key} не найдены.")
            return np.nan, np.nan, np.nan, np.nan

        forecast = self.forecasts[forecast_key]
        common_index = test_data_source.index.intersection(forecast.index)

        if common_index.empty:
            print(f"Нет пересекающихся дат между прогнозом и фактом для оценки {forecast_key}.")
            return np.nan, np.nan, np.nan, np.nan

        actual = test_data_source.loc[common_index, target_col]
        forecast_aligned = forecast.loc[common_index, target_col]

        if len(actual) == 0:
            print(f"Нет данных для оценки {forecast_key} после выравнивания.")
            return np.nan, np.nan, np.nan, np.nan

        mae = mean_absolute_error(actual, forecast_aligned)
        rmse = np.sqrt(mean_squared_error(actual, forecast_aligned))
        r2 = r2_score(actual, forecast_aligned)
        mask = actual != 0
        mape = np.mean(
            np.abs((actual[mask] - forecast_aligned[mask]) / actual[mask])
        ) * 100 if np.sum(mask) > 0 else np.nan

        self.metrics[(forecast_key, horizon)] = {
            'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2
        }

        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"
        print(f"Метрики оценки (горизонт {horizon}){item_str} [по тестовой выборке]:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%" if not np.isnan(mape) else "  MAPE: N/A")
        print(f"  R2:   {r2:.2f}")

        return mae, rmse, mape, r2

    def visualize_forecast(self, target_col='cnt', horizon='week', item_id=None):
        """
        Визуализирует прогноз, фактические данные и недавнюю историю.

        Отображает хвост исторических данных, предшествующих прогнозу,
        фактические данные на период прогноза (если доступны) и сам прогноз.

        Args:
            target_col (str): Название целевой переменной.
            horizon (str): Горизонт прогноза ('week', 'month', 'quarter') для
                           визуализации.
            item_id (str, optional): ID товара или None для агрегированных данных.

        Returns:
            tuple: Кортеж (history_plot_data, actual_plot_data, forecast_plot_data)
                   содержащий данные, отображенные на графике (pd.Series),
                   или (None, None, None), если визуализация не удалась.
        """
        forecast_key = f'CatBoost_{item_id}_{target_col}_{horizon}' if item_id \
            else f'CatBoost_{target_col}_{horizon}'

        if forecast_key not in self.forecasts:
            print(f"Прогноз {forecast_key} не найден для визуализации.")
            return None, None, None

        forecast_df = self.forecasts[forecast_key]
        if forecast_df.empty:
            print(f"Прогноз {forecast_key} пуст, визуализация невозможна.")
            return None, None, None

        history_plot_data = None
        actual_plot_data = None

        days_map = {'week': 7, 'month': 30, 'quarter': 90}
        days = days_map.get(horizon, 7)
        valid_forecast_index = forecast_df.index[:days]
        if not valid_forecast_index.empty:
            forecast_plot_data = forecast_df.loc[valid_forecast_index, target_col]
        else:
            forecast_plot_data = pd.Series(dtype=float)

        if forecast_plot_data.empty:
            print(f"Нет данных прогноза в пределах горизонта {horizon} для визуализации.")

        full_data_source = None
        if item_id is None:
            full_data_source = self.data
        elif item_id in self.items_data:
            full_data_source = self.items_data[item_id]

        if full_data_source is None or full_data_source.empty:
            item_str = item_id if item_id else 'агрег.'
            print(f"Предупреждение: Отсутствуют полные исторические данные для {item_str} для визуализации.")
        else:
            forecast_start_date = forecast_plot_data.index.min() if not forecast_plot_data.empty else None
            if forecast_start_date:
                history_before_forecast = full_data_source[
                    full_data_source.index < forecast_start_date
                ]
                if not history_before_forecast.empty:
                    history_plot_data = history_before_forecast[target_col].iloc[-30:]
                else:
                    print(f"Предупреждение: Не найдено исторических данных перед {forecast_start_date.date()} для показа хвоста.")

            actual_index_intersect = full_data_source.index.intersection(
                forecast_plot_data.index
            )
            if not actual_index_intersect.empty:
                actual_plot_data = full_data_source.loc[
                    actual_index_intersect, target_col
                ]

        plt.figure(figsize=(14, 7))
        plot_title = f'Прогноз {target_col.capitalize()} vs Факт - Горизонт {horizon.capitalize()}'
        item_str = f" (Товар: {item_id})" if item_id else " (Агрегированные)"
        plt.title(plot_title + item_str)

        if history_plot_data is not None and not history_plot_data.empty:
            plt.plot(history_plot_data.index, history_plot_data,
                     label='История (предшеств.)', color='blue', marker='.', linestyle='-')

        if actual_plot_data is not None and not actual_plot_data.empty:
            plt.plot(actual_plot_data.index, actual_plot_data,
                     label='Факт', color='green', marker='.', linestyle='-')
        elif self.data_split:
            test_data_source = None
            if item_id is None:
                test_data_source = self.test_data
            elif item_id in self.split_results:
                _, test_data_source = self.split_results[item_id]

            if test_data_source is not None and not test_data_source.empty:
                test_intersect = test_data_source.index.intersection(
                    forecast_plot_data.index
                )
                if not test_intersect.empty:
                    test_actual_plot = test_data_source.loc[test_intersect, target_col]
                    plt.plot(test_actual_plot.index, test_actual_plot,
                             label='Факт (Test Set)', color='limegreen', marker='.', linestyle=':')

        if not forecast_plot_data.empty:
            plt.plot(forecast_plot_data.index, forecast_plot_data,
                     label=f'Прогноз ({horizon})', color='red', marker='.', linestyle='--')

        plt.xlabel('Дата')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return history_plot_data, actual_plot_data, forecast_plot_data

    def predict_item(self, target_col='cnt', item_id=None, horizons=None):
        """
        Выполняет полный цикл обработки для одного товара или агрегированных данных.

        Включает подготовку (загрузка/обучение модели, определение истории
        и старта), генерацию прогнозов на заданные горизонты, оценку
        (если применимо) и визуализацию.

        Args:
            target_col (str): Название целевой переменной.
            item_id (str, optional): ID товара или None для агрегированных данных.
            horizons (list, optional): Список горизонтов прогнозирования
                                       (например, ['week', 'month']). Если None,
                                       используется ['week', 'month', 'quarter'].

        Returns:
            tuple: Кортеж (item_forecasts, item_metrics), где:
                   - item_forecasts (dict): Словарь {horizon: pd.DataFrame}
                     с прогнозами.
                   - item_metrics (dict): Словарь {horizon: dict} с метриками
                     оценки (может содержать NaN, если оценка не проводилась).
        """
        if horizons is None:
            horizons = ['week', 'month', 'quarter']

        item_forecasts = {}
        item_metrics = {}
        item_str = f" товара {item_id}" if item_id else " агрегированных данных"
        print(f"\n--- Обработка {item_str} ---")

        model, base_history, start_date, was_trained = \
            self._prepare_for_prediction(target_col, item_id)

        if model is None or base_history is None or start_date is None:
            print(f"Не удалось подготовиться к прогнозированию для{item_str}. Пропуск.")
            for horizon in horizons:
                item_metrics[horizon] = {
                    'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan
                }
            return {}, item_metrics

        for horizon in horizons:
            forecast_df = self.predict(
                model, base_history, start_date,
                target_col, horizon, item_id
            )

            if forecast_df is not None:
                item_forecasts[horizon] = forecast_df
                if was_trained:
                    mae, rmse, mape, r2 = self.evaluate(
                        target_col, horizon, item_id=item_id
                    )
                    item_metrics[horizon] = {
                        'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2
                    }
                self.visualize_forecast(target_col, horizon, item_id=item_id)
            else:
                print(f"Прогноз на {horizon} для{item_str} не сгенерирован.")
                item_metrics[horizon] = {
                    'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan
                }

        if not was_trained and self.data_split:
            print(f"\nОценка загруженной модели{item_str} по тестовой выборке:")
            for horizon in horizons:
                if horizon in item_forecasts:
                    mae, rmse, mape, r2 = self.evaluate(
                        target_col, horizon, item_id=item_id
                    )
                    if horizon not in item_metrics or pd.isna(
                            item_metrics.get(horizon, {}).get('MAE', np.nan)):
                        item_metrics[horizon] = {
                            'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2
                        }

        return item_forecasts, item_metrics

    def predict_all_items(self, target_col='cnt', items_limit=None, horizons=None):
        """
        Запускает цикл прогнозирования для всех (или ограниченного числа) валидных товаров магазина.

        Для каждого товара выполняется `predict_item`. Результаты (прогнозы
        и метрики) агрегируются. Выводится итоговая сводка по метрикам и
        информация об агрегированных прогнозах.

        Args:
            target_col (str): Название целевой переменной.
            items_limit (int, optional): Максимальное количество товаров для
                                         обработки. Если None, обрабатываются все.
            horizons (list, optional): Список горизонтов прогнозирования.
                                       Если None, используется
                                       ['week', 'month', 'quarter'].

        Returns:
            dict: Словарь {horizon: pd.DataFrame} с агрегированными прогнозами
                  для всех обработанных товаров по каждому горизонту.
        """
        if horizons is None:
            horizons = ['week', 'month', 'quarter']

        if not self.items:
            print("Ошибка: Нет валидных товаров для обработки.")
            return {horizon: pd.DataFrame() for horizon in horizons}

        self._ensure_data_split(test_size=self.test_split_size)

        items_to_process = self.items[:items_limit] if items_limit else self.items
        num_items = len(items_to_process)
        print(f"\nЗапуск обработки для {num_items} товаров...")

        all_forecasts_agg = {horizon: pd.DataFrame() for horizon in horizons}
        all_metrics_list = []

        for i, item_id in enumerate(items_to_process):
            print(f"\n>>> Товар: {item_id} ({i + 1}/{num_items}) <<<")
            item_forecasts, item_metrics = self.predict_item(
                target_col, item_id, horizons
            )

            for horizon, forecast_df in item_forecasts.items():
                if forecast_df is not None and not forecast_df.empty:
                    forecast_df_with_id = forecast_df.copy()
                    forecast_df_with_id['item_id'] = item_id
                    forecast_df_with_id.reset_index(inplace=True)
                    forecast_df_with_id.rename(
                        columns={'index': 'date'}, inplace=True
                    )
                    all_forecasts_agg[horizon] = pd.concat(
                        [all_forecasts_agg[horizon], forecast_df_with_id],
                        ignore_index=True
                    )

            for horizon, metrics_dict in item_metrics.items():
                metrics_dict_copy = metrics_dict.copy()
                metrics_dict_copy['item_id'] = item_id
                metrics_dict_copy['horizon'] = horizon
                all_metrics_list.append(metrics_dict_copy)

        print("\n=== Обработка всех товаров завершена ===")

        if all_metrics_list:
            metrics_df = pd.DataFrame(all_metrics_list)
            metric_cols = ['MAE', 'RMSE', 'MAPE', 'R2']
            existing_metric_cols = [
                col for col in metric_cols if col in metrics_df.columns
            ]

            if existing_metric_cols:
                metrics_df_clean = metrics_df.dropna(subset=existing_metric_cols)
                if not metrics_df_clean.empty:
                    print("\nСводка по метрикам (для товаров, где была оценка):")
                    summary = metrics_df_clean.groupby('horizon')[
                        existing_metric_cols
                    ].mean()
                    print(summary.round(2))
                else:
                    print("\nНет валидных данных для сводки по метрикам (оценка не проводилась или не удалась).")
            else:
                print("\nКолонки с метриками отсутствуют в результатах.")
        else:
            print("\nСписок метрик пуст (оценка не проводилась).")

        for horizon, df in all_forecasts_agg.items():
            print(f"\nИтоговый прогноз на {horizon}:")
            if not df.empty:
                print(f"  Размер: {df.shape}")
                print(f"  Товаров: {df['item_id'].nunique()}")
                print(f"  Период: {df['date'].min().date()} по {df['date'].max().date()}")
            else:
                print("  Данные отсутствуют.")

        return all_forecasts_agg