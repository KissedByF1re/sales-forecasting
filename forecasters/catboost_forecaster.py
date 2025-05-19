import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# Подавляем предупреждения для более чистого вывода
warnings.filterwarnings("ignore")

class CatBoostForecaster:
    """
    Класс для прогнозирования временных рядов продаж с использованием CatBoost.

    Обрабатывает данные по продажам, ценам и календарю для одного магазина,
    обучает отдельные модели для каждого товара (item_id) и генерирует
    прогнозы на различные горизонты. Включает предобработку, генерацию
    признаков, обучение, оценку и визуализацию.

    Атрибуты:
        store_id (str): ID магазина для обработки.
        items (list): Список ID товаров, успешно прошедших обработку.
        items_data (dict): Словарь с предобработанными дневными данными для каждого товара {item_id: pd.DataFrame}.
        data (pd.DataFrame): Предобработанные агрегированные дневные данные по всему магазину.
        _full_dates_data (pd.DataFrame): Исходные данные о датах.
        data_split (bool): Флаг, указывающий, разделены ли данные на train/test.
        split_results (dict): Словарь с результатами разделения данных для каждого товара {item_id: (train_df, test_df)}.
        train_data (pd.DataFrame): Обучающая выборка для агрегированных данных.
        test_data (pd.DataFrame): Тестовая выборка для агрегированных данных.
        features (list): Список названий признаков, используемых моделью.
        target (str): Название целевой переменной (например, 'cnt').
        cat_features (list): Список названий категориальных признаков.
        models (dict): Словарь с обученными моделями CatBoost {ключ_модели: модель}.
        forecasts (dict): Словарь с сгенерированными прогнозами {ключ_прогноза: pd.DataFrame}.
        metrics (dict): Словарь с метриками оценки {(ключ_прогноза, горизонт): dict_с_метриками}.
    """

    def __init__(self):
        """Инициализация класса CatBoostForecaster."""
        self.store_id = None
        self.items = []
        self.items_data = {}
        self.data = None
        self._full_dates_data = None
        self.data_split = False
        self.split_results = {}
        self.train_data = None
        self.test_data = None
        self.features = None
        self.target = None
        self.cat_features = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}

    def load_and_preprocess_data(self, sales_path, prices_path, dates_path,
                                 store_id='STORE_2'):
        """
        Загружает и предобрабатывает данные для указанного магазина.

        Объединяет данные о продажах, ценах и событиях, обрабатывает пропуски
        (ffill/bfill для цен, интерполяция для продаж), агрегирует данные
        по дням для каждого товара и магазина в целом, создает базовые
        календарные признаки.

        Args:
            sales_path (str): Путь к CSV файлу с продажами.
            prices_path (str): Путь к CSV файлу с ценами.
            dates_path (str): Путь к CSV файлу с календарем/событиями.
            store_id (str): ID магазина для фильтрации и обработки.

        Returns:
            dict: Словарь `items_data` с предобработанными данными по товарам.

        Raises:
            FileNotFoundError: Если один из входных файлов не найден.
        """
        self.store_id = store_id
        self.data_split = False  # Сброс статуса при загрузке новых данных
        self.split_results = {}

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

        self.items = sales['item_id'].unique()
        print(f"Найдено {len(self.items)} уникальных товаров в магазине {store_id}")

        dates['date'] = pd.to_datetime(dates['date'])

        # Определение необходимых столбцов из файла дат
        cashback_col_name = f'CASHBACK_{store_id}'
        date_columns = [
            'date_id', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
        ]
        if cashback_col_name in dates.columns:
            date_columns.append(cashback_col_name)
        else:
            print(f"Столбец кэшбека '{cashback_col_name}' не найден в данных дат.")

        # Объединение данных
        merged_data = pd.merge(sales, dates[date_columns], on='date_id',
                               how='left')
        merged_data = pd.merge(
            merged_data,
            prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )

        # Заполнение пропусков цен с использованием ffill/bfill внутри группы товаров
        merged_data.sort_values(by=['item_id', 'date'], inplace=True)
        merged_data['sell_price'] = merged_data.groupby('item_id')[
            'sell_price'].ffill()
        merged_data['sell_price'] = merged_data.groupby('item_id')[
            'sell_price'].bfill()

        # Определение полного диапазона дат
        all_dates = pd.date_range(
            start=dates['date'].min(),
            end=dates['date'].max(),
            freq='D'
        )

        # Обработка данных по каждому товару
        self.items_data = {}
        valid_items_post_processing = []
        for item_id in self.items:
            item_data = merged_data[merged_data['item_id'] == item_id].copy()

            if item_data.empty:
                continue

            # Агрегация по дням
            daily_data = item_data.groupby('date').agg(
                cnt=('cnt', 'sum'),
                sell_price=('sell_price', 'mean')
            ).reset_index()

            # Подготовка данных о событиях
            item_data['event_name_1'] = item_data['event_name_1'].fillna('no_event')
            item_data['event_type_1'] = item_data['event_type_1'].fillna('no_event')
            item_data['event_name_2'] = item_data['event_name_2'].fillna('no_event')
            item_data['event_type_2'] = item_data['event_type_2'].fillna('no_event')

            events_daily = item_data.groupby('date')[[
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
            ]].first().reset_index()

            daily_data = pd.merge(daily_data, events_daily, on='date', how='left')

            # Добавление данных о кэшбеке
            if cashback_col_name in item_data.columns:
                cashback_daily = item_data.groupby('date')[
                    cashback_col_name].first().reset_index()
                daily_data = pd.merge(daily_data, cashback_daily, on='date',
                                      how='left')
                if cashback_col_name in daily_data.columns:
                    daily_data[cashback_col_name] = daily_data[
                        cashback_col_name].fillna(0)

            # Реиндексация и интерполяция/заполнение пропусков
            daily_data.set_index('date', inplace=True)
            daily_data = daily_data.reindex(all_dates)

            for col in ['cnt', 'sell_price']:
                if col in daily_data.columns:
                    if col == 'cnt':
                        daily_data[col] = daily_data[col].replace(0, np.nan)

                    daily_data[col] = daily_data[col].interpolate(
                        method='linear', limit_direction='both'
                    )
                    daily_data[col] = daily_data[col].fillna(method='bfill')
                    daily_data[col] = daily_data[col].fillna(method='ffill')
                    daily_data[col] = daily_data[col].fillna(0)

            # Заполнение пропусков категориальных признаков после реиндексации
            for col in ['event_name_1', 'event_type_1', 'event_name_2',
                        'event_type_2']:
                if col in daily_data.columns:
                    daily_data[col] = daily_data[col].fillna('no_event')
            if cashback_col_name in daily_data.columns:
                daily_data[cashback_col_name] = daily_data[
                    cashback_col_name].fillna(0)

            # Добавление базовых календарных признаков
            daily_data['dayofweek'] = daily_data.index.dayofweek
            daily_data['month'] = daily_data.index.month
            daily_data['year'] = daily_data.index.year
            daily_data['day'] = daily_data.index.day
            daily_data['is_weekend'] = (daily_data.index.dayofweek >= 5).astype(int)

            # Проверка валидности данных после обработки
            if daily_data.empty or daily_data['cnt'].isnull().all():
                continue

            self.items_data[item_id] = daily_data
            valid_items_post_processing.append(item_id)

        # Обновление списка товаров
        self.items = valid_items_post_processing
        if not self.items:
            print("Предупреждение: Не найдено валидных товаров после предобработки.")

        # Подготовка агрегированных данных по магазину
        store_daily_data = merged_data.groupby('date').agg(
            cnt=('cnt', 'sum'),
        ).reset_index()

        merged_data['event_name_1'] = merged_data['event_name_1'].fillna('no_event')
        merged_data['event_type_1'] = merged_data['event_type_1'].fillna('no_event')
        merged_data['event_name_2'] = merged_data['event_name_2'].fillna('no_event')
        merged_data['event_type_2'] = merged_data['event_type_2'].fillna('no_event')

        events_daily = merged_data.groupby('date')[[
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
        ]].first().reset_index()

        store_daily_data = pd.merge(store_daily_data, events_daily, on='date',
                                    how='left')

        if cashback_col_name in merged_data.columns:
            cashback_daily = merged_data.groupby('date')[
                cashback_col_name].first().reset_index()
            store_daily_data = pd.merge(store_daily_data, cashback_daily,
                                        on='date', how='left')
            if cashback_col_name in store_daily_data.columns:
                store_daily_data[cashback_col_name] = store_daily_data[
                    cashback_col_name].fillna(0)

        store_daily_data.set_index('date', inplace=True)
        store_daily_data = store_daily_data.reindex(all_dates)

        if 'cnt' in store_daily_data.columns:
            store_daily_data['cnt'] = store_daily_data['cnt'].replace(0, np.nan)
            store_daily_data['cnt'] = store_daily_data['cnt'].interpolate(
                method='linear', limit_direction='both'
            )
            store_daily_data['cnt'] = store_daily_data['cnt'].fillna(method='bfill')
            store_daily_data['cnt'] = store_daily_data['cnt'].fillna(method='ffill')
            store_daily_data['cnt'] = store_daily_data['cnt'].fillna(0)

        for col in ['event_name_1', 'event_type_1', 'event_name_2',
                    'event_type_2']:
            if col in store_daily_data.columns:
                store_daily_data[col] = store_daily_data[col].fillna('no_event')
        if cashback_col_name in store_daily_data.columns:
            store_daily_data[cashback_col_name] = store_daily_data[
                cashback_col_name].fillna(0)

        store_daily_data['dayofweek'] = store_daily_data.index.dayofweek
        store_daily_data['month'] = store_daily_data.index.month
        store_daily_data['year'] = store_daily_data.index.year
        store_daily_data['day'] = store_daily_data.index.day
        store_daily_data['is_weekend'] = (
                store_daily_data.index.dayofweek >= 5).astype(int)

        self.data = store_daily_data

        print(f"Данные загружены и предобработаны для {len(self.items)} товаров.")
        print(f"Период данных: {all_dates.min().date()} по {all_dates.max().date()}")

        return self.items_data

    def add_time_features(self, data, item_id=None):
        """
        Добавляет временные признаки в DataFrame для обучения модели.

        Генерирует лаги, статистики скользящего окна (среднее, стд. отклонение),
        разницу цен и различные календарные признаки. Удаляет названия событий,
        оставляя типы событий для обработки CatBoost.

        Args:
            data (pd.DataFrame): Входной DataFrame с DatetimeIndex и столбцами целевой переменной ('cnt') и 'sell_price'.
            item_id (str, optional): Если указан, добавляет 'item_id' как признак.
                                     По умолчанию None.

        Returns:
            pd.DataFrame: DataFrame с добавленными признаками и удаленными NaN.
        """
        df = data.copy()

        # Лаговые признаки для целевой переменной
        lags_cnt = [1, 2, 3, 7, 14, 21, 28, 35]
        for lag in lags_cnt:
            df[f'cnt_lag_{lag}'] = df['cnt'].shift(lag)

        # Признаки скользящего окна для целевой переменной
        windows_cnt = [7, 14, 21, 28]
        for window in windows_cnt:
            df[f'cnt_rolling_mean_{window}'] = df['cnt'].rolling(
                window=window).mean()
            df[f'cnt_rolling_std_{window}'] = df['cnt'].rolling(
                window=window).std()

        if ('cnt_rolling_mean_7' in df.columns and
                'cnt_rolling_mean_28' in df.columns):
            df['cnt_rolling_mean_7_28_diff'] = (
                    df['cnt_rolling_mean_7'] - df['cnt_rolling_mean_28']
            )

        # Признаки, связанные с ценой
        if 'sell_price' in df.columns:
            df['sell_price_diff_1'] = df['sell_price'].diff(1)
            for lag in [1, 7]:
                df[f'sell_price_lag_{lag}'] = df['sell_price'].shift(lag)

        # Календарные признаки
        df['dayofweek'] = df['dayofweek'].astype(int)
        df['month'] = df['month'].astype(int)
        df['year'] = df['year'].astype(int)
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        df['weekofyear'] = df.index.strftime('%U').astype(int)
        # Циклические признаки для компонентов времени
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        # Признак взаимодействия
        df['weekend_month'] = df['is_weekend'] * df['month']

        # Добавление item_id как признака
        if item_id is not None:
            df['item_id'] = item_id

        # Удаление названий событий, сохранение типов для CatBoost
        cols_to_drop = ['event_name_1', 'event_name_2']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Заполнение NaN, возникших из-за diff или начальных скользящих окон
        if 'sell_price_diff_1' in df.columns:
            df['sell_price_diff_1'] = df['sell_price_diff_1'].fillna(0)
        if 'cnt_rolling_mean_7_28_diff' in df.columns:
            df['cnt_rolling_mean_7_28_diff'] = df[
                'cnt_rolling_mean_7_28_diff'].fillna(0)

        # Удаление строк с NaN, созданными лаговыми/скользящими признаками
        df = df.dropna()

        return df

    def train_test_split(self, test_size=0.2):
        """
        Разделяет предобработанные данные на обучающую и тестовую выборки.

        Применяет `add_time_features` для генерации признаков перед разделением.
        Гарантирует хронологическое разделение (train до test).
        Обновляет `self.items`, включая только товары с достаточным количеством
        данных для разделения.

        Args:
            test_size (float): Доля данных для тестовой выборки (например, 0.2).

        Returns:
            dict: Словарь `split_results` с DataFrame'ами train/test для
                  каждого успешно разделенного товара.

        Raises:
            ValueError: Если данные не были загружены.
        """
        if not self.items_data:
            raise ValueError(
                "Данные не загружены. Вызовите load_and_preprocess_data."
            )

        self.split_results = {}
        valid_items_for_split = []

        # Разделение данных для каждого товара
        for item_id, item_data in self.items_data.items():
            data_with_features = self.add_time_features(item_data, item_id)

            if data_with_features.empty or len(data_with_features) < 2:
                continue

            split_idx = int(len(data_with_features) * (1 - test_size))

            # Проверка корректности индекса разделения
            if split_idx <= 0 or split_idx >= len(data_with_features):
                continue

            train_data_item = data_with_features.iloc[:split_idx].copy()
            test_data_item = data_with_features.iloc[split_idx:].copy()

            self.split_results[item_id] = (train_data_item, test_data_item)
            valid_items_for_split.append(item_id)

        # Разделение агрегированных данных по магазину
        agg_data_with_features = self.add_time_features(self.data)
        if agg_data_with_features.empty or len(agg_data_with_features) < 2:
            print("Предупреждение: Недостаточно агрегированных данных после генерации признаков. Разделение пропущено.")
            self.train_data = None
            self.test_data = None
        else:
            split_idx_agg = int(len(agg_data_with_features) * (1 - test_size))
            if split_idx_agg <= 0 or split_idx_agg >= len(agg_data_with_features):
                print("Предупреждение: Некорректный индекс разделения для агрегированных данных. Разделение пропущено.")
                self.train_data = None
                self.test_data = None
            else:
                self.train_data = agg_data_with_features.iloc[:split_idx_agg].copy()
                self.test_data = agg_data_with_features.iloc[split_idx_agg:].copy()

        # Обновление списка товаров и статуса разделения
        self.items = valid_items_for_split
        self.data_split = True

        print(f"Разделение данных завершено для {len(self.items)} товаров.")
        if self.train_data is not None and not self.train_data.empty:
            print(f"Агрегированный период обучения: {self.train_data.index.min().date()} по {self.train_data.index.max().date()}")
            print(f"Агрегированный период теста:  {self.test_data.index.min().date()} по {self.test_data.index.max().date()}")

        return self.split_results

    def prepare_features_targets(self, target_col='cnt', item_id=None):
        """
        Подготавливает наборы признаков (X) и целевой переменной (y).

        Выбирает источник данных (по товару или агрегированный) на основе`item_id`.
        Определяет столбцы признаков и категориальные признаки.

        Args:
            target_col (str): Название столбца целевой переменной.
            item_id (str, optional): ID товара для подготовки данных.
                                     Если None, используются агрегированные данные.
                                     По умолчанию None.

        Returns:
            tuple: Кортеж, содержащий X_train, y_train, X_test, y_test (pandas
                   DataFrame/Series) и список названий категориальных признаков.

        Raises:
            ValueError: Если данные не разделены или данные для `item_id` не найдены в результатах разделения.
        """
        if not self.data_split:
            raise ValueError("Данные не разделены. Вызовите train_test_split.")

        train_data_source = None
        test_data_source = None

        if item_id is None:
            # Использование агрегированных данных
            if self.train_data is None or self.test_data is None:
                raise ValueError("Агрегированные данные train/test недоступны.")
            train_data_source = self.train_data
            test_data_source = self.test_data
        else:
            # Использование данных конкретного товара
            if item_id not in self.split_results:
                raise ValueError(f"Результаты разделения для товара {item_id} не найдены.")
            train_data_source, test_data_source = self.split_results[item_id]

        # Определение столбцов для исключения из признаков
        exclude_columns = [target_col]
        if 'sell_price' in train_data_source.columns and target_col != 'sell_price':
            exclude_columns.append('sell_price')

        feature_columns = [
            col for col in train_data_source.columns if col not in exclude_columns
        ]

        # Определение потенциальных категориальных признаков
        cat_features_potential = [
            'dayofweek', 'month', 'year', 'quarter', 'weekofyear', 'is_weekend',
            'event_type_1', 'event_type_2'
        ]
        if 'item_id' in feature_columns:
            cat_features_potential.append('item_id')

        # Сохранение определенных признаков и цели
        self.features = feature_columns
        self.target = target_col
        # Фильтрация категориальных признаков по наличию в данных
        self.cat_features = [
            f for f in cat_features_potential if f in feature_columns
        ]

        # Создание итоговых наборов train/test
        X_train = train_data_source[feature_columns]
        y_train = train_data_source[target_col]
        X_test = test_data_source[feature_columns]
        y_test = test_data_source[target_col]

        return X_train, y_train, X_test, y_test, self.cat_features

    def train_catboost_model(self, target_col='cnt', params=None, item_id=None):
        """
        Обучает модель CatBoostRegressor для указанной цели и товара.

        Использует признаки и цель, подготовленные `prepare_features_targets`.
        Сохраняет обученную модель в словаре `self.models`.

        Args:
            target_col (str): Название столбца целевой переменной.
            params (dict, optional): Словарь параметров для CatBoostRegressor.
                                     Если None, используются параметры по умолчанию.
                                     По умолчанию None.
            item_id (str, optional): ID товара для обучения модели.
                                     Если None, обучается на агрегированных данных.
                                     По умолчанию None.

        Returns:
            catboost.CatBoostRegressor or None: Обученная модель CatBoost или
                                                None в случае ошибки обучения.
        """
        try:
            X_train, y_train, _, _, current_cat_features = \
                self.prepare_features_targets(target_col, item_id)
        except ValueError as e:
            print(f"Ошибка подготовки данных для обучения товара {item_id}: {e}")
            return None

        if X_train.empty:
            print(f"Нет данных для обучения для товара {item_id}. Обучение пропущено.")
            return None

        # Параметры CatBoost по умолчанию
        if params is None:
            params = {
                'loss_function': 'MAE',
                'iterations': 700,
                'learning_rate': 0.03,
                'random_seed': 42,
                'verbose': False  # Подавление вывода итераций при обучении
            }
        fit_verbose = False  # Убедимся, что verbose=False для метода fit

        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"
        print(f"Обучение модели CatBoost для {target_col}{item_str}...")

        model = CatBoostRegressor(**params)

        # Обучение модели с передачей категориальных признаков
        model.fit(X_train, y_train, cat_features=current_cat_features,
                  verbose=fit_verbose)

        # Отображение важности признаков
        try:
            # ИСПРАВЛЕНИЕ: get_feature_importance не принимает data и cat_features
            feature_importance = model.get_feature_importance()
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)

            print(f"\nТоп-10 важных признаков{item_str}:")
            print(importance_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"Не удалось получить важность признаков: {e}")

        # Сохранение обученной модели и связанных с ней признаков
        model_key = f'CatBoost_{item_id}_{target_col}' if item_id \
            else f'CatBoost_{target_col}'
        self.models[model_key] = model
        # Сохраняем имена признаков и категориальных признаков, использованных для этой модели
        self.models[model_key]._train_features = list(X_train.columns)
        self.models[model_key]._cat_features_names = current_cat_features

        return model

    def forecast(self, target_col='cnt', horizon='week', item_id=None):
        """
        Генерирует прогнозы на указанный горизонт с использованием обученной модели.

        Выполняет итеративное прогнозирование: предсказывает на один шаг вперед,
        добавляет предсказание в историю, пересчитывает признаки (лаги,
        скользящие статистики) и повторяет для всего горизонта.

        Args:
            target_col (str): Название целевой переменной для прогнозирования.
            horizon (str): Горизонт прогнозирования ('week', 'month', 'quarter').
            item_id (str, optional): ID товара для прогнозирования.
                                     Если None, прогнозируются агрегированные данные.
                                     По умолчанию None.

        Returns:
            pd.DataFrame: DataFrame с прогнозами и DatetimeIndex.

        Raises:
            ValueError: Если указанная модель не найдена или не может быть обучена,
                        если горизонт некорректен, или если данные не разделены.
        """
        model_key = f'CatBoost_{item_id}_{target_col}' if item_id \
            else f'CatBoost_{target_col}'

        # Проверка наличия модели или попытка обучения
        if model_key not in self.models:
            print(f"Модель {model_key} не найдена. Попытка обучения...")
            model = self.train_catboost_model(target_col=target_col,
                                              item_id=item_id)
            if model is None:
                raise ValueError(f"Не удалось обучить или найти модель {model_key}.")
        else:
            model = self.models[model_key]

        # Проверка наличия необходимых атрибутов у модели
        if not hasattr(model, '_train_features'):
            raise ValueError(f"У модели {model_key} отсутствуют '_train_features'. Требуется переобучение.")
        model_features = model._train_features
        model_cat_features = getattr(model, '_cat_features_names', [])

        # Определение длительности прогноза
        if horizon == 'week':
            days = 7
        elif horizon == 'month':
            days = 30
        elif horizon == 'quarter':
            days = 90
        else:
            raise ValueError("Горизонт должен быть 'week', 'month' или 'quarter'")

        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"
        print(f"Генерация прогноза на {horizon} ({days} дней) для {target_col}{item_str}...")

        # Получение исторических данных для итеративного прогнозирования
        if not self.data_split:
            raise ValueError("Данные не разделены. Вызовите train_test_split.")

        if item_id is None:
            if self.train_data is None:
                raise ValueError("Агрегированные обучающие данные недоступны.")
            history = self.train_data.copy()
        else:
            if item_id not in self.split_results:
                raise ValueError(f"Результаты разделения для товара {item_id} не найдены.")
            train_data_item, _ = self.split_results[item_id]
            history = train_data_item.copy()

        # Подготовка DataFrame для прогноза и данных для поиска будущей информации
        start_forecast_date = history.index[-1] + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start=start_forecast_date, periods=days,
                                       freq='D')
        forecast_df = pd.DataFrame(index=forecast_dates)
        forecast_df[target_col] = np.nan

        future_info_lookup = self._full_dates_data.set_index('date') \
            if self._full_dates_data is not None else None
        cashback_col = f'CASHBACK_{self.store_id}'
        item_full_data = self.items_data.get(item_id, None)

        iterative_history = history.copy()

        # Параметры генерации признаков (должны совпадать с add_time_features)
        lags_cnt = [1, 2, 3, 7, 14, 21, 28, 35]
        windows_cnt = [7, 14, 21, 28]
        lags_price = [1, 7]

        # Цикл итеративного прогнозирования
        for current_date in forecast_dates:
            current_features_df = pd.DataFrame(index=[current_date])

            # 1. Создание базовых календарных признаков
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

            # 2. Добавление item_id, если необходимо
            if item_id is not None and 'item_id' in model_features:
                current_features_df['item_id'] = item_id

            # 3. Добавление внешних факторов (события, кэшбек, цена)
            event_type_1_val = 'no_event'
            event_type_2_val = 'no_event'
            cashback_value = 0

            if future_info_lookup is not None and current_date in future_info_lookup.index:
                date_info = future_info_lookup.loc[current_date]
                event_type_1_val = date_info.get('event_type_1', 'no_event')
                if pd.isna(event_type_1_val): event_type_1_val = 'no_event'
                event_type_2_val = date_info.get('event_type_2', 'no_event')
                if pd.isna(event_type_2_val): event_type_2_val = 'no_event'
                if cashback_col in date_info:
                    cashback_value = date_info[cashback_col]

            if 'event_type_1' in model_features:
                current_features_df['event_type_1'] = event_type_1_val
            if 'event_type_2' in model_features:
                current_features_df['event_type_2'] = event_type_2_val
            if cashback_col in model_features:
                current_features_df[cashback_col] = cashback_value

            # Получение текущей цены (будущей или последней известной)
            current_sell_price = np.nan
            if item_full_data is not None and current_date in item_full_data.index:
                current_sell_price = item_full_data.loc[current_date, 'sell_price']
            if pd.isna(current_sell_price) and 'sell_price' in iterative_history:
                current_sell_price = iterative_history['sell_price'].iloc[-1]
            if pd.isna(current_sell_price):
                current_sell_price = 0  # Значение по умолчанию, если цена неизвестна

            # Расчет признаков цены
            if 'sell_price_diff_1' in model_features:
                prev_date = current_date - pd.Timedelta(days=1)
                prev_sell_price = 0  # Значение по умолчанию
                if prev_date in iterative_history.index and 'sell_price' in iterative_history:
                    prev_sell_price = iterative_history.loc[prev_date, 'sell_price']
                    if pd.isna(prev_sell_price): prev_sell_price = 0
                current_features_df['sell_price_diff_1'] = current_sell_price - prev_sell_price

            if 'sell_price' in iterative_history:
                temp_series_price = iterative_history['sell_price'].fillna(0)
                for lag in lags_price:
                    feature_name = f'sell_price_lag_{lag}'
                    if feature_name in model_features:
                        if len(temp_series_price) >= lag:
                            current_features_df[feature_name] = temp_series_price.iloc[-lag]
                        else:
                            current_features_df[feature_name] = 0

            # 4. Расчет лагов и скользящих статистик целевой переменной из истории
            temp_series_cnt = iterative_history[target_col].fillna(0)

            for lag in lags_cnt:
                feature_name = f'cnt_lag_{lag}'
                if feature_name in model_features:
                    if len(temp_series_cnt) >= lag:
                        current_features_df[feature_name] = temp_series_cnt.iloc[-lag]
                    else:
                        current_features_df[feature_name] = 0

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
                mean7 = rolling_means.get(7, 0)
                mean28 = rolling_means.get(28, 0)
                current_features_df['cnt_rolling_mean_7_28_diff'] = mean7 - mean28

            # 5. Выравнивание признаков с ожидаемым входом модели
            for feature in model_features:
                if feature not in current_features_df.columns:
                    # Заполнение пропущенных признаков
                    if feature in model_cat_features:
                        current_features_df[feature] = 'no_event' # Категория по умолчанию
                    else:
                        current_features_df[feature] = 0 # Числовое значение по умолчанию

            final_features_for_pred = current_features_df[model_features]
            # Финальная проверка на NaN в числовых/категориальных столбцах
            num_cols = final_features_for_pred.select_dtypes(include=np.number).columns
            final_features_for_pred[num_cols] = final_features_for_pred[num_cols].fillna(0)
            cat_cols_in_final = [col for col in model_cat_features if col in final_features_for_pred.columns]
            final_features_for_pred[cat_cols_in_final] = final_features_for_pred[cat_cols_in_final].fillna('no_event')

            # 6. Предсказание
            prediction = model.predict(final_features_for_pred)[0]
            prediction = max(0, prediction)  # Гарантия неотрицательного прогноза

            # 7. Сохранение предсказания
            forecast_df.loc[current_date, target_col] = prediction

            # 8. Обновление истории для следующей итерации
            new_history_row_data = {}
            # Копирование признаков, использованных для предсказания
            for col in model_features:
                if col in final_features_for_pred.columns:
                    new_history_row_data[col] = final_features_for_pred.iloc[0][col]
            # Добавление самого предсказания
            new_history_row_data[target_col] = prediction
            # Добавление использованной цены (необходимо для будущих признаков цены)
            new_history_row_data['sell_price'] = current_sell_price
            # Добавление других столбцов из истории, не попавших в признаки
            for col in iterative_history.columns:
                if col not in new_history_row_data:
                    if col in current_features_df.columns:
                        new_history_row_data[col] = current_features_df.iloc[0][col]
                    else: # Значение по умолчанию на основе исходного типа данных
                        if pd.api.types.is_numeric_dtype(iterative_history[col]):
                            new_history_row_data[col] = 0
                        else:
                            new_history_row_data[col] = 'no_event'

            new_history_row_df = pd.DataFrame([new_history_row_data], index=[current_date])
            # Гарантия совпадения столбцов с историей перед конкатенацией
            new_history_row_df = new_history_row_df.reindex(columns=iterative_history.columns)
            iterative_history = pd.concat([iterative_history, new_history_row_df])

        # Сохранение итогового прогноза
        forecast_key = f'CatBoost_{item_id}_{target_col}_{horizon}' if item_id \
            else f'CatBoost_{target_col}_{horizon}'
        self.forecasts[forecast_key] = forecast_df

        return forecast_df

    def evaluate(self, target_col='cnt', horizon='week', item_id=None):
        """
        Оценивает прогноз по фактическим тестовым данным (MAE, RMSE, MAPE, R2).

        Получает прогноз для указанного горизонта и сравнивает его с соответствующей частью тестовой выборки.

        Args:
            target_col (str): Название целевой переменной.
            horizon (str): Горизонт прогнозирования ('week', 'month', 'quarter'),
                           который был предсказан.
            item_id (str, optional): ID товара для оценки. Если None,
                                     оценивается агрегированный прогноз.
                                     По умолчанию None.

        Returns:
            tuple: Кортеж, содержащий метрики MAE, RMSE, MAPE, R2.
                   Возвращает NaN, если оценка невозможна.
        """
        forecast_key = f'CatBoost_{item_id}_{target_col}_{horizon}' if item_id \
            else f'CatBoost_{target_col}_{horizon}'

        # Проверка наличия прогноза
        if forecast_key not in self.forecasts:
            print(f"Прогноз {forecast_key} не найден. Попытка генерации...")
            try:
                self.forecast(target_col, horizon, item_id)
            except Exception as e:
                print(f"Не удалось сгенерировать прогноз для оценки: {e}")
                return np.nan, np.nan, np.nan, np.nan
            if forecast_key not in self.forecasts:
                print(f"Прогноз {forecast_key} все еще не найден после попытки.")
                return np.nan, np.nan, np.nan, np.nan

        # Определение длины периода оценки
        if horizon == 'week':
            days = 7
        elif horizon == 'month':
            days = 30
        elif horizon == 'quarter':
            days = 90
        else:
            raise ValueError("Горизонт должен быть 'week', 'month' или 'quarter'")

        forecast = self.forecasts[forecast_key]

        # Получение соответствующих фактических тестовых данных
        if not self.data_split:
            print("Предупреждение: Данные не разделены. Оценка невозможна.")
            return np.nan, np.nan, np.nan, np.nan

        test_data_source = None
        if item_id is None:
            if self.test_data is None:
                print("Предупреждение: Агрегированные тестовые данные недоступны для оценки.")
                return np.nan, np.nan, np.nan, np.nan
            test_data_source = self.test_data
        else:
            if item_id not in self.split_results:
                print(f"Предупреждение: Результаты разделения для товара {item_id} не найдены для оценки.")
                return np.nan, np.nan, np.nan, np.nan
            _, test_data_source = self.split_results[item_id]

        # Выравнивание прогноза и фактических данных по горизонту оценки
        common_index = test_data_source.index.intersection(forecast.index)
        common_index = common_index[:days]  # Ограничение горизонтом прогноза

        if common_index.empty:
            print("Предупреждение: Не найдено пересекающихся дат между прогнозом и фактом для оценки.")
            return np.nan, np.nan, np.nan, np.nan

        actual = test_data_source.loc[common_index, target_col]
        forecast_aligned = forecast.loc[common_index, target_col]

        if len(actual) == 0:
            print("Предупреждение: Нет фактических данных для оценки после выравнивания.")
            return np.nan, np.nan, np.nan, np.nan

        # Расчет метрик
        mae = mean_absolute_error(actual, forecast_aligned)
        rmse = np.sqrt(mean_squared_error(actual, forecast_aligned))
        r2 = r2_score(actual, forecast_aligned)

        # Аккуратный расчет MAPE, избегая деления на ноль
        mask = actual != 0
        if np.sum(mask) > 0:
            actual_masked = actual[mask]
            forecast_masked = forecast_aligned[mask]
            # Исключаем случаи, когда прогноз равен 0, для корректного MAPE
            zero_forecast_mask = forecast_masked != 0
            if np.sum(zero_forecast_mask) > 0:
                mape = np.mean(np.abs(
                    (actual_masked[zero_forecast_mask] - forecast_masked[zero_forecast_mask]) /
                    actual_masked[zero_forecast_mask]
                )) * 100
            else:
                mape = np.nan # Все ненулевые факты имели нулевой прогноз
        else:
            mape = np.nan # Все фактические значения были нулевыми

        # Сохранение метрик
        self.metrics[(forecast_key, horizon)] = {
            'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2
        }

        # Вывод метрик
        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"
        print(f"Метрики оценки (горизонт {horizon}){item_str}:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%" if not np.isnan(mape) else "  MAPE: N/A")
        print(f"  R2:   {r2:.2f}")

        return mae, rmse, mape, r2

    def visualize_forecast(self, target_col='cnt', horizon='week', item_id=None):
        """
        Строит график прогноза по сравнению с фактическими тестовыми данными и недавними обучающими данными.

        Args:
            target_col (str): Название целевой переменной.
            horizon (str): Горизонт прогнозирования ('week', 'month', 'quarter') для визуализации.
            item_id (str, optional): ID товара для визуализации. Если None,
                                     визуализируется агрегированный прогноз.
                                     По умолчанию None.

        Returns:
            tuple: Кортеж, содержащий хвост обучающих данных, фактические
                   тестовые данные и прогнозные данные, использованные
                   на графике (pandas Series). Возвращает (None, None, None),
                   если визуализация невозможна.
        """
        forecast_key = f'CatBoost_{item_id}_{target_col}_{horizon}' if item_id \
            else f'CatBoost_{target_col}_{horizon}'

        # Проверка наличия прогноза
        if forecast_key not in self.forecasts:
            print(f"Прогноз {forecast_key} не найден. Попытка генерации...")
            try:
                self.forecast(target_col, horizon, item_id=item_id)
            except Exception as e:
                print(f"Не удалось сгенерировать прогноз для визуализации: {e}")
                return None, None, None
            if forecast_key not in self.forecasts:
                print(f"Прогноз {forecast_key} все еще не найден после попытки.")
                return None, None, None

        # Определение длины периода для графика
        if horizon == 'week':
            days = 7
        elif horizon == 'month':
            days = 30
        elif horizon == 'quarter':
            days = 90
        else:
            raise ValueError("Горизонт должен быть 'week', 'month' или 'quarter'")

        forecast = self.forecasts[forecast_key]

        # Получение соответствующих обучающих/тестовых данных
        if not self.data_split:
            print("Предупреждение: Данные не разделены. Визуализация невозможна.")
            return None, None, None

        train_data_source = None
        test_data_source = None
        if item_id is None:
            if self.train_data is None or self.test_data is None:
                print("Предупреждение: Агрегированные данные train/test недоступны для визуализации.")
                return None, None, None
            train_data_source = self.train_data
            test_data_source = self.test_data
        else:
            if item_id not in self.split_results:
                print(f"Предупреждение: Результаты разделения для товара {item_id} не найдены для визуализации.")
                return None, None, None
            train_data_source, test_data_source = self.split_results[item_id]

        # Подготовка данных для графика
        train_data_vis = train_data_source[target_col]
        # Получение фактических данных, соответствующих горизонту прогноза
        common_index = test_data_source.index.intersection(forecast.index)
        common_index = common_index[:days]

        if common_index.empty:
            print("Предупреждение: Нет пересекающихся дат между прогнозом и фактом для визуализации.")
            actual_vis = pd.Series(dtype=float) # Пустая серия для графика
            # Получение прогноза для указанного горизонта
            forecast_vis = forecast.loc[forecast.index[:days], target_col]
        else:
            actual_vis = test_data_source.loc[common_index, target_col]
            forecast_vis = forecast.loc[common_index, target_col]

        # Получение хвоста обучающих данных для контекста
        train_data_vis_tail = train_data_vis.iloc[-30:] # Показ последних 30 точек обучения

        # Создание графика
        plt.figure(figsize=(12, 6))
        plt.plot(train_data_vis_tail.index, train_data_vis_tail,
                 label='Обучающие данные', color='blue', marker='.')
        if not actual_vis.empty:
            plt.plot(actual_vis.index, actual_vis,
                     label='Фактические данные', color='green', marker='.')
        if not forecast_vis.empty:
            plt.plot(forecast_vis.index, forecast_vis,
                     label='Прогноз CatBoost', color='red', linestyle='--', marker='.')

        item_str = f" (Товар: {item_id})" if item_id else " (Агрегированные)"
        plt.title(f'Прогноз {target_col.capitalize()} vs Факт - Горизонт {horizon.capitalize()}{item_str}')
        plt.xlabel('Дата')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return train_data_vis_tail, actual_vis, forecast_vis

    def run_all(self, target_col='cnt', item_id=None):
        """
        Запускает полный цикл для одного товара или агрегированных данных.

        Обучает модель, генерирует прогнозы на горизонты 'week', 'month',
        'quarter', оценивает каждый прогноз и визуализирует их.

        Args:
            target_col (str): Название целевой переменной.
            item_id (str, optional): ID товара для обработки. Если None,
                                     обрабатываются агрегированные данные.
                                     По умолчанию None.

        Returns:
            pd.DataFrame: DataFrame с метриками оценки для каждого горизонта.
                          Возвращает пустой DataFrame, если обучение модели не удалось.
        """
        # Сначала обучаем модель
        model = self.train_catboost_model(target_col, item_id=item_id)
        if model is None:
            print(f"Не удалось обучить модель для {item_id if item_id else 'агрегированных данных'}. Пропуск run_all.")
            return pd.DataFrame()

        horizons = ['week', 'month', 'quarter']
        results = []

        # Прогноз, оценка и визуализация для каждого горизонта
        for horizon in horizons:
            try:
                self.forecast(target_col, horizon, item_id=item_id)
                mae, rmse, mape, r2 = self.evaluate(target_col, horizon, item_id=item_id)
                # Визуализация только при успешной оценке (метрики не NaN)
                if not np.isnan(mae):
                     self.visualize_forecast(target_col, horizon, item_id=item_id)
                else:
                     print(f"Пропуск визуализации для {horizon} из-за проблем с оценкой.")

            except Exception as e:
                print(f"Ошибка обработки горизонта '{horizon}' для {item_id if item_id else 'агрегированных данных'}: {e}")
                mae, rmse, mape, r2 = np.nan, np.nan, np.nan, np.nan

            results.append({
                'Horizon': horizon, 'MAE': mae, 'RMSE': rmse,
                'MAPE': mape, 'R2': r2
            })

        results_df = pd.DataFrame(results)
        item_str = f" для товара {item_id}" if item_id else " для агрегированных данных"
        print(f"\n--- Итоговые результаты{item_str} ---")
        print(results_df.round(2).to_string(index=False)) # Улучшенное форматирование

        return results_df

    def run_all_items(self, target_col='cnt', items_limit=None, horizons=None):
        """
        Запускает полный цикл прогнозирования для всех валидных товаров магазина.

        Итерирует по товарам, обучает модели, генерирует прогнозы, оценивает
        и собирает результаты. Предоставляет итоговую статистику и рейтинги.

        Args:
            target_col (str): Название целевой переменной.
            items_limit (int, optional): Максимальное количество товаров для обработки.
                                         Если None, обрабатываются все валидные товары.
                                         По умолчанию None.
            horizons (list, optional): Список горизонтов для прогнозирования и оценки
                                       (например, ['week', 'month']). Если None,
                                       используется ['week', 'month', 'quarter'].
                                       По умолчанию None.

        Returns:
            tuple: Кортеж, содержащий:
                - pd.DataFrame: DataFrame с метриками оценки для всех товаров
                                и горизонтов.
                - dict: Словарь, где ключи - горизонты, а значения -
                        DataFrame'ы с агрегированными прогнозами для всех
                        товаров на этом горизонте.
        """
        if horizons is None:
            horizons = ['week', 'month', 'quarter']

        if not self.data_split:
            print("Ошибка: Данные не разделены. Вызовите train_test_split() перед run_all_items.")
            return pd.DataFrame(), {}

        if not self.items:
            print("Ошибка: Нет валидных товаров для обработки в run_all_items.")
            return pd.DataFrame(), {}

        items_to_process = self.items[:items_limit] if items_limit else self.items
        num_items = len(items_to_process)
        print(f"Запуск полного цикла для {num_items} товаров...")

        all_results = []
        all_forecasts = {horizon: pd.DataFrame() for horizon in horizons}

        # Обработка каждого товара
        for i, item_id in enumerate(items_to_process):
            print(f"\n--- Обработка товара: {item_id} ({i + 1}/{num_items}) ---")

            try:
                # Обучение модели для товара
                model = self.train_catboost_model(target_col, item_id=item_id)
                if model is None:
                    print(f"Пропуск товара {item_id} из-за ошибки обучения.")
                    # Запись ошибки для всех горизонтов
                    for horizon in horizons:
                        all_results.append({
                            'item_id': item_id, 'horizon': horizon, 'MAE': np.nan,
                            'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan
                        })
                    continue

                item_horizon_results = []
                # Прогноз и оценка для каждого указанного горизонта
                for horizon in horizons:
                    forecast_df = self.forecast(target_col, horizon, item_id=item_id)
                    mae, rmse, mape, r2 = self.evaluate(target_col, horizon, item_id=item_id)

                    # Агрегация прогнозов
                    forecast_df_with_id = forecast_df.copy()
                    forecast_df_with_id['item_id'] = item_id
                    all_forecasts[horizon] = pd.concat(
                        [all_forecasts[horizon], forecast_df_with_id],
                        ignore_index=True
                    )

                    item_horizon_results.append({
                        'item_id': item_id, 'horizon': horizon, 'MAE': mae,
                        'RMSE': rmse, 'MAPE': mape, 'R2': r2
                    })

                # Визуализация прогноза на 'week' (или первый горизонт)
                vis_horizon = 'week' if 'week' in horizons else horizons[0]
                self.visualize_forecast(target_col, vis_horizon, item_id=item_id)
                all_results.extend(item_horizon_results)

            except Exception as e:
                print(f"Ошибка при полной обработке товара {item_id}: {e}")
                # Запись ошибки для всех горизонтов при непредвиденной ошибке
                for horizon in horizons:
                    # Избегаем дублирования записей об ошибке, если она произошла при обучении
                    if not any(r['item_id'] == item_id and r['horizon'] == horizon for r in all_results):
                         all_results.append({
                             'item_id': item_id, 'horizon': horizon, 'MAE': np.nan,
                             'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan
                         })
                continue # Переход к следующему товару

        results_df = pd.DataFrame(all_results)

        # --- Итоговый отчет ---
        print("\n=== Итоговые результаты по всем товарам ===")
        results_df_clean = results_df.dropna(subset=['MAE']) # Используем только успешные запуски для средних

        if not results_df_clean.empty:
            # Расчет средних метрик по горизонтам
            metrics_summary = results_df_clean.groupby('horizon')[
                ['MAE', 'RMSE', 'MAPE', 'R2']
            ].mean()
            print("Средние метрики по горизонтам (для успешно обработанных товаров):")
            print(metrics_summary.round(2))
        else:
            print("Нет успешных результатов для расчета средних метрик.")

        # Вывод лучших/худших товаров по горизонтам
        for horizon in horizons:
            horizon_results = results_df_clean[
                results_df_clean['horizon'] == horizon
            ].sort_values('MAE') # Сортировка по MAE

            if not horizon_results.empty:
                print(f"\n--- Топ 5 товаров (низкий MAE) для горизонта {horizon.capitalize()} ---")
                print(horizon_results.head(5)[
                          ['item_id', 'MAE', 'MAPE', 'R2']
                      ].round(2).to_string(index=False))

                print(f"\n--- Худшие 5 товаров (высокий MAE) для горизонта {horizon.capitalize()} ---")
                # Убедимся, что нет NaN перед вызовом tail
                print(horizon_results.dropna(subset=['MAE']).tail(5)[
                          ['item_id', 'MAE', 'MAPE', 'R2']
                      ].round(2).to_string(index=False))
            else:
                print(f"\nНет валидных результатов для рейтинга товаров на горизонте {horizon}.")

        return results_df, all_forecasts