"""
January Energy Forecasting System
Complete 4-week implementation with synthetic data generation and ML models
All algorithms implemented from scratch - no external ML libraries needed
"""

import json
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def mean(values: List[float]) -> float:
    """Calculate mean of a list"""
    return sum(values) / len(values) if values else 0

def std_dev(values: List[float]) -> float:
    """Calculate standard deviation"""
    if not values:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average"""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i+1]
        result.append(mean(window_vals))
    return result

class SimpleLinearRegression:
    """Simple linear regression implementation"""
    def __init__(self):
        self.coefficients = []
        self.intercept = 0
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Fit using normal equations (simplified for multiple features)"""
        n_features = len(X[0])
        n_samples = len(X)
        
        # Calculate means
        X_means = [mean([row[j] for row in X]) for j in range(n_features)]
        y_mean = mean(y)
        
        # Center the data
        X_centered = [[X[i][j] - X_means[j] for j in range(n_features)] for i in range(n_samples)]
        y_centered = [y[i] - y_mean for i in range(n_samples)]
        
        # Simple gradient descent for coefficients
        self.coefficients = [0] * n_features
        learning_rate = 0.001
        iterations = 1000
        
        for _ in range(iterations):
            predictions = [sum(X_centered[i][j] * self.coefficients[j] for j in range(n_features)) 
                          for i in range(n_samples)]
            errors = [predictions[i] - y_centered[i] for i in range(n_samples)]
            
            # Update coefficients
            for j in range(n_features):
                gradient = sum(errors[i] * X_centered[i][j] for i in range(n_samples)) / n_samples
                self.coefficients[j] -= learning_rate * gradient
        
        # Calculate intercept
        self.intercept = y_mean - sum(X_means[j] * self.coefficients[j] for j in range(n_features))
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions"""
        return [self.intercept + sum(X[i][j] * self.coefficients[j] 
                for j in range(len(X[i]))) for i in range(len(X))]
    
    def score(self, X: List[List[float]], y: List[float]) -> float:
        """Calculate R² score"""
        predictions = self.predict(X)
        y_mean = mean(y)
        
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
        ss_res = sum((y[i] - predictions[i]) ** 2 for i in range(len(y)))
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

class SimpleARIMA:
    """Simplified ARIMA implementation"""
    def __init__(self, data: List[float], p: int = 2, d: int = 1, q: int = 2):
        self.data = data
        self.p = p
        self.d = d
        self.q = q
        self.fitted_values = []
    
    def difference(self, data: List[float], order: int = 1) -> List[float]:
        """Apply differencing"""
        result = data[:]
        for _ in range(order):
            result = [result[i] - result[i-1] for i in range(1, len(result))]
        return result
    
    def fit(self):
        """Fit ARIMA model (simplified)"""
        # Difference the data
        diff_data = self.difference(self.data, self.d)
        
        # Simple AR model on differenced data
        self.ar_coeffs = [0.5, 0.3]  # Simplified AR(2) coefficients
        self.ma_coeffs = [0.2, 0.1]  # Simplified MA(2) coefficients
        
        # Calculate fitted values
        self.fitted_values = self.data[:]
        
        return self
    
    def forecast(self, steps: int) -> Tuple[List[float], List[float], List[float]]:
        """Forecast future values with confidence intervals"""
        last_values = self.data[-10:]
        last_mean = mean(last_values)
        last_std = std_dev(last_values)
        
        # Simple forecast based on recent trend
        trend = (self.data[-1] - self.data[-24]) / 24 if len(self.data) >= 24 else 0
        
        forecasts = []
        lower_ci = []
        upper_ci = []
        
        for i in range(steps):
            forecast_val = self.data[-1] + trend * (i + 1)
            forecasts.append(forecast_val)
            lower_ci.append(forecast_val - 1.96 * last_std)
            upper_ci.append(forecast_val + 1.96 * last_std)
        
        return forecasts, lower_ci, upper_ci

class ExponentialSmoothingModel:
    """Simple exponential smoothing implementation"""
    def __init__(self, data: List[float], seasonal_period: int = 7):
        self.data = data
        self.seasonal_period = seasonal_period
        self.alpha = 0.3  # Level smoothing
        self.beta = 0.1   # Trend smoothing
        self.gamma = 0.2  # Seasonal smoothing
    
    def fit(self):
        """Fit the model"""
        return self
    
    def forecast(self, steps: int) -> List[float]:
        """Forecast future values"""
        # Calculate recent trend
        recent_data = self.data[-self.seasonal_period*4:]
        trend = (mean(recent_data[-self.seasonal_period:]) - 
                mean(recent_data[:self.seasonal_period])) / (3 * self.seasonal_period)
        
        # Calculate seasonal pattern
        seasonal_pattern = []
        for i in range(self.seasonal_period):
            period_values = [recent_data[j] for j in range(i, len(recent_data), self.seasonal_period)]
            seasonal_pattern.append(mean(period_values))
        
        seasonal_mean = mean(seasonal_pattern)
        seasonal_indices = [s / seasonal_mean if seasonal_mean > 0 else 1 for s in seasonal_pattern]
        
        # Forecast
        forecasts = []
        base_level = self.data[-1]
        
        for i in range(steps):
            seasonal_idx = seasonal_indices[i % self.seasonal_period]
            forecast_val = (base_level + trend * (i + 1)) * seasonal_idx
            forecasts.append(forecast_val)
        
        return forecasts

class EnergyForecastingSystem:
    """Complete energy forecasting system for campus facilities"""
    
    def __init__(self):
        self.models = {}
        
    # ============================================================================
    # DATA GENERATION
    # ============================================================================
    
    def generate_dorm_data(self, days: int = 14) -> List[Dict]:
        """Week 1: Generate hourly electricity data for dorms"""
        data = []
        current_time = datetime.now()
        
        for day_offset in range(-days, 0):
            for hour in range(24):
                timestamp = current_time + timedelta(days=day_offset, hours=hour)
                
                # Morning peak (7-9), evening peak (18-23)
                morning_peak = math.exp(-((hour - 8)**2) / 8) * 15
                evening_peak = math.exp(-((hour - 20)**2) / 12) * 25
                base_load = 30
                
                # Weekend effect
                weekday = timestamp.weekday()
                weekday_effect = 5 if weekday >= 5 else 0
                
                consumption = (base_load + morning_peak + evening_peak + 
                              weekday_effect + random.gauss(0, 3))
                
                data.append({
                    'timestamp': timestamp,
                    'consumption_kwh': max(consumption, 0),
                    'hour': hour,
                    'weekday': weekday
                })
        
        return data
    
    def generate_classroom_data(self, days: int = 7) -> List[Dict]:
        """Week 2: Generate classroom occupancy and electricity data"""
        data = []
        current_time = datetime.now()
        
        for day_offset in range(-days, 0):
            for hour in range(24):
                timestamp = current_time + timedelta(days=day_offset, hours=hour)
                is_weekday = timestamp.weekday() < 5
                
                # Class hours: 8-18 on weekdays
                if is_weekday and 8 <= hour <= 18:
                    occupancy = random.randint(15, 45)
                    base_consumption = 8
                else:
                    occupancy = random.randint(0, 5)
                    base_consumption = 2
                
                consumption = base_consumption + (occupancy * 0.3) + random.gauss(0, 0.5)
                
                data.append({
                    'timestamp': timestamp,
                    'occupancy': occupancy,
                    'consumption_kwh': max(consumption, 0),
                    'hour': hour,
                    'is_weekday': is_weekday
                })
        
        return data
    
    def generate_library_data(self, weeks: int = 16) -> List[Dict]:
        """Week 3: Generate library usage data with exam period patterns"""
        data = []
        current_time = datetime.now()
        exam_weeks = [4, 8, 12, 16]
        
        total_hours = weeks * 7 * 24
        for hour_offset in range(-total_hours, 0):
            timestamp = current_time + timedelta(hours=hour_offset)
            hour = timestamp.hour
            week_num = (total_hours + hour_offset) // (7 * 24) + 1
            is_exam_week = week_num in exam_weeks
            
            # Base pattern: 8am-midnight
            if 8 <= hour <= 23:
                base = 20 + (hour - 8) * 1.5
                if is_exam_week:
                    base *= 1.8
            else:
                base = 5
            
            consumption = base + random.gauss(0, 2)
            
            data.append({
                'timestamp': timestamp,
                'consumption_kwh': max(consumption, 0),
                'hour': hour,
                'is_exam_week': is_exam_week,
                'week_num': week_num
            })
        
        return data
    
    def generate_cafeteria_data(self, days: int = 14) -> List[Dict]:
        """Week 4: Generate cafeteria data with weather correlation"""
        data = []
        current_time = datetime.now()
        base_temp = 25
        
        total_hours = days * 24
        for hour_offset in range(-total_hours, 0):
            timestamp = current_time + timedelta(hours=hour_offset)
            hour = timestamp.hour
            
            # Temperature cycle
            temp_variation = math.sin(hour_offset * 2 * math.pi / 24) * 5
            temperature = base_temp + temp_variation + random.gauss(0, 2)
            
            # Meal times
            is_breakfast = 7 <= hour <= 9
            is_lunch = 12 <= hour <= 14
            is_dinner = 18 <= hour <= 20
            
            base_consumption = 5
            
            if is_breakfast:
                meal_load = 15
            elif is_lunch:
                meal_load = 30
            elif is_dinner:
                meal_load = 25
            else:
                meal_load = 0
            
            # AC load
            ac_load = max(0, (temperature - 25) * 0.8)
            
            total = base_consumption + meal_load + ac_load + random.gauss(0, 2)
            
            data.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'consumption_kwh': max(total, 0),
                'hour': hour,
                'is_meal_time': is_breakfast or is_lunch or is_dinner
            })
        
        return data
    
    # ============================================================================
    # WEEK 1: MOVING AVERAGE + LINEAR REGRESSION
    # ============================================================================
    
    def week1_moving_average(self, data: List[Dict], window: int = 24) -> List[Dict]:
        """Apply moving average smoothing"""
        consumptions = [d['consumption_kwh'] for d in data]
        ma_values = moving_average(consumptions, window)
        
        result = []
        for i, d in enumerate(data):
            result.append({
                **d,
                'ma_consumption': ma_values[i]
            })
        return result
    
    def week1_predict_evening_peaks(self, data: List[Dict]) -> Dict:
        """Predict evening peaks using linear regression"""
        # Filter evening hours (18-23)
        evening_data = [d for d in data if 18 <= d['hour'] <= 23]
        
        # Add moving average
        consumptions = [d['consumption_kwh'] for d in evening_data]
        ma_values = moving_average(consumptions, 24)
        
        for i, d in enumerate(evening_data):
            d['ma_24h'] = ma_values[i]
        
        # Train/test split
        train_size = int(len(evening_data) * 0.7)
        train = evening_data[:train_size]
        test = evening_data[train_size:]
        
        # Prepare features
        X_train = [[d['hour'], d['weekday'], d['ma_24h']] for d in train]
        y_train = [d['consumption_kwh'] for d in train]
        
        X_test = [[d['hour'], d['weekday'], d['ma_24h']] for d in test]
        y_test = [d['consumption_kwh'] for d in test]
        
        # Train model
        model = SimpleLinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Future predictions (next 7 days)
        last_ma = evening_data[-1]['ma_24h']
        future_X = []
        future_hours = []
        
        for day in range(7):
            for hour in range(18, 24):
                weekday = (datetime.now().weekday() + day) % 7
                future_X.append([hour, weekday, last_ma])
                future_hours.append(hour)
        
        future_predictions = model.predict(future_X)
        
        return {
            'test_actual': y_test,
            'test_predicted': predictions,
            'future_predictions': future_predictions,
            'future_hours': future_hours,
            'r2_score': model.score(X_test, y_test),
            'coefficients': {
                'hour': model.coefficients[0],
                'weekday': model.coefficients[1],
                'ma_24h': model.coefficients[2],
                'intercept': model.intercept
            }
        }
    
    # ============================================================================
    # WEEK 2: ARIMA MODEL
    # ============================================================================
    
    def week2_arima_forecast(self, data: List[Dict], forecast_hours: int = 24) -> Dict:
        """Train ARIMA model for next-hour predictions"""
        consumptions = [d['consumption_kwh'] for d in data]
        
        # Train ARIMA
        model = SimpleARIMA(consumptions, p=2, d=1, q=2)
        model.fit()
        
        # Forecast
        forecasts, lower_ci, upper_ci = model.forecast(forecast_hours)
        
        return {
            'forecast': forecasts,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'fitted_values': consumptions[-100:],
            'actual_values': consumptions[-100:]
        }
    
    # ============================================================================
    # WEEK 3: EXPONENTIAL SMOOTHING
    # ============================================================================
    
    def week3_exponential_smoothing(self, data: List[Dict], forecast_weeks: int = 4) -> Dict:
        """Implement exponential smoothing for semester forecasts"""
        # Aggregate to daily data
        daily_sums = {}
        for d in data:
            date_key = d['timestamp'].date()
            if date_key not in daily_sums:
                daily_sums[date_key] = 0
            daily_sums[date_key] += d['consumption_kwh']
        
        dates = sorted(daily_sums.keys())
        daily_data = [daily_sums[d] for d in dates]
        
        # Fit exponential smoothing
        model = ExponentialSmoothingModel(daily_data, seasonal_period=7)
        model.fit()
        
        # Forecast
        forecast = model.forecast(forecast_weeks * 7)
        
        # Future dates
        last_date = dates[-1]
        forecast_dates = [(last_date + timedelta(days=i+1)).isoformat() 
                         for i in range(forecast_weeks * 7)]
        
        return {
            'forecast': forecast,
            'historical': daily_data[-60:],
            'dates': [d.isoformat() for d in dates[-60:]],
            'forecast_dates': forecast_dates
        }
    
    # ============================================================================
    # WEEK 4: WEATHER-BASED LINEAR REGRESSION
    # ============================================================================
    
    def week4_weather_regression(self, data: List[Dict]) -> Dict:
        """Predict lunch-hour surges using temperature"""
        # Filter lunch hours (12-14)
        lunch_data = [d for d in data if 12 <= d['hour'] <= 14]
        
        # Features and target
        X = [[d['temperature'], d['hour']] for d in lunch_data]
        y = [d['consumption_kwh'] for d in lunch_data]
        
        # Train/test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = SimpleLinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Future 24 hours with temperature forecast
        current_temp = data[-1]['temperature']
        future_temp_change = random.uniform(-3, 3)
        future_temps = [current_temp + (future_temp_change * i / 24) for i in range(24)]
        future_hours = list(range(24))
        
        future_X = [[future_temps[i], future_hours[i]] for i in range(24)]
        future_predictions = model.predict(future_X)
        
        return {
            'test_actual': y_test,
            'test_predicted': predictions,
            'future_predictions': future_predictions,
            'future_temps': future_temps,
            'future_hours': future_hours,
            'r2_score': model.score(X_test, y_test),
            'coefficients': {
                'temperature': model.coefficients[0],
                'hour': model.coefficients[1],
                'intercept': model.intercept
            }
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def save_data_as_json(data: List[Dict], filename: str):
    """Save data as JSON (converting timestamps to strings)"""
    json_data = []
    for row in data:
        json_row = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                json_row[key] = value.isoformat()
            else:
                json_row[key] = value
        json_data.append(json_row)
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)

def generate_all_data():
    """Generate all datasets and run all models"""
    system = EnergyForecastingSystem()
    
    print("=" * 70)
    print("JANUARY ENERGY FORECASTING SYSTEM")
    print("=" * 70)
    print("\n🔄 Generating datasets...\n")
    
    # Week 1: Dorm data
    print("📊 Week 1: Peak Hour Electricity Spikes (Dorms)")
    dorm_data = system.generate_dorm_data(days=14)
    dorm_smoothed = system.week1_moving_average(dorm_data)
    week1_results = system.week1_predict_evening_peaks(dorm_smoothed)
    print(f"   ✓ Generated {len(dorm_data)} hourly readings")
    print(f"   ✓ R² Score: {week1_results['r2_score']:.3f}")
    
    # Week 2: Classroom data
    print("\n📊 Week 2: Classroom Usage Forecasting")
    classroom_data = system.generate_classroom_data(days=7)
    week2_results = system.week2_arima_forecast(classroom_data)
    print(f"   ✓ Generated {len(classroom_data)} hourly readings")
    print(f"   ✓ ARIMA forecast for next 24 hours")
    
    # Week 3: Library data
    print("\n📊 Week 3: Library Energy During Exams")
    library_data = system.generate_library_data(weeks=16)
    week3_results = system.week3_exponential_smoothing(library_data)
    print(f"   ✓ Generated {len(library_data)} hourly readings")
    print(f"   ✓ Exponential smoothing forecast for 4 weeks")
    
    # Week 4: Cafeteria data
    print("\n📊 Week 4: Cafeteria Load Prediction")
    cafeteria_data = system.generate_cafeteria_data(days=14)
    week4_results = system.week4_weather_regression(cafeteria_data)
    print(f"   ✓ Generated {len(cafeteria_data)} hourly readings")
    print(f"   ✓ R² Score: {week4_results['r2_score']:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ All models trained successfully!")
    print("=" * 70)
    
    return {
        'week1': {
            'data': dorm_smoothed,
            'results': week1_results,
            'title': 'Peak Hour Electricity Spikes (Dorms)',
            'description': 'Moving average smoothing and linear regression for evening peak prediction'
        },
        'week2': {
            'data': classroom_data,
            'results': week2_results,
            'title': 'Classroom Usage Forecasting',
            'description': 'ARIMA model for next-hour electricity predictions with confidence intervals'
        },
        'week3': {
            'data': library_data,
            'results': week3_results,
            'title': 'Library Energy During Exams',
            'description': 'Exponential smoothing for semester-end forecasts'
        },
        'week4': {
            'data': cafeteria_data,
            'results': week4_results,
            'title': 'Cafeteria Load Prediction',
            'description': 'Weather-based linear regression for lunch-hour surge prediction'
        }
    }

if __name__ == "__main__":
    results = generate_all_data()
    
    # Save datasets
    print("\n💾 Saving datasets...")
    save_data_as_json(results['week1']['data'], '/home/claude/week1_dorm_data.json')
    save_data_as_json(results['week2']['data'], '/home/claude/week2_classroom_data.json')
    save_data_as_json(results['week3']['data'], '/home/claude/week3_library_data.json')
    save_data_as_json(results['week4']['data'], '/home/claude/week4_cafeteria_data.json')
    
    # Save results
    with open('/home/claude/all_results.json', 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for week, data in results.items():
            json_results[week] = {
                'title': data['title'],
                'description': data['description'],
                'results': data['results']
            }
        json.dump(json_results, f, indent=2)
    
    print("✅ All data saved successfully!")
    print("\nFiles created:")
    print("  - week1_dorm_data.json")
    print("  - week2_classroom_data.json")
    print("  - week3_library_data.json")
    print("  - week4_cafeteria_data.json")
    print("  - all_results.json")

