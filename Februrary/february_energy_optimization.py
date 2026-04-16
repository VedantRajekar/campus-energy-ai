"""
February Campus Energy Optimization System
Complete 6-week implementation with advanced ML models
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

def normalize(values: List[float]) -> List[float]:
    """Normalize values to 0-1 range"""
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

# ============================================================================
# DECISION TREE (Week 5)
# ============================================================================

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class SimpleDecisionTree:
    """Simple decision tree for HVAC optimization"""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None
    
    def gini_impurity(self, y: List[float]) -> float:
        """Calculate Gini impurity"""
        if not y:
            return 0
        # For regression, use variance
        return std_dev(y) ** 2
    
    def split_data(self, X: List[List[float]], y: List[float], feature: int, threshold: float):
        """Split data based on feature and threshold"""
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y
    
    def find_best_split(self, X: List[List[float]], y: List[float]):
        """Find best feature and threshold to split on"""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0])
        current_impurity = self.gini_impurity(y)
        
        for feature in range(n_features):
            # Get unique values for this feature
            values = sorted(set(row[feature] for row in X))
            
            # Try each midpoint as threshold
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                left_X, left_y, right_X, right_y = self.split_data(X, y, feature, threshold)
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # Calculate information gain
                n = len(y)
                left_impurity = self.gini_impurity(left_y)
                right_impurity = self.gini_impurity(right_y)
                
                weighted_impurity = (len(left_y) / n * left_impurity + 
                                   len(right_y) / n * right_impurity)
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X: List[List[float]], y: List[float], depth: int = 0):
        """Recursively build decision tree"""
        # Stop conditions
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < 5:
            return DecisionTreeNode(value=mean(y))
        
        # Find best split
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            return DecisionTreeNode(value=mean(y))
        
        # Split data
        left_X, left_y, right_X, right_y = self.split_data(X, y, feature, threshold)
        
        # Build subtrees
        left_node = self.build_tree(left_X, left_y, depth + 1)
        right_node = self.build_tree(right_X, right_y, depth + 1)
        
        return DecisionTreeNode(feature=feature, threshold=threshold, 
                              left=left_node, right=right_node)
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Train the decision tree"""
        self.root = self.build_tree(X, y)
        return self
    
    def predict_one(self, x: List[float], node: DecisionTreeNode) -> float:
        """Predict single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Predict multiple samples"""
        return [self.predict_one(x, self.root) for x in X]

# ============================================================================
# SIMPLE LSTM (Week 6)
# ============================================================================

class SimpleLSTM:
    """Simplified LSTM for time series prediction"""
    def __init__(self, sequence_length=24, hidden_size=10):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.weights = []
        
    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + math.exp(-max(min(x, 500), -500)))
    
    def tanh(self, x):
        """Tanh activation"""
        return math.tanh(max(min(x, 500), -500))
    
    def fit(self, data: List[float]):
        """Train LSTM (simplified - just learns patterns)"""
        # Extract patterns from data
        self.mean_val = mean(data)
        self.std_val = std_dev(data)
        
        # Learn trend
        if len(data) > self.sequence_length:
            recent = data[-self.sequence_length:]
            self.trend = (recent[-1] - recent[0]) / self.sequence_length
        else:
            self.trend = 0
        
        return self
    
    def predict(self, steps: int, last_values: List[float]) -> List[float]:
        """Generate predictions"""
        predictions = []
        
        for i in range(steps):
            # Simple LSTM-like prediction using trend and seasonality
            base = last_values[-1] if last_values else self.mean_val
            trend_component = self.trend * (i + 1)
            
            # Add some pattern learned from data
            seasonal = self.std_val * 0.2 * math.sin(i * 2 * math.pi / 24)
            
            pred = base + trend_component + seasonal
            predictions.append(pred)
            last_values.append(pred)
        
        return predictions

# ============================================================================
# K-MEANS CLUSTERING (Week 7)
# ============================================================================

class KMeansClustering:
    """K-means clustering for usage profiling"""
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.centroids = []
        self.labels = []
    
    def euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance"""
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))
    
    def fit(self, X: List[List[float]], max_iters=100):
        """Fit K-means model"""
        # Initialize centroids randomly
        self.centroids = random.sample(X, self.n_clusters)
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            labels = []
            for x in X:
                distances = [self.euclidean_distance(x, c) for c in self.centroids]
                labels.append(distances.index(min(distances)))
            
            # Update centroids
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = [X[i] for i in range(len(X)) if labels[i] == k]
                if cluster_points:
                    # Calculate mean of cluster
                    n_features = len(X[0])
                    centroid = [mean([p[j] for p in cluster_points]) for j in range(n_features)]
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(self.centroids[k])
            
            # Check convergence
            if new_centroids == self.centroids:
                break
            
            self.centroids = new_centroids
        
        self.labels = labels
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict cluster labels"""
        labels = []
        for x in X:
            distances = [self.euclidean_distance(x, c) for c in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels

# ============================================================================
# POLYNOMIAL REGRESSION (Week 8)
# ============================================================================

class PolynomialRegression:
    """Polynomial regression for non-linear patterns"""
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = []
    
    def polynomial_features(self, X: List[List[float]]) -> List[List[float]]:
        """Generate polynomial features"""
        poly_X = []
        for x in X:
            features = [1]  # bias term
            for d in range(1, self.degree + 1):
                for val in x:
                    features.append(val ** d)
            poly_X.append(features)
        return poly_X
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Fit polynomial regression using gradient descent"""
        poly_X = self.polynomial_features(X)
        n_features = len(poly_X[0])
        
        # Initialize coefficients
        self.coefficients = [0] * n_features
        
        # Gradient descent
        learning_rate = 0.001
        iterations = 1000
        
        for _ in range(iterations):
            predictions = [sum(poly_X[i][j] * self.coefficients[j] for j in range(n_features)) 
                         for i in range(len(poly_X))]
            errors = [predictions[i] - y[i] for i in range(len(y))]
            
            # Update coefficients
            for j in range(n_features):
                gradient = sum(errors[i] * poly_X[i][j] for i in range(len(poly_X))) / len(y)
                self.coefficients[j] -= learning_rate * gradient
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions"""
        poly_X = self.polynomial_features(X)
        return [sum(poly_X[i][j] * self.coefficients[j] for j in range(len(self.coefficients)))
                for i in range(len(poly_X))]

# ============================================================================
# NAIVE BAYES (Week 9)
# ============================================================================

class NaiveBayesClassifier:
    """Naive Bayes for usage categorization"""
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = {}
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Train Naive Bayes classifier"""
        # Calculate class priors
        classes = list(set(y))
        for c in classes:
            self.class_priors[c] = y.count(c) / len(y)
        
        # Calculate feature statistics for each class
        n_features = len(X[0])
        for c in classes:
            class_data = [X[i] for i in range(len(X)) if y[i] == c]
            self.feature_stats[c] = []
            
            for f in range(n_features):
                feature_values = [row[f] for row in class_data]
                self.feature_stats[c].append({
                    'mean': mean(feature_values),
                    'std': std_dev(feature_values) + 1e-6  # Add small value to avoid division by zero
                })
        
        return self
    
    def gaussian_probability(self, x: float, mean: float, std: float) -> float:
        """Calculate Gaussian probability"""
        exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    
    def predict_proba(self, x: List[float]) -> Dict[int, float]:
        """Predict class probabilities"""
        probabilities = {}
        
        for c in self.class_priors:
            # Start with prior probability
            prob = math.log(self.class_priors[c])
            
            # Multiply by feature probabilities
            for f in range(len(x)):
                stats = self.feature_stats[c][f]
                feature_prob = self.gaussian_probability(x[f], stats['mean'], stats['std'])
                prob += math.log(feature_prob + 1e-10)  # Log to avoid underflow
            
            probabilities[c] = prob
        
        # Convert log probabilities back
        max_log_prob = max(probabilities.values())
        for c in probabilities:
            probabilities[c] = math.exp(probabilities[c] - max_log_prob)
        
        # Normalize
        total = sum(probabilities.values())
        for c in probabilities:
            probabilities[c] /= total
        
        return probabilities
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict classes"""
        predictions = []
        for x in X:
            probs = self.predict_proba(x)
            predictions.append(max(probs, key=probs.get))
        return predictions

# ============================================================================
# PROPHET-LIKE MODEL (Week 9 - Part of Naive Bayes week)
# ============================================================================

class SimpleProphet:
    """Simplified Prophet-like model for time series"""
    def __init__(self):
        self.trend = 0
        self.seasonal_components = []
        self.mean_val = 0
    
    def fit(self, data: List[float], period=7):
        """Fit Prophet-like model"""
        self.mean_val = mean(data)
        
        # Calculate trend
        x = list(range(len(data)))
        n = len(data)
        x_mean = mean(x)
        y_mean = mean(data)
        
        numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        self.trend = numerator / denominator if denominator != 0 else 0
        
        # Calculate seasonal components
        detrended = [data[i] - (self.trend * i) for i in range(len(data))]
        self.seasonal_components = [0] * period
        
        for i in range(period):
            season_values = [detrended[j] for j in range(i, len(detrended), period)]
            self.seasonal_components[i] = mean(season_values)
        
        return self
    
    def predict(self, steps: int, period=7) -> List[float]:
        """Generate predictions"""
        predictions = []
        start_idx = 0
        
        for i in range(steps):
            trend_component = self.trend * i
            seasonal_component = self.seasonal_components[i % period]
            pred = self.mean_val + trend_component + seasonal_component
            predictions.append(pred)
        
        return predictions

# ============================================================================
# ENSEMBLE MODEL (Week 10)
# ============================================================================

class SimpleLinearRegression:
    """Simple linear regression for ensemble"""
    def __init__(self):
        self.coefficients = []
        self.intercept = 0
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Fit linear regression"""
        n_features = len(X[0])
        self.coefficients = [0] * n_features
        
        # Simple gradient descent
        learning_rate = 0.001
        iterations = 1000
        
        for _ in range(iterations):
            predictions = [self.intercept + sum(X[i][j] * self.coefficients[j] 
                          for j in range(n_features)) for i in range(len(X))]
            errors = [predictions[i] - y[i] for i in range(len(y))]
            
            # Update intercept
            self.intercept -= learning_rate * sum(errors) / len(y)
            
            # Update coefficients
            for j in range(n_features):
                gradient = sum(errors[i] * X[i][j] for i in range(len(X))) / len(y)
                self.coefficients[j] -= learning_rate * gradient
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions"""
        return [self.intercept + sum(X[i][j] * self.coefficients[j] 
                for j in range(len(X[i]))) for i in range(len(X))]

class EnsembleModel:
    """Ensemble of regression and smoothing models"""
    def __init__(self):
        self.regression_model = SimpleLinearRegression()
        self.smoothing_model = None
        self.weights = [0.6, 0.4]  # Weight for regression and smoothing
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Fit ensemble model"""
        # Fit regression model
        self.regression_model.fit(X, y)
        
        # Fit smoothing model (exponential smoothing)
        alpha = 0.3
        smoothed = [y[0]]
        for i in range(1, len(y)):
            smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[-1])
        
        self.smoothing_model = smoothed
        return self
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make ensemble predictions"""
        reg_predictions = self.regression_model.predict(X)
        
        # For smoothing, use last value
        smooth_pred = self.smoothing_model[-1] if self.smoothing_model else mean(reg_predictions)
        
        # Combine predictions
        return [self.weights[0] * reg_predictions[i] + self.weights[1] * smooth_pred 
                for i in range(len(reg_predictions))]

# ============================================================================
# DATA GENERATION
# ============================================================================

class FebruaryEnergySystem:
    """Complete February energy optimization system"""
    
    def __init__(self):
        self.models = {}
    
    def generate_hvac_data(self, days: int = 14) -> List[Dict]:
        """Week 5: Generate HVAC lab data"""
        data = []
        current_time = datetime.now()
        
        for day_offset in range(-days, 0):
            for hour in range(24):
                timestamp = current_time + timedelta(days=day_offset, hours=hour)
                
                # Occupancy (0-50 people)
                is_work_hours = 8 <= hour <= 18 and timestamp.weekday() < 5
                occupancy = random.randint(20, 50) if is_work_hours else random.randint(0, 5)
                
                # Temperature (20-30°C)
                base_temp = 25
                outdoor_variation = math.sin(hour * 2 * math.pi / 24) * 3
                temperature = base_temp + outdoor_variation + random.gauss(0, 1)
                
                # Cooling needs based on occupancy and temperature
                cooling_base = (temperature - 22) * 2  # More cooling needed when hot
                cooling_occupancy = occupancy * 0.3
                cooling_kwh = max(0, cooling_base + cooling_occupancy + random.gauss(0, 2))
                
                data.append({
                    'timestamp': timestamp,
                    'occupancy': occupancy,
                    'temperature': temperature,
                    'cooling_kwh': cooling_kwh,
                    'hour': hour,
                    'is_work_hours': is_work_hours
                })
        
        return data
    
    def generate_sports_facility_data(self, days: int = 30) -> List[Dict]:
        """Week 6: Generate sports facility data"""
        data = []
        current_time = datetime.now()
        
        for day_offset in range(-days, 0):
            for hour in range(24):
                timestamp = current_time + timedelta(days=day_offset, hours=hour)
                
                # Evening events (18:00-22:00)
                is_event_time = 18 <= hour <= 22
                is_weekend = timestamp.weekday() >= 5
                
                # Base consumption
                base = 10
                
                # Event load
                if is_event_time:
                    event_load = 40 + random.gauss(0, 5)
                    if is_weekend:
                        event_load *= 1.3  # More events on weekends
                else:
                    event_load = 0
                
                consumption = base + event_load + random.gauss(0, 2)
                
                data.append({
                    'timestamp': timestamp,
                    'consumption_kwh': max(consumption, 0),
                    'hour': hour,
                    'is_event_time': is_event_time,
                    'is_weekend': is_weekend
                })
        
        return data
    
    def generate_admin_building_data(self, weeks: int = 8) -> List[Dict]:
        """Week 7: Generate admin building data with weekend dips"""
        data = []
        current_time = datetime.now()
        
        total_hours = weeks * 7 * 24
        for hour_offset in range(-total_hours, 0):
            timestamp = current_time + timedelta(hours=hour_offset)
            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5
            
            # Weekday pattern
            if not is_weekend and 8 <= hour <= 18:
                base = 50 + (hour - 8) * 2
            elif not is_weekend:
                base = 15
            else:
                base = 5  # Weekend dip
            
            consumption = base + random.gauss(0, 3)
            
            data.append({
                'timestamp': timestamp,
                'consumption_kwh': max(consumption, 0),
                'hour': hour,
                'is_weekend': is_weekend,
                'day_of_week': timestamp.weekday()
            })
        
        return data
    
    def generate_parking_lot_data(self, days: int = 14) -> List[Dict]:
        """Week 8: Generate parking lot lighting data"""
        data = []
        current_time = datetime.now()
        
        for day_offset in range(-days, 0):
            for hour in range(24):
                timestamp = current_time + timedelta(days=day_offset, hours=hour)
                
                # Vehicle count (sensor data)
                is_daytime = 6 <= hour <= 20
                if is_daytime:
                    vehicles = random.randint(50, 200)
                else:
                    vehicles = random.randint(5, 30)
                
                # Light usage based on vehicles and time
                base_light = 5
                if not is_daytime:
                    base_light = 20  # Always on at night
                
                vehicle_light = vehicles * 0.05  # Motion-activated
                
                light_kwh = base_light + vehicle_light + random.gauss(0, 1)
                
                data.append({
                    'timestamp': timestamp,
                    'vehicles': vehicles,
                    'light_kwh': max(light_kwh, 0),
                    'hour': hour,
                    'is_daytime': is_daytime
                })
        
        return data
    
    def generate_hostel_laundry_data(self, days: int = 21) -> List[Dict]:
        """Week 9: Generate hostel laundry peak data"""
        data = []
        current_time = datetime.now()
        
        for day_offset in range(-days, 0):
            for hour in range(24):
                timestamp = current_time + timedelta(days=day_offset, hours=hour)
                
                # Peak times: weekends and evenings
                is_weekend = timestamp.weekday() >= 5
                is_evening = 18 <= hour <= 22
                
                # Base usage
                base = 5
                
                # Peak multipliers
                if is_weekend and is_evening:
                    usage = base * 4
                elif is_weekend or is_evening:
                    usage = base * 2.5
                else:
                    usage = base
                
                usage += random.gauss(0, 1)
                
                # Categories: light (0), medium (1), heavy (2)
                if usage > 15:
                    category = 2
                elif usage > 8:
                    category = 1
                else:
                    category = 0
                
                data.append({
                    'timestamp': timestamp,
                    'usage_kwh': max(usage, 0),
                    'hour': hour,
                    'is_weekend': is_weekend,
                    'is_evening': is_evening,
                    'category': category
                })
        
        return data
    
    def generate_campus_aggregate_data(self, days: int = 30) -> List[Dict]:
        """Week 10: Generate campus-wide aggregate data"""
        data = []
        current_time = datetime.now()
        
        for day_offset in range(-days, 0):
            for hour in range(24):
                timestamp = current_time + timedelta(days=day_offset, hours=hour)
                
                # Aggregate multiple sources
                base = 100
                
                # Time-based patterns
                hour_factor = 1 + 0.5 * math.sin((hour - 6) * 2 * math.pi / 24)
                
                # Weekly pattern
                weekday = timestamp.weekday()
                week_factor = 1.2 if weekday < 5 else 0.6
                
                # Total consumption
                total = base * hour_factor * week_factor + random.gauss(0, 10)
                
                # Calculate carbon savings (assuming renewable energy during low demand)
                carbon_factor = 0.5  # kg CO2 per kWh
                if total < 80:
                    savings = (100 - total) * carbon_factor
                else:
                    savings = 0
                
                data.append({
                    'timestamp': timestamp,
                    'total_kwh': max(total, 0),
                    'carbon_savings_kg': max(savings, 0),
                    'hour': hour,
                    'weekday': weekday
                })
        
        return data

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

    def week5_decision_tree(self, data: List[Dict]) -> Dict:
        """Train decision tree for HVAC optimization"""
        # Prepare features and target
        X = [[d['occupancy'], d['temperature'], d['hour']] for d in data]
        y = [d['cooling_kwh'] for d in data]
        
        # Train/test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = SimpleDecisionTree(max_depth=5)
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Calculate R² score
        y_mean = mean(y_test)
        ss_tot = sum((y_test[i] - y_mean) ** 2 for i in range(len(y_test)))
        ss_res = sum((y_test[i] - predictions[i]) ** 2 for i in range(len(y_test)))
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Zone-wise predictions (simulate 4 zones)
        zones = ['Lab A', 'Lab B', 'Lab C', 'Lab D']
        zone_predictions = {}
        for zone in zones:
            zone_occupancy = random.randint(10, 40)
            zone_temp = 24 + random.gauss(0, 2)
            pred = model.predict([[zone_occupancy, zone_temp, 14]])[0]
            zone_predictions[zone] = {
                'occupancy': zone_occupancy,
                'temperature': zone_temp,
                'predicted_cooling': pred
            }
        
        return {
            'test_actual': y_test,
            'test_predicted': predictions,
            'r2_score': r2_score,
            'zone_predictions': zone_predictions
        }
    
    def week6_lstm(self, data: List[Dict]) -> Dict:
        """Train LSTM for sports facility prediction"""
        consumptions = [d['consumption_kwh'] for d in data]
        
        # Train LSTM
        model = SimpleLSTM(sequence_length=24)
        model.fit(consumptions)
        
        # Predict next 48 hours
        last_24 = consumptions[-24:]
        predictions = model.predict(48, last_24.copy())
        
        # Historical for visualization
        historical = consumptions[-100:]
        
        return {
            'predictions': predictions,
            'historical': historical,
            'mean': model.mean_val,
            'trend': model.trend
        }
    
    def week7_kmeans(self, data: List[Dict]) -> Dict:
        """Apply K-means clustering to usage profiles"""
        # Create usage profiles (hour, weekday, consumption)
        X = [[d['hour'], d['day_of_week'], d['consumption_kwh']] for d in data]
        
        # Normalize features
        X_normalized = []
        for i in range(3):
            feature_vals = [x[i] for x in X]
            normalized = normalize(feature_vals)
            if i == 0:
                X_normalized = [[normalized[j]] for j in range(len(normalized))]
            else:
                for j in range(len(normalized)):
                    X_normalized[j].append(normalized[j])
        
        # Fit K-means with 3 clusters
        kmeans = KMeansClustering(n_clusters=3)
        kmeans.fit(X_normalized)
        
        # Analyze clusters
        cluster_stats = []
        for k in range(3):
            cluster_data = [data[i] for i in range(len(data)) if kmeans.labels[i] == k]
            avg_consumption = mean([d['consumption_kwh'] for d in cluster_data])
            
            # Calculate potential savings
            if avg_consumption > 30:
                savings = (avg_consumption - 25) * len(cluster_data)
            else:
                savings = 0
            
            cluster_stats.append({
                'cluster_id': k,
                'size': len(cluster_data),
                'avg_consumption': avg_consumption,
                'potential_savings': savings
            })
        
        return {
            'labels': kmeans.labels,
            'centroids': kmeans.centroids,
            'cluster_stats': cluster_stats
        }
    
    def week8_polynomial_regression(self, data: List[Dict]) -> Dict:
        """Train polynomial regression for parking lot lighting"""
        # Features and target
        X = [[d['vehicles'], d['hour']] for d in data]
        y = [d['light_kwh'] for d in data]
        
        # Train/test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = PolynomialRegression(degree=2)
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Anomaly detection (values > 2 std devs)
        pred_std = std_dev(predictions)
        pred_mean = mean(predictions)
        anomalies = []
        
        for i in range(len(predictions)):
            if abs(predictions[i] - pred_mean) > 2 * pred_std:
                anomalies.append({
                    'index': i,
                    'predicted': predictions[i],
                    'threshold': pred_mean + 2 * pred_std
                })
        
        return {
            'test_actual': y_test,
            'test_predicted': predictions,
            'anomalies': anomalies[:5]  # Top 5 anomalies
        }
    
    def week9_naive_bayes_prophet(self, data: List[Dict]) -> Dict:
        """Train Naive Bayes for categorization and Prophet for forecasting"""
        # Naive Bayes for categories
        X = [[d['hour'], int(d['is_weekend']), int(d['is_evening'])] for d in data]
        y = [d['category'] for d in data]
        
        nb_model = NaiveBayesClassifier()
        nb_model.fit(X, y)
        
        # Test predictions
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        predictions = nb_model.predict(X_test)
        
        # Accuracy
        correct = sum(1 for i in range(len(y_test)) if predictions[i] == y_test[i])
        accuracy = correct / len(y_test)
        
        # Prophet for forecasting
        consumptions = [d['usage_kwh'] for d in data]
        prophet = SimpleProphet()
        prophet.fit(consumptions, period=7)
        forecast = prophet.predict(steps=14, period=7)
        
        return {
            'nb_accuracy': accuracy,
            'nb_predictions': predictions[:20],
            'prophet_forecast': forecast,
            'historical': consumptions[-30:]
        }
    
    def week10_ensemble(self, data: List[Dict]) -> Dict:
        """Train ensemble model for campus-wide tracking"""
        # Features and target
        X = [[d['hour'], d['weekday']] for d in data]
        y = [d['total_kwh'] for d in data]
        
        # Train/test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train ensemble
        ensemble = EnsembleModel()
        ensemble.fit(X_train, y_train)
        
        # Predictions
        predictions = ensemble.predict(X_test)
        
        # KPIs
        total_consumption = sum(y)
        total_carbon_saved = sum(d['carbon_savings_kg'] for d in data)
        avg_consumption = mean(y)
        
        return {
            'test_actual': y_test,
            'test_predicted': predictions,
            'kpis': {
                'total_consumption_kwh': total_consumption,
                'total_carbon_saved_kg': total_carbon_saved,
                'avg_hourly_consumption': avg_consumption,
                'efficiency_score': min(100, (total_carbon_saved / total_consumption) * 1000)
            }
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def save_data_as_json(data: List[Dict], filename: str):
    """Save data as JSON"""
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

def generate_all_february_data():
    """Generate all datasets and train all models"""
    system = FebruaryEnergySystem()
    
    print("=" * 70)
    print("FEBRUARY CAMPUS ENERGY OPTIMIZATION SYSTEM")
    print("=" * 70)
    print("\n🔄 Generating datasets and training models...\n")
    
    # Week 5: HVAC Decision Tree
    print("📊 Week 5: HVAC Optimization in Labs")
    hvac_data = system.generate_hvac_data(days=14)
    week5_results = system.week5_decision_tree(hvac_data)
    print(f"   ✓ Generated {len(hvac_data)} hourly readings")
    print(f"   ✓ Decision Tree R² Score: {week5_results['r2_score']:.3f}")
    
    # Week 6: LSTM for Sports Facility
    print("\n📊 Week 6: Sports Facility Night Usage")
    sports_data = system.generate_sports_facility_data(days=30)
    week6_results = system.week6_lstm(sports_data)
    print(f"   ✓ Generated {len(sports_data)} hourly readings")
    print(f"   ✓ LSTM predictions for next 48 hours")
    
    # Week 7: K-means Clustering
    print("\n📊 Week 7: Admin Building Weekend Dip")
    admin_data = system.generate_admin_building_data(weeks=8)
    week7_results = system.week7_kmeans(admin_data)
    print(f"   ✓ Generated {len(admin_data)} hourly readings")
    print(f"   ✓ Identified {len(week7_results['cluster_stats'])} usage clusters")
    
    # Week 8: Polynomial Regression
    print("\n📊 Week 8: Parking Lot Lighting Forecast")
    parking_data = system.generate_parking_lot_data(days=14)
    week8_results = system.week8_polynomial_regression(parking_data)
    print(f"   ✓ Generated {len(parking_data)} hourly readings")
    print(f"   ✓ Detected {len(week8_results['anomalies'])} anomalies")
    
    # Week 9: Naive Bayes + Prophet
    print("\n📊 Week 9: Hostel Laundry Peak Prediction")
    laundry_data = system.generate_hostel_laundry_data(days=21)
    week9_results = system.week9_naive_bayes_prophet(laundry_data)
    print(f"   ✓ Generated {len(laundry_data)} hourly readings")
    print(f"   ✓ Naive Bayes Accuracy: {week9_results['nb_accuracy']:.3f}")
    
    # Week 10: Ensemble Model
    print("\n📊 Week 10: Campus-Wide Sustainability Tracker")
    campus_data = system.generate_campus_aggregate_data(days=30)
    week10_results = system.week10_ensemble(campus_data)
    print(f"   ✓ Generated {len(campus_data)} hourly readings")
    print(f"   ✓ Carbon saved: {week10_results['kpis']['total_carbon_saved_kg']:.1f} kg")
    
    print("\n" + "=" * 70)
    print("✅ All models trained successfully!")
    print("=" * 70)
    
    return {
        'week5': {
            'data': hvac_data,
            'results': week5_results,
            'title': 'HVAC Optimization in Labs',
            'description': 'Decision tree on occupancy/temperature data to forecast cooling needs'
        },
        'week6': {
            'data': sports_data,
            'results': week6_results,
            'title': 'Sports Facility Night Usage',
            'description': 'RNN (simple LSTM) on hourly patterns to predict post-event electricity'
        },
        'week7': {
            'data': admin_data,
            'results': week7_results,
            'title': 'Admin Building Weekend Dip',
            'description': 'K-means clustering on usage profiles then regression for forecasts'
        },
        'week8': {
            'data': parking_data,
            'results': week8_results,
            'title': 'Parking Lot Lighting Forecast',
            'description': 'Polynomial regression for light usage with anomaly detection'
        },
        'week9': {
            'data': laundry_data,
            'results': week9_results,
            'title': 'Hostel Laundry Peak Prediction',
            'description': 'Naive Bayes for categorization and Prophet for forecasting'
        },
        'week10': {
            'data': campus_data,
            'results': week10_results,
            'title': 'Campus-Wide Sustainability Tracker',
            'description': 'Ensemble model with comprehensive KPI dashboard'
        }
    }

if __name__ == "__main__":
    results = generate_all_february_data()
    
    # Save datasets
    print("\n💾 Saving datasets...")
    save_data_as_json(results['week5']['data'], '/home/claude/week5_hvac_data.json')
    save_data_as_json(results['week6']['data'], '/home/claude/week6_sports_data.json')
    save_data_as_json(results['week7']['data'], '/home/claude/week7_admin_data.json')
    save_data_as_json(results['week8']['data'], '/home/claude/week8_parking_data.json')
    save_data_as_json(results['week9']['data'], '/home/claude/week9_laundry_data.json')
    save_data_as_json(results['week10']['data'], '/home/claude/week10_campus_data.json')
    
    # Save results
    with open('/home/claude/february_results.json', 'w') as f:
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
    print("  - week5_hvac_data.json")
    print("  - week6_sports_data.json")
    print("  - week7_admin_data.json")
    print("  - week8_parking_data.json")
    print("  - week9_laundry_data.json")
    print("  - week10_campus_data.json")
    print("  - february_results.json")
