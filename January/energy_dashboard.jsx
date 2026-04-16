import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Import the generated data
const week1Data = require('./week1_dorm_data.json');
const week2Data = require('./week2_classroom_data.json');
const week3Data = require('./week3_library_data.json');
const week4Data = require('./week4_cafeteria_data.json');
const allResults = require('./all_results.json');

const EnergyDashboard = () => {
  const [activeWeek, setActiveWeek] = useState(1);
  const [liveData, setLiveData] = useState([]);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Simulate real-time updates for Week 4
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
      
      // Simulate new data point
      if (activeWeek === 4) {
        const lastPoint = week4Data[week4Data.length - 1];
        const newPoint = {
          ...lastPoint,
          timestamp: new Date().toISOString(),
          consumption_kwh: lastPoint.consumption_kwh + (Math.random() - 0.5) * 5
        };
        setLiveData(prev => [...prev.slice(-50), newPoint]);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeWeek]);

  // Week navigation tabs
  const weeks = [
    { id: 1, title: 'Dorms - Peak Hours', icon: '🏢' },
    { id: 2, title: 'Classrooms - ARIMA', icon: '🎓' },
    { id: 3, title: 'Library - Exams', icon: '📚' },
    { id: 4, title: 'Cafeteria - Weather', icon: '🍽️' }
  ];

  // Week 1: Dorm Peak Hour Dashboard
  const Week1Dashboard = () => {
    const results = allResults.week1.results;
    const eveningData = week1Data
      .filter(d => d.hour >= 18 && d.hour <= 23)
      .slice(-50)
      .map((d, i) => ({
        hour: `${d.hour}:00`,
        actual: d.consumption_kwh,
        smoothed: d.ma_consumption,
        index: i
      }));

    const predictionData = results.future_hours.map((hour, i) => ({
      hour: `${hour}:00`,
      predicted: results.future_predictions[i]
    }));

    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-amber-500 to-orange-600 p-8 rounded-2xl text-white">
          <h2 className="text-3xl font-bold mb-2">{allResults.week1.title}</h2>
          <p className="text-amber-100">{allResults.week1.description}</p>
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">R² Score</div>
              <div className="text-2xl font-bold">{results.r2_score.toFixed(3)}</div>
            </div>
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Data Points</div>
              <div className="text-2xl font-bold">{week1Data.length}</div>
            </div>
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Peak Hours</div>
              <div className="text-2xl font-bold">18:00-23:00</div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-2 gap-6">
          {/* Historical with Moving Average */}
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">Historical Evening Peaks</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={eveningData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="hour" stroke="#666" />
                <YAxis stroke="#666" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
                />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#f59e0b" strokeWidth={2} dot={false} name="Actual" />
                <Line type="monotone" dataKey="smoothed" stroke="#10b981" strokeWidth={2} dot={false} name="Moving Avg" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Future Predictions */}
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">7-Day Evening Peak Forecast</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={predictionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="hour" stroke="#666" />
                <YAxis stroke="#666" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
                />
                <Legend />
                <Bar dataKey="predicted" fill="#f59e0b" name="Predicted kWh" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Coefficients */}
        <div className="bg-white p-6 rounded-xl shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Linear Regression Coefficients</h3>
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center p-4 bg-amber-50 rounded-lg">
              <div className="text-sm text-gray-600">Hour</div>
              <div className="text-xl font-bold text-amber-600">{results.coefficients.hour.toFixed(3)}</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-sm text-gray-600">Weekday</div>
              <div className="text-xl font-bold text-orange-600">{results.coefficients.weekday.toFixed(3)}</div>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-lg">
              <div className="text-sm text-gray-600">Moving Avg</div>
              <div className="text-xl font-bold text-yellow-600">{results.coefficients.ma_24h.toFixed(3)}</div>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-lg">
              <div className="text-sm text-gray-600">Intercept</div>
              <div className="text-xl font-bold text-red-600">{results.coefficients.intercept.toFixed(3)}</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Week 2: Classroom ARIMA Dashboard
  const Week2Dashboard = () => {
    const results = allResults.week2.results;
    
    const forecastData = results.forecast.map((val, i) => ({
      hour: `Hour ${i + 1}`,
      forecast: val,
      lower: results.lower_ci[i],
      upper: results.upper_ci[i]
    }));

    const historicalData = results.actual_values.slice(-48).map((val, i) => ({
      hour: `H-${48 - i}`,
      actual: val,
      fitted: results.fitted_values[i]
    }));

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 p-8 rounded-2xl text-white">
          <h2 className="text-3xl font-bold mb-2">{allResults.week2.title}</h2>
          <p className="text-blue-100">{allResults.week2.description}</p>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Forecast Horizon</div>
              <div className="text-2xl font-bold">24 Hours</div>
            </div>
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Model Type</div>
              <div className="text-2xl font-bold">ARIMA(2,1,2)</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6">
          {/* ARIMA Forecast with CI */}
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">24-Hour Forecast with Confidence Intervals</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={forecastData}>
                <defs>
                  <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="hour" stroke="#666" />
                <YAxis stroke="#666" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
                />
                <Legend />
                <Area type="monotone" dataKey="upper" stroke="none" fill="#93c5fd" fillOpacity={0.3} name="Upper CI" />
                <Area type="monotone" dataKey="lower" stroke="none" fill="#93c5fd" fillOpacity={0.3} name="Lower CI" />
                <Line type="monotone" dataKey="forecast" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} name="Forecast" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Historical Fit */}
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">Model Fit (Last 48 Hours)</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="hour" stroke="#666" />
                <YAxis stroke="#666" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
                />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#6366f1" strokeWidth={2} dot={false} name="Actual" />
                <Line type="monotone" dataKey="fitted" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Fitted" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };

  // Week 3: Library Exponential Smoothing Dashboard
  const Week3Dashboard = () => {
    const results = allResults.week3.results;
    
    const historicalData = results.dates.map((date, i) => ({
      date: new Date(date).toLocaleDateString(),
      consumption: results.historical[i]
    }));

    const forecastData = results.forecast_dates.map((date, i) => ({
      date: new Date(date).toLocaleDateString(),
      forecast: results.forecast[i]
    }));

    const combinedData = [
      ...historicalData.slice(-30).map(d => ({ ...d, type: 'historical' })),
      ...forecastData.map(d => ({ ...d, type: 'forecast' }))
    ];

    // Calculate gauge value (current vs predicted)
    const currentAvg = results.historical.slice(-7).reduce((a, b) => a + b, 0) / 7;
    const futureAvg = results.forecast.slice(0, 7).reduce((a, b) => a + b, 0) / 7;
    const changePercent = ((futureAvg - currentAvg) / currentAvg * 100).toFixed(1);

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-purple-500 to-pink-600 p-8 rounded-2xl text-white">
          <h2 className="text-3xl font-bold mb-2">{allResults.week3.title}</h2>
          <p className="text-purple-100">{allResults.week3.description}</p>
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Current Avg (7d)</div>
              <div className="text-2xl font-bold">{currentAvg.toFixed(1)} kWh</div>
            </div>
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Predicted Avg</div>
              <div className="text-2xl font-bold">{futureAvg.toFixed(1)} kWh</div>
            </div>
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Change</div>
              <div className={`text-2xl font-bold ${changePercent > 0 ? 'text-red-200' : 'text-green-200'}`}>
                {changePercent > 0 ? '+' : ''}{changePercent}%
              </div>
            </div>
          </div>
        </div>

        {/* Gauge Visualization */}
        <div className="bg-white p-6 rounded-xl shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Energy Demand Gauge</h3>
          <div className="flex justify-center items-center h-48">
            <div className="relative w-64 h-32">
              {/* Gauge background */}
              <svg viewBox="0 0 200 100" className="w-full h-full">
                <path
                  d="M 10 90 A 90 90 0 0 1 190 90"
                  fill="none"
                  stroke="#e5e7eb"
                  strokeWidth="20"
                />
                <path
                  d="M 10 90 A 90 90 0 0 1 190 90"
                  fill="none"
                  stroke={changePercent > 10 ? '#ef4444' : changePercent > 0 ? '#f59e0b' : '#10b981'}
                  strokeWidth="20"
                  strokeDasharray={`${Math.min(Math.abs(changePercent) * 3, 283)} 283`}
                />
                <circle cx="100" cy="90" r="8" fill="#374151" />
                <line
                  x1="100"
                  y1="90"
                  x2={100 + 70 * Math.cos((Math.PI / 180) * (180 - Math.min(Math.abs(changePercent) * 9, 180)))}
                  y2={90 - 70 * Math.sin((Math.PI / 180) * (180 - Math.min(Math.abs(changePercent) * 9, 180)))}
                  stroke="#374151"
                  strokeWidth="3"
                />
              </svg>
              <div className="absolute bottom-0 left-0 right-0 text-center">
                <div className="text-3xl font-bold text-gray-800">{changePercent}%</div>
                <div className="text-sm text-gray-600">Predicted Change</div>
              </div>
            </div>
          </div>
        </div>

        {/* Time Series Chart */}
        <div className="bg-white p-6 rounded-xl shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Daily Energy Consumption Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={combinedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#666" />
              <YAxis stroke="#666" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="consumption" 
                stroke="#a855f7" 
                strokeWidth={2} 
                dot={false} 
                name="Historical"
                connectNulls
              />
              <Line 
                type="monotone" 
                dataKey="forecast" 
                stroke="#ec4899" 
                strokeWidth={2} 
                strokeDasharray="5 5"
                dot={false} 
                name="Forecast"
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  // Week 4: Cafeteria Weather-Based Dashboard with Live Updates
  const Week4Dashboard = () => {
    const results = allResults.week4.results;
    
    const testData = results.test_actual.map((val, i) => ({
      index: i,
      actual: val,
      predicted: results.test_predicted[i]
    }));

    const futureData = results.future_hours.map((hour, i) => ({
      hour: `${hour}:00`,
      predicted: results.future_predictions[i],
      temperature: results.future_temps[i]
    }));

    // Live update simulation
    const displayData = liveData.length > 0 
      ? liveData.map((d, i) => ({
          time: new Date(d.timestamp).toLocaleTimeString(),
          consumption: d.consumption_kwh,
          temperature: d.temperature
        }))
      : futureData.slice(-20);

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-8 rounded-2xl text-white">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-3xl font-bold mb-2">{allResults.week4.title}</h2>
              <p className="text-green-100">{allResults.week4.description}</p>
            </div>
            <div className="flex items-center space-x-2 bg-white/20 rounded-lg px-4 py-2 backdrop-blur">
              <div className="w-3 h-3 bg-red-400 rounded-full animate-pulse"></div>
              <span className="font-semibold">LIVE</span>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">R² Score</div>
              <div className="text-2xl font-bold">{results.r2_score.toFixed(3)}</div>
            </div>
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Temp Coefficient</div>
              <div className="text-2xl font-bold">{results.coefficients.temperature.toFixed(3)}</div>
            </div>
            <div className="bg-white/20 rounded-lg p-4 backdrop-blur">
              <div className="text-sm opacity-90">Current Time</div>
              <div className="text-lg font-bold">{currentTime.toLocaleTimeString()}</div>
            </div>
          </div>
        </div>

        {/* Live Chart */}
        <div className="bg-white p-6 rounded-xl shadow-lg border-2 border-green-200">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Real-Time Load Monitoring</h3>
            <div className="flex items-center space-x-2 text-green-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium">Updating every 2s</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey={liveData.length > 0 ? "time" : "hour"} stroke="#666" />
              <YAxis stroke="#666" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey={liveData.length > 0 ? "consumption" : "predicted"} 
                stroke="#10b981" 
                strokeWidth={3} 
                dot={{ r: 3 }} 
                name="Consumption (kWh)"
                animationDuration={500}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Temperature Correlation */}
        <div className="bg-white p-6 rounded-xl shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Temperature Impact on Energy</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={futureData}>
              <defs>
                <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="hour" stroke="#666" />
              <YAxis yAxisId="left" stroke="#10b981" />
              <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
              />
              <Legend />
              <Area 
                yAxisId="right"
                type="monotone" 
                dataKey="temperature" 
                stroke="#f59e0b" 
                fillOpacity={1} 
                fill="url(#colorTemp)" 
                name="Temperature (°C)"
              />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="predicted" 
                stroke="#10b981" 
                strokeWidth={3} 
                name="Energy (kWh)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Model Performance */}
        <div className="bg-white p-6 rounded-xl shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Model Validation</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={testData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="index" stroke="#666" />
              <YAxis stroke="#666" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
              />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#6366f1" strokeWidth={2} dot={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Campus Energy Forecasting
              </h1>
              <p className="text-gray-600 mt-1">January 2024 - Complete 4-Week Model</p>
            </div>
            <div className="flex items-center space-x-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-full shadow-lg">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span className="font-semibold">All Systems Active</span>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex space-x-3 bg-white p-2 rounded-xl shadow-md">
          {weeks.map(week => (
            <button
              key={week.id}
              onClick={() => setActiveWeek(week.id)}
              className={`flex-1 px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                activeWeek === week.id
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg transform scale-105'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <div className="text-2xl mb-1">{week.icon}</div>
              <div className="text-sm">Week {week.id}</div>
              <div className="text-xs opacity-80">{week.title}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 pb-12">
        {activeWeek === 1 && <Week1Dashboard />}
        {activeWeek === 2 && <Week2Dashboard />}
        {activeWeek === 3 && <Week3Dashboard />}
        {activeWeek === 4 && <Week4Dashboard />}
      </div>

      {/* Footer */}
      <div className="bg-gray-900 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-gray-400">
            January Energy Forecasting System • Built with React & Recharts • 2024
          </p>
          <div className="mt-4 flex justify-center space-x-6 text-sm text-gray-500">
            <span>✓ Moving Average</span>
            <span>✓ Linear Regression</span>
            <span>✓ ARIMA</span>
            <span>✓ Exponential Smoothing</span>
            <span>✓ Real-time Updates</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnergyDashboard;
