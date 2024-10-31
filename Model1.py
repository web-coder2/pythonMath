import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
     "N": [1, 2, 3, 4, 5, 6], 
     "h(m)": [1, 1.2, 1.4, 1.6, 1.8, 2], 
     "t(s)": [0.45, 0.5, 0.53, 0.56, 0.6, 0.62],
     "t^2(s^2)": [0.2, 0.25, 0.28, 0.31, 0.36, 0.39],
     "g+delta(g)": [9.21, 9.07, 9.43, 10.27, 9.85, 9.57],
     "delta(g)": [0.55, 0.63, 0.45, 0.03, 0.24, 0.38],
     "g": [9.77, 9.7, 9.87, 10.3, 10.08, 10.32]
}

pd_data = pd.DataFrame(data)
print(pd_data)

# Создаем модель линейной регрессии
model = LinearRegression()
model.fit(pd_data[["h(m)"]], pd_data["t^2(s^2)"])

# Получаем коэффициенты регрессии
slope = model.coef_[0]
intercept = model.intercept_

# Выводим уравнение прямой
print(f"Уравнение прямой: t^2 = {slope:.3f} * h + {intercept:.3f}")

# Строим график с прямой
plt.figure(figsize=(8, 6))
plt.plot(pd_data["h(m)"], pd_data["t^2(s^2)"], "o-", label="Экспериментальные данные")
plt.plot(pd_data["h(m)"], model.predict(pd_data[["h(m)"]]), "r-", label="Линейная регрессия")
plt.xlabel("h (м)")
plt.ylabel("t^2 (с^2)")
plt.title("Зависимость времени падения от высоты")
plt.grid(True)
plt.legend()
plt.show()

# Оценка стандартной ошибки коэффициента наклона
mse = mean_squared_error(pd_data["t^2(s^2)"], model.predict(pd_data[["h(m)"]]))
std_err_slope = mse ** 0.5 / (len(pd_data["h(m)"]) - 2) ** 0.5

# Оценка погрешности для g
std_err_g = 2 * std_err_slope / slope ** 2

# Вывод результата
print(f"g = {2/slope:.3f} ± {std_err_g:.3f} м/с²")

g_theoretical = 9.81


relative_error = abs(g_theoretical - 2/slope) / g_theoretical * 100
print(f"Относительная погрешность: {relative_error:.2f}%")