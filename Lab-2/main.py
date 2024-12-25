# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Функция для поворота вектора на заданный угол
def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


# Параметры моделирования
START_VALUE = 0
END_VALUE = 45
STEPS = 1000
R = 5  # Радиус кольца
L = 3  # Расстояние от точки C точки A/B


r = np.sqrt(R * R - L * L)  # Расстояние от O до C

# Создание массива времени для моделирования
T = np.linspace(START_VALUE, END_VALUE, STEPS)

# Определение функций углов поворота
phi = 0.3 * T
ksi = 0.5 * T

# Вычисление координат точки O (центр окружности)
X_O = -R * np.sin(phi)  # -R из-за инвертированной оси
Y_O = R * np.cos(phi)

# Вычисление координат точки C (точка на окружности)
X_C = X_O - r * np.sin(ksi)
Y_C = Y_O + r * np.cos(ksi)

# Вычисление относительных координат точки C относительно точки O
X_REL = -r * np.sin(ksi)
Y_REL = r * np.cos(ksi)

# Настройка графика
fig = plt.figure(figsize=(8, 8))
fig.canvas.manager.set_window_title('Анимация динамической системы | Вариант 23')
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')  # Одинаковый масштаб по осям
ax1.set(xlim=[-12, 12], ylim=[-12, 12])
ax1.invert_xaxis()  # Инвертирования оси OX
ax1.invert_yaxis()  # Инвертирования оси OY

# Инициализация графических объектов для анимации
PointO1, = ax1.plot([0], [0], 'bo')
Circ_Angle = np.linspace(0, 2 * np.pi, 100)
Circ, = ax1.plot(X_O[0] + R * np.cos(Circ_Angle), Y_O[0] + R * np.sin(Circ_Angle), 'g')

AB_Line_X = np.array([0, 0, 0])
AB_Line_Y = np.array([-L, 0, L])
R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(AB_Line_X, AB_Line_Y, np.atan2(Y_REL[0], X_REL[0]))
Stick_Arrow, = ax1.plot(R_Stick_ArrowX + X_C[0], R_Stick_ArrowY + Y_C[0])  # Отрезок AB
O1O, = ax1.plot([0, X_O[0]], [0, Y_O[0]], 'b:')  # Отрезок O1-O
OC, = ax1.plot([X_O[0], X_C[0]], [Y_O[0], Y_C[0]], 'b:')  # Отрезок OC


# Функция анимации
def animateFunction(i):
    O1O.set_data([0, X_O[i]], [0, Y_O[i]])  # Обновление отрезка O1-O
    OC.set_data([X_O[i], X_C[i]], [Y_O[i], Y_C[i]])  # Обновление отрезка OC
    Circ.set_data(X_O[i] + R * np.cos(Circ_Angle), Y_O[i] + R * np.sin(Circ_Angle))  # Обновление кольца
    R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(AB_Line_X, AB_Line_Y, np.atan2(Y_REL[i], X_REL[i]))  # Поворот отрезка AB
    Stick_Arrow.set_data(R_Stick_ArrowX + X_C[i], R_Stick_ArrowY + Y_C[i])  # Обновление отрезка AB

    return O1O, OC, Circ, Stick_Arrow


# Создание анимации
anim = FuncAnimation(fig, animateFunction, frames=STEPS, interval=30)

# Отображение графика
plt.show()
