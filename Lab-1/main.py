# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

# Параметры моделирования
START_VALUE = 0
END_VALUE = 10
STEPS = 1001
SCALE_FACTOR = .8  # коэффициент масштабирования графика
SCALE_FACTOR_VA = .1  # коэффициент масштабирования вектора скорости и ускорения
ARROW_WIDTH, ARROW_HEIGHT = 0.2 * SCALE_FACTOR, 0.14 * SCALE_FACTOR  # парамерты ширины и высоты стрелок


# Функция для поворота вектора на заданный угол
def rot2D(X, Y, Alpha):
    RotX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RotY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RotX, RotY


# Определение параметрической функций радиус-вектора материальной точки от времени
def r(t: float) -> float:
    return 2 + sp.sin(12 * t)


# Определение параметрической функций угла материальной точки от времени
def phi(t: float) -> float:
    return t + 0.2 * sp.cos(13 * t)


# Определение символьной переменной
t = sp.Symbol('t')

# Преобразование полярных координат в декартовы

x = r(t) * sp.cos(phi(t)) * SCALE_FACTOR
y = r(t) * sp.sin(phi(t)) * SCALE_FACTOR

# Вычисление скорости (первая производная по времени)
Vx = sp.diff(x, t)  # проекция на x
Vy = sp.diff(y, t)  # проекция на y

# Вычисление ускорения (вторая производная по времени)
Ax = sp.diff(Vx, t)  # проекция на x
Ay = sp.diff(Vy, t)  # проекция на y

# Преобразование sympy выражений в функции numpy для численного вычисления
F_x = sp.lambdify(t, x, "numpy")
F_y = sp.lambdify(t, y, "numpy")
F_Vx = sp.lambdify(t, Vx, "numpy")
F_Vy = sp.lambdify(t, Vy, "numpy")
F_Ax = sp.lambdify(t, Ax, "numpy")
F_Ay = sp.lambdify(t, Ay, "numpy")

# Создание массива времени для моделирования
T = np.linspace(START_VALUE, END_VALUE, STEPS)

# Вычисление координат для каждого момента времени
X = F_x(T)
Y = F_y(T)

# Вычисление x и y компонент скоростей для каждого момента времени
VX = F_Vx(T)
VY = F_Vy(T)


# Вычисление x и y компонент ускорения для каждого момента времени
AX = F_Ax(T)
AY = F_Ay(T)


# Настройка графика
figure = plt.figure(figsize=(10, 10))
figure.canvas.manager.set_window_title('Вариант 13')
ax = figure.add_subplot(1, 1, 1)
ax.axis('equal')  # Одинаковый масштаб по осям
ax.set(xlim=[-12, 12], ylim=[-12, 12])

# Построение траектории
ax.plot(X, Y)

# Инициализация графических объектов для анимации
P, = ax.plot(X[0], Y[0], marker='o')
V_line, = ax.plot([X[0], X[0] + SCALE_FACTOR_VA * VX[0]], [Y[0], Y[0] + SCALE_FACTOR_VA * VY[0]], label="Вектор скорости",
                  color='r')
A_line, = ax.plot([X[0], X[0] + SCALE_FACTOR_VA * AX[0]], [Y[0], Y[0] + SCALE_FACTOR_VA * AY[0]], label="Вектор ускорения",
                  color='g')
Radius_vector_line, = ax.plot([0, X[0]], [0, Y[0]], label="Радиус-вектор", color='b')

# Настройка легенды графика
ax.legend(loc='lower left')

# Создание начальных стрелок для векторов
AlphaVArrow = np.arctan2(VY, VX)
ArrowVX = np.array([-ARROW_WIDTH, 0, -ARROW_WIDTH])
ArrowVY = np.array([ARROW_HEIGHT, 0, -ARROW_HEIGHT])
RotX, RotY = rot2D(ArrowVX, ArrowVY, AlphaVArrow[0])
V_Arrow, = ax.plot(X[0] + SCALE_FACTOR_VA * VX[0] + RotX, Y[0] + SCALE_FACTOR_VA * VY[0] + RotY,
                   color=[1, 0, 0])  # стрелка скорости

AlphaRArrow = np.arctan2(Y, X)
ArrowRX = np.array([-ARROW_WIDTH, 0, -ARROW_WIDTH])
ArrowRY = np.array([ARROW_HEIGHT, 0, -ARROW_HEIGHT])
RotX, RotY = rot2D(ArrowRX, ArrowRY, AlphaRArrow[0])
Radius_vector_Arrow, = ax.plot(X[0] + RotX, Y[0] + RotY, color='b')  # стрелка радиус-вектора

AlphaAccArrow = np.arctan2(AY, AX)
ArrowAX = np.array([-ARROW_WIDTH, 0, -ARROW_WIDTH])
ArrowAY = np.array([ARROW_HEIGHT, 0, -ARROW_HEIGHT])
RotX, RotY = rot2D(ArrowAX, ArrowAY, AlphaAccArrow[0])
A_Arrow, = ax.plot(X[0] + SCALE_FACTOR_VA * AX[0] + RotX, Y[0] + SCALE_FACTOR_VA * AY[0] + RotY, color='g')  # стрелка ускорения


# Функция анимации
def animateFunction(i):
    # Обновление положения точки
    P.set_data([X[i]], [Y[i]])
    V_line.set_data([X[i], X[i] + SCALE_FACTOR_VA * VX[i]], [Y[i], Y[i] + SCALE_FACTOR_VA * VY[i]])  # Обновление вектора скорости
    A_line.set_data([X[i], X[i] + SCALE_FACTOR_VA * AX[i]], [Y[i], Y[i] + SCALE_FACTOR_VA * AY[i]])  # Обновление вектора ускорения
    Radius_vector_line.set_data([0, X[i]], [0, Y[i]])  # Обновление радиус-вектора

    # Обновление стрелок для скорости, ускорения и радиус-вектора
    RotX, RotY = rot2D(ArrowVX, ArrowVY, AlphaVArrow[i])
    V_Arrow.set_data(X[i] + SCALE_FACTOR_VA * VX[i] + RotX, Y[i] + SCALE_FACTOR_VA * VY[i] + RotY)

    RotX, RotY = rot2D(ArrowAX, ArrowAY, AlphaAccArrow[i])
    A_Arrow.set_data(X[i] + SCALE_FACTOR_VA * AX[i] + RotX, Y[i] + SCALE_FACTOR_VA * AY[i] + RotY)

    RotX, RotY = rot2D(ArrowRX, ArrowRY, AlphaRArrow[i])
    Radius_vector_Arrow.set_data(X[i] + RotX, Y[i] + RotY)
    return P, V_line, V_Arrow, A_line, Radius_vector_line, A_Arrow, Radius_vector_Arrow


# Создание анимации
animation = FuncAnimation(figure, animateFunction, frames=STEPS, interval=1)

plt.title("Симуляция движения точки")  # Установка заголовка графика
plt.show()  # Отображение графика
