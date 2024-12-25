# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Параметры моделирования и начальные состояния системы
START_VALUE = 0
END_VALUE = 100
STEPS = 1000
M1 = 2  # масса кольца
M2 = 1  # масса стержня
R = 0.5  # радиус кольца
L = 0.25  # половина длины стержня
M = 10  # Амлитуда момента
GAMMA = 3 * np.pi / 2  # Угловая частота
K = 10  # коэффициент сопротивления
G = 9.81  # ускорение свободного падения
PHI0 = 0
PSI0 = np.pi / 6
DPHI0 = 0
DPSI0 = 0


# Функция для поворота вектора на заданный угол
def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


def EqOfMovement(y, t, m1, m2, R, l, M, gamma, k, g):
    dy = np.zeros_like(y)
    dy[0] = y[2]  # Тривиальные решения вида dphi=phi', dpsi=psi'
    dy[1] = y[3]

    # Коэффициенты первого уравнения
    a11 = (2 * m1 + m2) * (R ** 2)
    a12 = m2 * R * np.sqrt(R * R - l * l) * np.cos(y[1] - y[0])
    b1 = M * np.sin(gamma * t) - k * y[2] + m2 * R * np.sqrt(R * R - l * l) * (y[3] ** 2) * np.sin(y[1] - y[0]) - (
            m1 + m2) * g * R * np.sin(y[0])

    # Коэффициенты второго уравнения
    a21 = np.sqrt(R * R - l * l) * R * np.cos(y[1] - y[0])
    a22 = (R ** 2) - ((2 / 3) * (l ** 2))
    b2 = -g * np.sin(y[1]) * np.sqrt(R * R - l * l) - (y[2] ** 2) * np.sin(y[1] - y[0]) * np.sqrt(R * R - l * l) * R

    # Вычисление вторых производных методом Крамера
    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a21 * a12)
    dy[3] = (a11 * b2 - a21 * b1) / (a11 * a22 - a21 * a12)
    return dy


Y0 = [PHI0, PSI0, DPHI0, DPSI0]
T = np.linspace(START_VALUE, END_VALUE, STEPS)

Y = odeint(EqOfMovement, Y0, T, (M1, M2, R, L, M, GAMMA, K, G))

phi = Y[:, 0]
psi = Y[:, 1]
dphi = Y[:, 2]
dpsi = Y[:, 3]

# Вычисление вторых производных в каждой точке времени
dY = np.zeros_like(Y)
for i, t in enumerate(T):
    dY[i, :] = EqOfMovement(Y[i, :], t, M1, M2, R, L, M, GAMMA, K, G)
ddphi = Y[:, 2]
ddpsi = Y[:, 3]

# Вычисление реакции шарнира
N_x = -(M1 + M2) * R * (ddphi * np.cos(phi) + (dphi ** 2) * np.sin(phi)) - M2 * np.sqrt(R * R - L * L) * (
        ddpsi * np.cos(psi) - (dpsi ** 2) * np.sin(psi))
N_y = -(M1 + M2) * (R * (ddphi * np.sin(phi) + (dphi ** 2) * np.cos(phi)) + G) - M2 * np.sqrt(R * R - L * L) * (
        ddpsi * np.sin(psi) + (dpsi ** 2) * np.cos(psi))
N_t = np.sqrt((N_x ** 2) + (N_y ** 2))

# Настройка графика
fig = plt.figure(figsize=[13, 9])
fig.canvas.manager.set_window_title('Симуляция динамической системы | Вариант 23')

# Подграфик зависимости phi от времени
ax1 = fig.add_subplot(2, 3, 1)
ax1.set(xlim=[START_VALUE, END_VALUE], ylim=[min(phi) / 1.1, max(phi) * 1.1], xlabel=r'$t$', ylabel=r'$\phi$')
ax1.grid(True)
PhiT, = ax1.plot([], [], label=r"$\phi$")
ax1.set_title(r'$\phi(t)$')

# Подграфик зависимости psi от времени
ax2 = fig.add_subplot(2, 3, 2)
ax2.set(xlim=[START_VALUE, END_VALUE], ylim=[min(psi) / 1.1, max(psi) * 1.1], xlabel=r'$t$', ylabel=r'$\psi$')
ax2.grid(True)
KsiT, = ax2.plot([], [], label=r"$\psi$")
ax2.set_title(r'$\psi(t)$')

# Подграфик зависимости реакции шарнира от времени
ax3 = fig.add_subplot(2, 3, 3)
ax3.set(xlim=[START_VALUE, END_VALUE], ylim=[min(N_t) / 1.1, max(N_t) * 1.1], xlabel=r'$t$', ylabel=r'$N$')
ax3.grid(True)
NT, = ax3.plot([], [], label=r"$N$")
ax3.set_title(r'$N(t)$')

for subplot in [ax1, ax2, ax3]:
    subplot.grid(True)
    subplot.legend()
    subplot.set_xlim(0, END_VALUE)

r = np.sqrt(R * R - L * L)
X_O = -R * np.sin(phi)
Y_O = R * np.cos(phi)

X_C = X_O - r * np.sin(psi)
Y_C = Y_O + r * np.cos(psi)

X_REL = -r * np.sin(psi)
Y_REL = r * np.cos(psi)

# Подграфик для визуализации движения системы
ax1 = fig.add_subplot(2, 1, 2)
ax1.axis('equal')  # Одинаковый масштаб по осям
ax1.set(xlim=[-R, R], ylim=[-R, R * 3])
ax1.invert_xaxis()  # Инвертирования оси OX
ax1.invert_yaxis()  # Инвертирования оси OY

# Инициализация графических объектов для анимации
PointO1, = ax1.plot([0], [0], 'bo')
Circ_Angle = np.linspace(0, 2 * np.pi, 100)
Circ, = ax1.plot(X_O[0] + R * np.cos(Circ_Angle), Y_O[0] + R * np.sin(Circ_Angle), 'g')

ArrowX = np.array([0, 0, 0])
ArrowY = np.array([-L, 0, L])
R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, np.atan2(Y_REL[0], X_REL[0]))
Stick_Arrow, = ax1.plot(R_Stick_ArrowX + X_C[0], R_Stick_ArrowY + Y_C[0])  # Линия AB
O1O, = ax1.plot([0, X_O[0]], [0, Y_O[0]], 'b:')  # Линия O1-O
OC, = ax1.plot([X_O[0], X_C[0]], [Y_O[0], Y_C[0]], 'b:')  # Линия OC


# Функция анимации
def anima(i):
    O1O.set_data([0, X_O[i]], [0, Y_O[i]])  # Обновление отрезка O1-O
    OC.set_data([X_O[i], X_C[i]], [Y_O[i], Y_C[i]])  # Обновление отрезка OC
    Circ.set_data(X_O[i] + R * np.cos(Circ_Angle), Y_O[i] + R * np.sin(Circ_Angle))  # Обновление кольца
    R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, np.atan2(Y_REL[i], X_REL[i]))  # Поворот отрезка AB
    Stick_Arrow.set_data(R_Stick_ArrowX + X_C[i], R_Stick_ArrowY + Y_C[i])  # Обновление отрезка AB
    PhiT.set_data(T[:i], phi[:i])  # Обновления подграфика зависимости phi от времени
    KsiT.set_data(T[:i], psi[:i])  # Обновления подграфика зависимости psi от времени
    NT.set_data(T[:i], N_t[:i])  # Обновления подграфика зависимости N от времени

    return O1O, OC, Circ, Stick_Arrow, PhiT, KsiT, NT


# Автоматическая подгонка макета графика
plt.tight_layout()

# Создание анимации
anim = FuncAnimation(fig, anima, frames=STEPS, interval=1, blit=True)

# Отображение графика
plt.show()
