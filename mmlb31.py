
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
# Читать изображение
img = cv.imread('/home/aliumar/Рабочий стол/Studies/MM/MM_lb3/krst.png', 0)
 
 # Алгоритм быстрого преобразования Фурье для получения частотного распределения
f = np.fft.fft2(img)
 
 # По умолчанию центральная точка результата находится в верхнем левом углу,
 # Вызов функции fftshift () для перехода в среднее положение
fshift = np.fft.fftshift(f)       
 
'''Логарифмируем абсолютное значение (т.е. амплитуду, убирая комплексную сост-щую) 
результата ДПФ  для нормального отображения контрастности'''
fimg = np.log(np.abs(fshift))
 
'''np.mean вычисляет ср знач массива. В случае изображение вычисляет 
среднее знач яркости (или сперва ср. 3 rgb) по осям''' 
# Вычисляем одномерный спектр по оси x
f_x = np.mean(fimg, axis=0) # вычисляет среднее по вретикальным осям
# Вычисляем одномерный спектр по оси y
f_y = np.mean(fimg, axis=1) #вычисляет среднее по горизонтальным осям

# Определяем общий диапазон значений для оси y
y_min = min(np.min(f_x), np.min(f_y))
y_max = max(np.max(f_x), np.max(f_y))

# Показать результаты
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Изображение')
plt.axis('off')
plt.subplot(222), plt.imshow(fimg, 'gray'), plt.title('Двумерный спектр ДПФ')
plt.axis('off')
plt.subplot(223), plt.plot(f_x), plt.title('Одномерный спектр по оси "х"')
plt.ylim(y_min, y_max)
plt.subplot(224), plt.plot(f_y), plt.title('Одномерный спектр по оси "у"')
plt.ylim(y_min, y_max)
plt.show()
