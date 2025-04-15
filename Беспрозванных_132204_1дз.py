import math
a = float(input("Длина стороны а: "))
b = float(input("Длина стороны b: "))
angle_degrees = float(input("Введите угол между сторонами в градусах: "))
angle_radians = math.radians(angle_degrees)
c = math.sqrt(a**2 + b**2 + - 2 * a * b *
math.cos(angle_radians))
print("Длина третьей стороны:", c)
