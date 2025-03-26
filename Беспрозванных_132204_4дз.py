def find_primes(n):
    
    try:
        n = int(n)
        if n <= 1:
            print("Введите число больше 2")
        else:
            primes = []
            for num in range(2, n + 1):
                is_prime = True
                for i in range(2, int(num**0.5) + 1):
                    if num % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(num)

            for prime in primes:
                print(prime)

    except ValueError:
        print("Некорректный ввод, введите число")

n = input("Введите число N: ")
find_primes(n)
