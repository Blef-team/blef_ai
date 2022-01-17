BlefCards = [str(value) + str(suit) for value in range(6) for suit in range(4)] 

def report_progress(i, n):
    if (any(i == int(n * k/100) for k in range(100))):
        print("Run " + str(i))
