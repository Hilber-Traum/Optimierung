
## TODO: Aufgabenteil 1a. Alle Zahlen zwischen 1 und 10 ausgeben
print('### 1a ###')
# BEGIN SOLUTION
for i in range(10):
    print(i+1)
# END SOLUTION


## TODO: Aufgabenteil 1b. Alle geraden Zahlen zwischen 1 und 10 ausgeben, auf 2 Arten
print('\n### 1b ###')
# BEGIN SOLUTION
for i in range(1,11):
    if i%2==0:
        print(i)
for i in range(2,11,2):
    print(i)
# END SOLUTION


## TODO: Aufgabenteil 1c. Funktion, die alle geraden Zahlen zwischen 1 und n ausgibt
def alle_gerade_zahlen(n):
    # BEGIN SOLUTION
    for i in range(2,n+1,2):
        print(i)
    # END SOLUTION

# Funktion testen
print('\n### 1d ###')
alle_gerade_zahlen(10)
alle_gerade_zahlen(4)
alle_gerade_zahlen(0)
