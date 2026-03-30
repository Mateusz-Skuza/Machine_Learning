import math

def sprawdz_zwyciezce(plansza):
    wygrane = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for linia in wygrane:
        if plansza[linia[0]] == plansza[linia[1]] == plansza[linia[2]] != ' ':
            return plansza[linia[0]]
    if ' ' not in plansza:
        return 'Remis'
    return None

def alpha_beta(plansza, glebokosc, alpha, beta, czy_maksymalizuje):
    rezultat = sprawdz_zwyciezce(plansza)
    if rezultat == 'X': return 10 - glebokosc
    if rezultat == 'O': return glebokosc - 10
    if rezultat == 'Remis': return 0

    if czy_maksymalizuje:
        najlepszy_wynik = -math.inf
        for i in range(9):
            if plansza[i] == ' ':
                plansza[i] = 'X'
                wynik = alpha_beta(plansza, glebokosc + 1, alpha, beta, False)
                plansza[i] = ' '
                najlepszy_wynik = max(wynik, najlepszy_wynik)
                alpha = max(alpha, wynik)
                if beta <= alpha:
                    break
        return najlepszy_wynik
    else:
        najlepszy_wynik = math.inf
        for i in range(9):
            if plansza[i] == ' ':
                plansza[i] = 'O'
                wynik = alpha_beta(plansza, glebokosc + 1, alpha, beta, True)
                plansza[i] = ' '
                najlepszy_wynik = min(wynik, najlepszy_wynik)
                beta = min(beta, wynik)
                if beta <= alpha:
                    break
        return najlepszy_wynik

def znajdz_najlepszy_ruch(plansza):
    najlepsza_wartosc = -math.inf
    ruch = -1
    for i in range(9):
        if plansza[i] == ' ':
            plansza[i] = 'X'
            wartosc_ruchu = alpha_beta(plansza, 0, -math.inf, math.inf, False)
            plansza[i] = ' '
            if wartosc_ruchu > najlepsza_wartosc:
                najlepsza_wartosc = wartosc_ruchu
                ruch = i
    return ruch

def rysuj_plansze(p):
    print(f"\n {p[0]} | {p[1]} | {p[2]} ")
    print("-----------")
    print(f" {p[3]} | {p[4]} | {p[5]} ")
    print("-----------")
    print(f" {p[6]} | {p[7]} | {p[8]} \n")


plansza = [' ' for _ in range(9)]
print("Gra w Kółko i Krzyżyk! Ty jesteś 'O', komputer to 'X'.")
print("Pola są numerowane od 0 do 8.")

while True:
    rysuj_plansze(plansza)
    try:
        wybor = int(input("Twój ruch (0-8): "))
        if plansza[wybor] != ' ':
            print("To pole jest zajęte!")
            continue
        plansza[wybor] = 'O'
    except (ValueError, IndexError): 
        print("Wpisz poprawną liczbę od 0 do 8!")
        continue
    if sprawdz_zwyciezce(plansza): break

    print("Komputer analizuje ruchy...")
    ruch_komputer = znajdz_najlepszy_ruch(plansza)
    if ruch_komputer != -1:
        plansza[ruch_komputer] = 'X'

    if sprawdz_zwyciezce(plansza): break

rysuj_plansze(plansza)
wynik_koncowy = sprawdz_zwyciezce(plansza)
if wynik_koncowy == 'Remis':
    print("Mamy remis!")
else:
    print(f"Koniec gry! Zwycięzca: {wynik_koncowy}")