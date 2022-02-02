# Praca Magisterska
## Temat: Analiza porównawcza wybranych metod ekstrakcji cech na potrzeby zadania klasyfikacji fałszywych wiadomości

Repozytorium zawiera kod źródłowy wykonanych eksperymentów. Ze względu na ograniczenia repozytorium, dane znajdują się na [Dysku Google](https://drive.google.com/drive/folders/1eMP5Z22RhdnNQ_IBJXbgfXJm729j_uI9?usp=sharing) i należy je pobrać osobno (po przyznaniu dostępu).

### Jak uruchomić kod?
1. Plik `Dane.ipynb` zawiera kod pozwalający na preprocessing danych. Można go uruchomić przy użyciu narzędzia Jupyter Notebook. Przed uruchomieniem należy zmienić ścieżkę dostępu do plików z danymi.
2. Plik `fake_news.py` zawiera kod bezpośrednio wykonujący pojedynczy eksperyment. Uruchamia się go poleceniem 
```
python fake_news.py <nazwa_pliku> <indeks_klasyfikatora> <indeks_zbioru_cech> [--use_counts]
```
gdzie `<nazwa_pliku>` oznacza nazwę pliku z przetworzonymi danymi, a ustawiona flaga `--use_counts` pozwala na skorzystanie z dodatkowego zbioru cech z wybranymi cechami.

3. Katalog `Wizualizacja` zawiera dodatkowe skrypty pozwalające na wizualizację danych.

Do uruchomienia kodu może być wymagane zainstalowanie dodatkowych modułów.
