# Dataset

Questo progetto utilizza un dataset di testo per l’addestramento del filtro anti-hating.

## Contenuto
In questa cartella è presente il file:
- `Filter_Toxic_Comments_dataset.csv` — contiene i dati testuali e le relative etichette (`0` = non offensivo, `1` = offensivo) alle sette differenti labels che caratterizzano la tipologia di odio (se presente) del testo.

## Descrizione del dataset
Ogni riga del file rappresenta un esempio con le seguenti colonne principali:
- `corpus: comment_text` — il contenuto testuale (commento o frase)
- `label 1: toxic` - 
- `label 2: severe_toxic` -  
- `label 3: obscene` - 
- `label 4: threat` -
- `label 5: insult` - 
- `label 6: identity_hate` - 
- `label 7: sum_injurious` — 

## Origine
Il dataset è scaricabile dal mio drive personale
[link-al-dataset-originale](https://drive.google.com/file/d/1Y0750AWh4Wp6M1WOOxaNH0Rv7hy9BlUx/view?usp=drive_link) 

## Utilizzo
Il notebook principale (`notebooks/Anti-Hating_Filter.ipynb`) legge automaticamente il file:
```python
import pandas as pd
data = pd.read_csv("data/dataset.csv")
```

