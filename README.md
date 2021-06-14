# Progetto-Linguistica-Computazionale
**Mini-progetto in Python svolto per sostenere l'esame di Linguistica Computazionale (a.a 2017-2018)**

*Dopo aver creato due corpora in inglese, di almeno 5000 token ciascuno, contenenti testi estratti rispettivamente da blog di racconti di viaggio scritti da uomini e donne, sui suddetti corpora sono state eseguite alcune operazioni*



**Programma 1**: confronto dei testi in base ad alcune informazioni statistiche
  - la lunghezza media delle frasi in termini di token e la lunghezza media delle parole in termini di caratteri;
  - la grandezza del vocabolario e il numero di hapax all'aumentare del corpus per porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.)
  - la ricchezza lessicale calcolata attraverso la Type Token Ratio (TTR) sui primi 5000 token;
  - la distribuzione (in termini percentuali) di Sostantivi, Aggettivi, Verbi e Pronomi;
  - il numero medio di Sostantivi, Aggettivi, Verbi e Pronomi per frase.



**Programma 2**: Estrazione di informazioni da ognuno dei due corpora separatamente
1. Per ogni corpora sono stati estratti e ordinati 
    - in ordine di frequenza decrescente:
        - **i 20 token più frequenti** escludendo la punteggiatura;
        - i 20 **Aggettivi** più frequenti;
        - i 20 **Verbi** più frequenti;
        - le **10 PoS (Part-of-Speech)** più frequenti;
        - i **10 trigrammi di PoS (Part-of-Speech)** più frequenti; 
    
    - in ordine decrescente i **10 bigrammi di PoS (Part-of-Speech)**:
        - con **probabilità congiunta massima**, indicando anche la relativa probabilità;
        - con **probabilità condizionata massima**, indicando anche la relativa probabilità;

2. Creazione di un’unica lista con i 10 Sostantivi più frequenti contenuti nei blog maschili e i 10 Sostantivi più frequenti dei blog femminili e per ognuno di questi Sostantivi sono stati ordinati gli Aggettivi che li precedono rispetto alla forza associativa (calcolata in termini di Local Mutual Information);

3. Individuazione e classificazione delle e **Entità Nominate (NE)** presenti nel testo ed estrazione dei 20 nomi propri di luogo più frequenti (tipi), ordinati per frequenza.
