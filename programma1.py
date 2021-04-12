# -*- coding: utf-8 -*- 
import sys
import codecs
import nltk

def AnnotazioneLinguisticaTesto(frasi):
    #lunghezza totale del testo
    lunghezzaTotale=0.0
    #lista dei token 
    listaToken = []
    #lista delle POS
    tokenPOSTot = []
    for frase in frasi:
        #divido in token la frase
        token = nltk.word_tokenize(frase)
        #lista con tutti i token del testo
        tokenPOS = nltk.pos_tag(token)
        #Calcolo tutti i token del testo
        listaToken = listaToken + token
        #calcolo tutti i POS del testo
        tokenPOSTot = tokenPOSTot + tokenPOS
        lunghezzaTotale = lunghezzaTotale + len(token) 
    #restituisco la lunghezza totale del testo e la lunghezza dei token 
    return lunghezzaTotale, listaToken, tokenPOSTot

def CalcolaLunghezzaMediaFrasi (frasi, token):
    #calcolo la media tra il numero dei token e delle frasi 
    lunghezzaMediaFrasi = float(token)/float(frasi)
    return lunghezzaMediaFrasi

def CalcolaLunghezzaMediaParole (token,listaToken): 
    #variabile che conta i caratteri
    numeroCaratteri = 0
    #per ogni token calcolo il numero dei caratteri
    for tok in listaToken:
        numeroCaratteri = numeroCaratteri + len(tok)
     #calcolo la media tra il numero dei token e dei caratteri 
    lunghezzaMediaParole = float(numeroCaratteri)/float(token)
    return lunghezzaMediaParole

#funzione che calcola il numero degli hapax
def CalcolaHapax (listaToken):
    #varaibile che conta gli hapax
    hapax = 0
    #calcolo il vocabolario
    vocabolario = set(listaToken)
    #per ogni token presente nel vocabolario
    for tok in vocabolario:
        #calcolo la frequenza del token
        freqToken = listaToken.count(tok)
        #se la frequenza del token e' 1 aggiungo il token agli hapax 
        if freqToken == 1:
            hapax += 1
    #restituisco il numero degli hapax
    return hapax

def CalcolaVocabolarioEHapax(lunghezzaTotale, listaToken):
    #variabile che incrementa di 1000 unita'
    n = 1000
    #variabili che dividono il corpus in porzioni di 1000 token
    vocabolario1000 = []
    vocabolario2000 = []
    vocabolario3000 = []
    vocabolario4000 = []
    vocabolario5000 = []
    while n < lunghezzaTotale: 
        #calcolo la grandezza del vocabolario
        vocabolario = set(listaToken[:n])
        #per ogni porzione di 1000 token calcolo la grandezza del vocabolario e il numero degli hapax (richiamando la funzione CalcolaHapax)
        if n == 1000:
            vocabolario1000 = len(vocabolario)
            hapax1000 = CalcolaHapax(listaToken[:n])
        if n == 2000:
            vocabolario2000 = len(vocabolario)
            hapax2000 = CalcolaHapax(listaToken[:n])
        if n == 3000:
            vocabolario3000 = len(vocabolario)
            hapax3000 = CalcolaHapax(listaToken[:n])
        if n == 4000:
            vocabolario4000 = len(vocabolario)
            hapax4000 = CalcolaHapax(listaToken[:n])
        if n == 5000:
            vocabolario5000 = len(vocabolario)
            hapax5000 = CalcolaHapax(listaToken[:n])
        #incremento la variabile e restituisco i risultati
        n += 1000
    return vocabolario1000, hapax1000, vocabolario2000, hapax2000, vocabolario3000, hapax3000, vocabolario4000, hapax4000, vocabolario5000, hapax5000

#funzione che calcola la Type Token Ratio (TTR) dei primi 5000 token del testo
def CalcolaTTR(listaToken):
    #calcolo la Type Token Ratio dei primi 5000 token del corpus
    token5000 = listaToken[:5000]
    TTR = float(len(token5000))/float(len(listaToken))
    return TTR

#funzione che calcola la distribuzione in percentuale dei sostantivi, degli aggettivi, dei verbi e dei pronomi per frase
def percentualeEMediaPOS(tokenPOSTot, frasi):
    #variabili che contano il numero dei sostantivi, degli aggettivi, dei verbi e dei pronomi
    Numero_Sostantivi = 0
    Numero_Aggettivi = 0
    Numero_Verbi = 0
    Numero_Pronomi = 0
    #per ogni token contenuto all'interno delle POS calcolo il numero delle varie POS e incremento le variabili
    for token in tokenPOSTot:
        #sostantivi
        if token[1] in {"NN", "NNS", "NNP", "NNPS"}:
            Numero_Sostantivi += 1
        #aggettivi
        if token[1] in {"JJ", "JJR", "JJS"}:
            Numero_Aggettivi += 1
        #verbi
        if token[1] in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
            Numero_Verbi +=  1
        #pronomi
        if token[1] in {"PRP", "PRP$"}:
           Numero_Pronomi += 1
    
    #Calcolo la percetuale dei Sostantivi, degli aggettivi, dei verbi e dei pronomi
    Percentuale_Sostantivi = float(Numero_Sostantivi/100)
    Percentuale_Aggettivi = float(Numero_Aggettivi/100)
    Percentuale_Verbi = float(Numero_Verbi/100)
    Percentuale_Pronomi = float(Numero_Pronomi/100)
    
    #Calcolo la media dei Sostantivi, degli aggettivi, dei verbi e dei pronomi per frase
    Media_Sostantivi = float(Numero_Sostantivi)/(frasi)
    Media_Aggettivi = float(Numero_Aggettivi)/(frasi)
    Media_Verbi = float(Numero_Verbi)/(frasi)
    Media_Pronomi = float(Numero_Pronomi)/float(frasi)
    return Percentuale_Sostantivi, Media_Sostantivi, Percentuale_Aggettivi, Media_Aggettivi, Percentuale_Verbi, Media_Verbi, Percentuale_Pronomi, Media_Pronomi

def main(file1, file2):
    #apro i due file
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
    
    #leggo i due file
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
   
    #divido in token 
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    #divido in frasi i due file
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    #confronto le lunghezze dei 2 file
    lunghezza1, listaToken1, tokenPOSTot1 = AnnotazioneLinguisticaTesto(frasi1)
    lunghezza2, listaToken2, tokenPOSTot2 = AnnotazioneLinguisticaTesto(frasi2)
    
    #numero delle frasi
    numeroFrasi1 = len(frasi1)
    numeroFrasi2 = len(frasi2)

    #lunghezza media frasi in termini di token
    lunghezzaMediaFrasi1 = CalcolaLunghezzaMediaFrasi(numeroFrasi1, lunghezza1)
    lunghezzaMediaFrasi2 = CalcolaLunghezzaMediaFrasi(numeroFrasi2, lunghezza2)

    #lunghezza media delle parole in termini di caratteri
    lunghezzaMediaParole1 = CalcolaLunghezzaMediaParole(lunghezza1, listaToken1)
    lunghezzaMediaParole2= CalcolaLunghezzaMediaParole(lunghezza2, listaToken2)

    #vocabolario e hapax all'aumentare del coprus 
    vocabolario1000_1, hapax1000_1, vocabolario2000_1, hapax2000_1, vocabolario3000_1, hapax3000_1, vocabolario4000_1, hapax4000_1, vocabolario5000_1, hapax5000_1 = CalcolaVocabolarioEHapax(lunghezza1, listaToken1)
    vocabolario1000_2, hapax1000_2, vocabolario2000_2, hapax2000_2, vocabolario3000_2, hapax3000_2, vocabolario4000_2, hapax4000_2, vocabolario5000_2, hapax5000_2 = CalcolaVocabolarioEHapax(lunghezza2, listaToken2)
    
    #percentuale e media dei Sostantivi, Aggettivi, Pronomi e Verbi
    PercentualeSostantivi_1, MediaSostantivi_1, PercentualeAggettivi_1, MediaAggettivi_1, PercentualeVerbi_1, MediaVerbi_1, PercentualiPronomi_1, MediaPronomi_1 = percentualeEMediaPOS(tokenPOSTot1, numeroFrasi1)
    PercentualeSostantivi_2, MediaSostantivi_2, PercentualeAggettivi_2, MediaAggettivi_2, PercentualeVerbi_2, MediaVerbi_2, PercentualiPronomi_2, MediaPronomi_2 = percentualeEMediaPOS(tokenPOSTot2, numeroFrasi2)   
    
                                                    #OUTPUT
    print "\n"
    print "***************OUPUT PROGRAMMA 1***************"
    print "\n"

    print "------NUMERO DEI TOKEN------"
    print "Il corpus dei racconti di viaggio delle donne e' lungo", lunghezza1, "token"
    print "Il corpus dei racconti di viaggio degli uomini e' lungo", lunghezza2, "token"
    print "\n"
   
    print "------NUMERO DELLE FRASI------ "
    print "Il corpus dei racconti di viaggio delle donne ha", numeroFrasi1, "frasi"
    print "Il corpus dei racconti di viaggio degli uomini ha", numeroFrasi2, "frasi"
    print "\n"
   
    print "------LUNGHEZZA DELLE FRASI IN TERMINI DI TOKEN-----"
    print  "Lunghezza media delle frasi del corpus dei racconti di viaggio delle donne: ", lunghezzaMediaFrasi1
    print  "Lunghezza media delle frasi del corpus dei racconti di viaggio degli uomini: ", lunghezzaMediaFrasi2
    print "\n"
   
    print "-------LUNGHEZZA MEDIA DELLE PAROLE IN TERMINI DI CARATTERI-----"
    print "Lunghezza media delle parole del corpus dei racconti di viaggio delle donne: ", lunghezzaMediaParole1
    print "Lunghezza media delle parole del corpus dei racconti di viaggio degli uomini: ", lunghezzaMediaParole2
    print "\n"

    print "------GRANDEZZA DEL VOCABOLARIO ALL'AUMENTARE DEL CORPUS------ "
    print "Vocabolario del all'aumentare del corpus dei racconti di viaggio delle donne per porzioni di 1000 token: " 
    print "\t", vocabolario1000_1
    print "\t", vocabolario2000_1
    print "\t", vocabolario3000_1
    print "\t", vocabolario4000_1
    print "\t", vocabolario5000_1
    print 

    print "Grandezza del vocabolario all'aumentare del corpus dei racconti di viaggio degli uomini per porzioni di 1000 token: "
    print "\t", vocabolario1000_2
    print "\t", vocabolario2000_2
    print "\t", vocabolario3000_2
    print "\t", vocabolario4000_2
    print "\t", vocabolario5000_2
    print "\n"

    print "------NUMERO DEGLI HAPAX ALL'AUMENTARE DEL CORPUS-------"
    print "Numero di Hapax all'aumentare del corpus dei racconti di viaggio delle donne per porzioni di 1000 token:"
    print "\t", hapax1000_1
    print "\t", hapax2000_1
    print "\t", hapax3000_1
    print "\t", hapax4000_1
    print "\t", hapax5000_1
    print

    print "Numero di Hapax all'aumentare del corpus dei racconti di viaggio degli uomini per porzioni di 1000 token:"
    print "\t", hapax1000_2
    print "\t", hapax2000_2
    print "\t", hapax3000_2
    print "\t", hapax4000_2
    print "\t", hapax5000_2
    print "\n"
       
    print "------LA RICCHEZZA LESSICALE CALCOLATA ATTRAVERSO LA TYPE TOKEN RATIO (TTR) SUI PRIMI 5000 TOKEN------"
    print "TTR sui primi 5000 token del corpus dei racconti di viaggio delle donne:", CalcolaTTR(listaToken1)
    print "TTR sui primi 5000 token del corpus dei racconti di viaggio degli uomini:", CalcolaTTR(listaToken2)
    print "\n"

    print "------PERCENTUALE DELLE POS (SOSTANTIVI, VERBI, AGGETTIVI E PRONOMI) PER FRASE ------"
    print "Percentuale delle POS del corpus dei racconti di viaggio delle donne\t"
    print "\tSostantivi -->", PercentualeSostantivi_1,"%"
    print "\tAggettivi --> ", PercentualeAggettivi_1,"%"
    print "\tVerbi -->", PercentualeVerbi_1,"%"
    print "\tPronomi -->", PercentualiPronomi_1,"%"
    print

    print "Percentuale delle POS del corpus dei racconti di viaggio degli uomini\t"
    print "\tSostantivi -->", PercentualeSostantivi_2,"%"
    print "\tAggettivi -->", PercentualeAggettivi_2,"%"
    print "\tVerbi -->", PercentualeVerbi_2, "%"
    print "\tPronomi -->", PercentualiPronomi_2,"%"
    print "\n"

    print "------NUMERO MEDIO DELLE POS (SOSTANTIVI, VERBI, AGGETTIVI E PRONOMI) PER FRASE ------"
    print "Numero medio delle POS nel corpus dei racconti di viaggio delle donne\t"
    print "\tSostantivi -->",  MediaSostantivi_1 
    print "\tAggettivi -->", MediaAggettivi_1
    print "\tVerbi -->", MediaVerbi_1
    print "\tPronomi -->", MediaPronomi_1
    print
    
    print "Numero medio delle POS nel corpus dei racconti di viaggio degli uomini\t"
    print "\tSostantivi -->",  MediaSostantivi_2
    print "\tAggettivi -->", MediaAggettivi_2
    print "\tVerbi -->", MediaVerbi_2
    print "\tPronomi -->", MediaPronomi_2

main(sys.argv[1], sys.argv[2])