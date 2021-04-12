# -*- coding: utf-8 -*- 
import sys
import nltk
import codecs
import re
import math
from nltk import FreqDist 
from nltk import trigrams
from nltk import bigrams

def AnnotazioneLinguistica(frasi):
    #lista dei token 
    listaToken = []
    #lista delle POS
    tokenPOSTot = []
    #lista delle NE
    for frase in frasi:
        #divido in token la frase
        tokens = nltk.word_tokenize(frase)
        #faccio il POS Tag dei token 
        tokenPOS = nltk.pos_tag(tokens)
        #Calcolo tutti i token del testo
        listaToken = listaToken + tokens
        #calcolo tutti i POS del testo
        tokenPOSTot = tokenPOSTot + tokenPOS
    #calcolo la lunghezza del testo    
    lunghezza = len(listaToken)
    #restituisco la lista dei token e dei POS
    return listaToken, tokenPOSTot, lunghezza

#funzione che prende in input un dizionario e restituisce la lista di tutte le coppie chiave-valore ordinate sul valore
def OrdinaBigrammi(dict):
    return sorted(dict.items(), key =lambda x: x[1], reverse = True)

#funzione che calcola i 20 token più frequenti 
def CalcolaFrequenzaToken(listaToken):
    #lista dei token frequenti
    tokenFreq = []
    #elimino la punteggiatura
    punteggiatura = [".", ",", ":", "?", "!", "-", " ' ",  " '' ", "``"]
    #scorro i token
    for token in listaToken:
        #se il token non è un segno di punteggiatura lo aggiungo alla lista dei token frequenti
        if token not in punteggiatura:
            tokenFreq.append(token)
        #calcolo la distribuzione di frequenza dei token    
        DistribuzioneDiFrequenza = nltk.FreqDist(tokenFreq)
        #callcolo i 20 token più frequenti
        DistribuzioneDiFrequenzaOrdinata = DistribuzioneDiFrequenza.most_common(20)
    return DistribuzioneDiFrequenzaOrdinata

#funzione che calcola i 20 aggettivi piu' frequenti
def AggettiviFreq(tokenPOSTot):
    #lista dgli aggettivi frequenti
    AggFreq = []
    #Scorro le POS token per token
    for token in tokenPOSTot:
        #se il token e' un aggettivo lo aggiungo alla lista degli aggettivi frquenti
        if token[1] in {"JJ", "JJR", "JJS"}:
            AggFreq.append(token[0])
    #calcolo la distribuzione di frequenza degli aggettivi e i 20 aggettivi piu' frequenti
    DistribuzioneAggettivi = nltk.FreqDist(AggFreq)
    AggettiviFrequenti = DistribuzioneAggettivi.most_common(20)
    return AggettiviFrequenti

#funzione che calcola i 20 verbi piu' frequenti
def VerbiFreq(tokenPOSTot):
    #lista dei verbi frequenti
    VFreq = []
    #Scorro le POS token per token
    for token in tokenPOSTot:
        #se il token e' un verbo lo aggiungo alla lista dei verbi frequenti
        if token[1] in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
            VFreq.append(token[0])
    #calcolo la distribuzione d frequenza dei verbi e i 20 verbi piu' frequenti
    DistribuzioneVerbi = nltk.FreqDist(VFreq)
    VerbiFrequenti = DistribuzioneVerbi.most_common(20)
    return VerbiFrequenti

#funzione che calcola i 10 POS piu' frequenti
def POSfreq(tokenPOSTot):
    #lista che contiene le POS
    PartOfSpeech = []
    #scorro le POS le aggiungo alla lista definita precedentemente
    for token in tokenPOSTot:
        #calcolo la distribuzione di frequenza delle POS e le 10 POS piu' frequenti 
        PartOfSpeech.append(token[1])
    #calcolo la distribuzione di frequenza delle POS e le 10 POS piu' frequenti
    DistribuzionePOS = nltk.FreqDist(PartOfSpeech)
    POSFrequenti = DistribuzionePOS.most_common(10)
    return POSFrequenti

#funzione che calcola la frequenza de trigrammi
def CalcolaFeqTrigrammi(trigrammi):
    #calcolo la frequenza dei trigrammi di POS
    DistribuzioneTrigrammi = nltk.FreqDist(trigrammi)
    #estraggo i 10 trigrammi di POS più frequenti
    trigrammiFrequenti = DistribuzioneTrigrammi.most_common(10)
    return trigrammiFrequenti
    
def CalcolaBigrammi(tokenPOSTot):
    #calcolo i bigrammi
    bigrammiPOS = list(bigrams(tokenPOSTot))
    #elimino i bigrammi uguali
    bigrammiDiversiPOS = set(bigrammiPOS)
    return bigrammiPOS, bigrammiDiversiPOS

def CalcolaProbCondizionataECongiunta(listaToken, bigrammiPOS, bigrammiDiversiPOS, lunghezza):
    #lista che raccoglie i bigrammi e le loro prob. condizionate
    bigrammiCondizionati = {}
    #lista che raccoglie i bigrammi e le loro prob. congiunte
    bigrammiCongiunti = {}
    for bigramma in bigrammiDiversiPOS: 
        #calcolo la frequenza dei trigrammi
        frequenzaBigrammi = bigrammiPOS.count(bigramma)
        #calcolo la frequenza della prima parte del trigramma
        frequenzaPrimoElemento = listaToken.count(bigramma[0][0])
        #calcolo la probabilita' condizionata
        probCondizionata = float(frequenzaBigrammi)/float(frequenzaPrimoElemento)
        #calcolo la frequenza relativa
        frequenzaRelativa = float(frequenzaPrimoElemento)/lunghezza
        #calcolo la probabilita' congiunta
        probCongiunta = probCondizionata*frequenzaRelativa #probabilita' congiunta = prob condizionata*frequenzarelativa del primo elemento
        #assegno le due probabilità ai dizionari
        bigrammiCondizionati[bigramma] = probCondizionata        
        bigrammiCongiunti[bigramma] = probCongiunta
        #ordino i bigrammi richiamando la funzione 'OrdinaBigrammi' definita precedentemente
        bigrammiCondizionatiOrdinati = OrdinaBigrammi(bigrammiCondizionati)
        bigrammiCongiuntiOrdinati = OrdinaBigrammi(bigrammiCongiunti)
    return bigrammiCondizionatiOrdinati, bigrammiCongiuntiOrdinati

def sostantiviFreq(bigrammiPOS):
    #creo la lista dei sostantivi frequenti
    ListaSostantiviFreq = []
    #scorro i token
    for token in bigrammiPOS:
        if token[1] in {"NN", "NNS", "NP", "NPS"}: #se il token è un sostantivo lo aggiungo alla lista
            ListaSostantiviFreq.append(token)
    #calcolo la distribuzione di frequenza degli aggettvi
    distribuzioneSostantivi = FreqDist(ListaSostantiviFreq)
    #calcolo i dieci aggettivi più frequenti
    sostantivi_dieci = distribuzioneSostantivi.most_common(10)
    return sostantivi_dieci

def CreaListaAggSostantivo(bigrammiPOS, sostantivi_dieci):
    #lista dei bigrammi aggettivo-Sostantivo
    listaSostAgg = []
    #scorro i bigrammi
    for bigramma in bigrammiPOS:
        #se il primo bigramma è un aggettivo lo aggiungo alla lista
        if bigramma[0][1]in [ "JJ", "JJR","JJS"]:
            listaSostAgg.append(bigramma)
        #se il secondo bigramma e un sostantivo e fa parte della lista deii sostantivi frequenti calcolati precedentemente
        elif (bigramma[1][1] in ["NNP","NN","NNS","NNPS"]) and (bigramma[1] in sostantivi_dieci):
            listaSostAgg.append(bigramma[0]) #lo aggiungo alla lista degli Aggettivi-sostantivi
    return listaSostAgg

#calcolo la lmi     
def lmi(listaToken, bigrammiPOS, listaSostAgg):
    #dizionario che contiene i bigrammi e la loro lmi
    lmi = {}
    #lunghezza dei bigrammi
    lunghezzaBigrammi = len(bigrammiPOS)
    #scorro i bigrammi all'interno dela lista dei Sostantivi-Aggettivi
    for bigramma in listaSostAgg:
        #calcolo la frequenza del bigramma
        frequenzaBigramma = bigrammiPOS.count(bigramma)
        #calcolo solo la frequenza del sostantivo
        frequenzaSostantivo = listaToken.count(bigramma[0][0])
        #calcolo solo la frequenza dell'aggettivo
        frequenzaAggettivo = listaToken.count(bigramma[1][0])
        #calcolo la probabilita' del bigramma
        probabilitaBigr = float(frequenzaBigramma)/lunghezzaBigrammi
        #calcolo la probabilità del sostantivo
        probSostantivo = float(frequenzaSostantivo)/lunghezzaBigrammi
        #calcolo la probabilita' dell'aggettivo
        probAgg = float(frequenzaAggettivo)/lunghezzaBigrammi
        #calcolo la Local Mutual Information 
        CalcoloProbabilita = probabilitaBigr/(probAgg*probSostantivo)
        CalcoloLmi = probabilitaBigr*math.log(CalcoloProbabilita ,2)
        #inserisco la lmi nel dizionario
        lmi[bigramma] = CalcoloLmi
        #ordino la lmi richiamando la funzione 'OrdinaBigrammi' definita precedentemente
        lmiOrdinata = OrdinaBigrammi(lmi)
        #estraggo i dieci bigrammi 
        lmiOrdinata_venti = lmiOrdinata[:10]
    return lmiOrdinata_venti 

#calcolo i nomi prorpi di luogo con il Named Entity Tagger
def nomiPropriLuogo(frasi):
    #lista contente i nomi propri di luogo
    NamedEntityLuoghi = []
    #scorro le frasi del testo
    for frase in frasi:
        #tokenizzo la frase
        tokens = nltk.word_tokenize(frase)
        #faccio il POS Tag dei token 
        tokenPOS = nltk.pos_tag(tokens)
        #calcolo le Named Entity
        namedEntity = nltk.ne_chunk(tokenPOS)
        #scorro i nodi
        for nodo in namedEntity:
            NE = ''
            #controllo se nodo è un nodo intermedio o una foglia
            if hasattr(nodo, "label"):
                #estraggo l'etichetta del nodo, in questo caso i nomi propri di luogo
                if nodo.label() in ["GPE"]:
                    #scorro le foglie del nodo e le unisco alle NE
                    for partNE in nodo.leaves():
                        NE = NE + ' '+ partNE[0]
                        NamedEntityLuoghi.append(NE)
    #infine restituisco i nomi propri di luogo 
    return NamedEntityLuoghi

def VentiNE(NamedEntityType):
    #lista che contiene i tipi
    listaType = []
    #scorro i tipi
    for Type in NamedEntityType:
        #aggiungo i tipi alla lista definita precedentemente
        listaType.append(Type)
        #calcolo al distribuzione di frequenza dei tipi
        distribuzioneType = nltk.FreqDist(listaType)
        #estraggo i 20 tipi di nomi di luogo più frequenti e li restituisco
        Type_venti = distribuzioneType.most_common(20)
    return Type_venti

#funzione che stampa le distribuzioni di frequenza 
def stampaFrequenze(DistribuzioneDiFrequenza):
    for elem in DistribuzioneDiFrequenza:
        print "\t", elem[0].encode("utf-8"), " occorre ", elem[1], "volte"

#funzione che stampa le frequenze dei trigrammi
def stampaFrequenzeTrigrammi(trigrammiFrequenti):
    for elem in trigrammiFrequenti:
        print "\tIl trigramma", elem[0][0][1], elem[0][1][1], elem[0][2][1], "occorre", elem[1], "volte"

#funzione che stampa le probabilita' dei bigrammi
def stampaProb(bigrammi):
    for bigramma in bigrammi[:10]:
        print "\tBigramma", bigramma[0][0][1], bigramma[0][1][1], "\t\t\tProbabilità -> ", bigramma[1]

#funzione che stampa le lmi dei bigrammi Sostantivo-Aggettivo
def stampalmi(lmiOrdinata_venti):
    for elem in lmiOrdinata_venti :
        print "\tBigramma:", elem[0], "\t\t\tLMI-> ", elem[1]

#funzione che stampa le frequenze dei nomi propri di luogo
def stampaType(Type_venti):
    for elem in Type_venti:
        print "\tIl nome proprio di luogo", elem[0].encode("utf-8"), "occorre", elem[1], "volte"
        
#funzione principale        
def main(file1, file2):
    #apro i due file
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
   
    #leggo i due file
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    
    #tokenizzo i testi
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
   
    #divido in frasi i due file 
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    
    #calcolo la lista di tutti i token, la lista di tutte le POS e la lunghezza dei token
    listaToken1, tokenPOSTot1, lunghezza1 = AnnotazioneLinguistica(frasi1)
    listaToken2,  tokenPOSTot2, lunghezza2 = AnnotazioneLinguistica(frasi2)
    
    #calcolo i 20 token piu' frequenti
    TokenFrequenti1 = CalcolaFrequenzaToken(listaToken1)
    TokenFrequenti2 = CalcolaFrequenzaToken(listaToken2)

    #calcolo i 20 aggettivi piu' frequenti
    AggettiviFreq1 = AggettiviFreq(tokenPOSTot1)
    AggettiviFreq2 = AggettiviFreq(tokenPOSTot2)
    
    #calcolo i 20 verbi piu' frequenti
    VerbiFreq1 = VerbiFreq(tokenPOSTot1)
    VerbiFreq2 = VerbiFreq(tokenPOSTot2)

    #calcolo le 10 POS piu' frequenti
    POSfreq1 = POSfreq(tokenPOSTot1)
    POSfreq2 = POSfreq(tokenPOSTot2)
    
    #calcolo i brigrammi
    bigrammiPOS1, bigrammiDiversiPOS1 = CalcolaBigrammi(tokenPOSTot1)
    bigrammiPOS2, bigrammiDiversiPOS2 = CalcolaBigrammi(tokenPOSTot2)
    
    #Calcolo i trigrammi
    trigrammi1 = list(trigrams(tokenPOSTot1))
    trigrammi2 = list(trigrams(tokenPOSTot2))

    #calcolo i 10 trigrammi di POS piu' frequenti
    trigrammiFreq1 = CalcolaFeqTrigrammi(trigrammi1)
    trigrammiFreq2 = CalcolaFeqTrigrammi(trigrammi2)

    #calcolo la probabilita' congiunta e condizionata dei bigrammi
    probCondizionataBigrammi1, probCongiuntaBigrammi1,  = CalcolaProbCondizionataECongiunta(listaToken1, bigrammiPOS1, bigrammiDiversiPOS1, lunghezza1)
    probCondizionataBigrammi2, probCongiuntaBigrammi2 = CalcolaProbCondizionataECongiunta(listaToken2, bigrammiPOS2, bigrammiDiversiPOS2, lunghezza2)
    
    #calcolo la lista dei 10 sostantivi piu' frequenti
    SostantiviFreq1 = sostantiviFreq(tokenPOSTot1)
    SostantiviFreq2 = sostantiviFreq(tokenPOSTot2)

    #calcolo i bigrammi sostantivo-aggettivo
    BigrammiSostAgg1 = CreaListaAggSostantivo(bigrammiPOS1, SostantiviFreq1)
    BigrammiSostAgg2 = CreaListaAggSostantivo(bigrammiPOS2, SostantiviFreq2)
    
    #calcolo la lmi
    lmi1 = lmi(listaToken1, bigrammiPOS1, BigrammiSostAgg1)
    lmi2 = lmi(listaToken2, bigrammiPOS2, BigrammiSostAgg2)

    #calcolo i nomi propri di luogo
    nomiLuoghi1 = nomiPropriLuogo(frasi1)
    nomiLuoghi2 = nomiPropriLuogo(frasi2)

    #calcolo i 20 nomi propri di luogo più frequenti
    Type_venti1 = VentiNE(nomiLuoghi1)
    Type_venti2 = VentiNE(nomiLuoghi2)
                                                                #OUTPUT
    print "\n"
    print "***************OUPUT PROGRAMMA 2***************"
    print "\n"

    print "------2O TOKEN PIU' FREQUENTI------"
    print "20 token piu' frequenti (esclusa la punteggiatura) del corpus dei racconti di viaggio delle donne:"
    stampaFrequenze_1 = stampaFrequenze(TokenFrequenti1)
    print
    print "20 token piu' frequenti (esclusa la punteggiatura) del corpus dei racconti di viaggio degli uomini:"
    stampaFrequenze_2 = stampaFrequenze(TokenFrequenti2)
    print "\n"

    print "------2O AGGETTIVI PIU' FREQUENTI------"
    print "20 aggettivi piu' frequenti del corpus dei racconti di viaggio delle donne: "
    stampaFrequenzeAggettivi_1 = stampaFrequenze(AggettiviFreq1)
    print
    print "20 aggettivi piu' frequenti del corpus dei racconti di viaggio degli uomini: "
    stampaFrequenzeAggettivi_2 = stampaFrequenze(AggettiviFreq2)
    print "\n"
    
    print "------2O VERBI PIU' FREQUENTI------"
    print "20 verbi piu' frequenti del corpus dei racconti di viaggio delle donne: "
    stampaFrequenzeVerbi_1 = stampaFrequenze(VerbiFreq1)
    print
    print "20 verbi piu' frequenti del corpus dei racconti di viaggio degli uomini: "
    stampaFrequenzeVerbi_2 = stampaFrequenze(VerbiFreq2)
    print "\n"
    
    print "------1O POS PIU' FREQUENTI------"
    print "10 POS piu' frequenti del corpus dei racconti di viaggio delle donne: "
    stampaFrequenzePOS_1 = stampaFrequenze(POSfreq1)
    print
    print "10 POS piu' frequenti del corpus dei racconti di viaggio degli uomini: "
    stampaFrequenzePOS_1 = stampaFrequenze(POSfreq2)
    print "\n"
    
    print "------1O TRIGRAMMI DI POS PIU' FREQUENTI------"
    print "10 trigrammi di POS piu' frequenti del corpus dei racconti di viaggio delle donne: "
    stampaTrigrammi_1 = stampaFrequenzeTrigrammi(trigrammiFreq1)
    print
    print "10 trigrammi di POS piu' frequenti del corpus dei racconti di viaggio degli uomini: \t"
    stampaTrigrammi_2 = stampaFrequenzeTrigrammi(trigrammiFreq2)
    print "\n"

    print "------1O BIGRAMMI DI POS CON PROBABILITA' CONDIZIONATA MASSIMA------"
    print "10 bigrammi di POS con probabilita' condizionata massima del corpus dei racconti di viaggio delle donne "
    stampaProb(probCondizionataBigrammi1)
    print
    print "10 bigrammi di POS con probabilita' condizionata massima del corpus dei racconti di viaggio degli uomini "
    stampaProb(probCondizionataBigrammi2)
    print"\n"
    
    print "------1O BIGRAMMI DI POS CON PROBABILITA' CONGIUNTA MASSIMA------"
    print "10 bigrammi di POS con probabilita' congiunta massima del corpus dei racconti di viaggio delle donne:"
    stampaProb(probCongiuntaBigrammi1)
    print
    print "10 bigrammi di POS con probabilita' congiunta massima del corpus dei racconti di viaggio degli uomini:"
    stampaProb(probCongiuntaBigrammi2)
    print "\n"
   
    print "------LOCAL MUTUAL INFORMATION (LMI) DEI BIGRAMMI SOSTANTIVO-AGGETTIVO------"
    print "LMI dei bigrammi Sostantivo-Aggettivo del corpus dei racconti di viaggio delle donne: "
    stampalmi(lmi1)
    print
    print "LMI dei bigrammi Sostantivo-Aggettivo del corpus dei racconti di viaggio degli uomini: "
    stampalmi(lmi2)
    print "\n"
    
    print "-----20 NOMI PROPRI DI LUOGO PIU' FREQUENTI-----"
    print "20 nomi di luogo piu' frequenti del corpus dei racconti di viaggio delle donne:"
    stampaType(Type_venti1)
    print
    print "20 nomi di luogo piu' frequenti del corpus dei racconti di viaggio degli uomini"
    stampaType(Type_venti2)

main(sys.argv[1], sys.argv[2])