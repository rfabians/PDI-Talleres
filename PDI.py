import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import math as m


pd.set_option('display.max_rows', None)

class Banda:
    matriz = []
    nivelesDigitales = []
    nombre = ''
    def __init__(self, rutaArchivoBanda):
        self.name = self.obtenerNombre(rutaArchivoBanda)
        archivo = open(rutaArchivoBanda)
        lineasArchivo = archivo.readlines()
        self.matriz = []
        self.nivelesDigitales=[]
        for linea in lineasArchivo:
            linea = linea.replace('\n','')
            nivelesDigitales = linea.split('\t')
            filaMatriz = []
            for nivel in nivelesDigitales:
                filaMatriz.append(int(nivel))
                self.nivelesDigitales.append(int(nivel))
            self.matriz.append(filaMatriz)
        self.arrayNP = np.array(self.matriz)
        self.dataframe =  pd.DataFrame(self.arrayNP)
    
    def obtenerNombre(self, rutaArhivo:str):
        return rutaArhivo.split(r'/')[-1].replace('.txt','')


    def getGraficaMatriz(self):
        self.arrayNP = np.array(self.matriz)
        self.dataframe =  pd.DataFrame(self.arrayNP)
        plt.figure(figsize=(10, 10))
        return sb.heatmap(self.dataframe, square=True, annot=True, xticklabels=[], yticklabels=[],fmt='g', vmin=0, vmax=255)
        
    def histograma(self):
        histograma = sb.displot(data=pd.DataFrame(np.array(self.nivelesDigitales)), binwidth=1, legend=False, facet_kws={'xlim':(0, 255)}, palette='mako')
        histograma.set(ylabel=None)
        histograma.set(title='Histograma Banda '+self.name)
    
    def frecuencia(self):
        frecuencia={}
        for elemento in sorted(self.nivelesDigitales):
            if elemento not in frecuencia:
                frecuencia[elemento]=0
            frecuencia[elemento]+=1
        return frecuencia
    

    def frecuencia_relativa(self):
        frecuencias=self.frecuencia()
        n = len(self.nivelesDigitales)
        frecuenciasRelativas = []
        for frecuencia in frecuencias:
            frecuenciasRelativas.append(frecuencias[frecuencia]*100/n)
        return frecuenciasRelativas
    
    def frecuencia_acumulada(self):
        frecuencias = list(self.frecuencia().values())
        frecuenciaAcumulada = []
        contador = 0
        acumulada = 0
        for frecuencia in frecuencias:
            if contador == 0:
                frecuenciaAcumulada.append(frecuencia)
                contador = contador+1
                acumulada = frecuencia
            else:
                frecuenciaAcumulada.append(acumulada+frecuencia)
                acumulada = acumulada + frecuencia
        return frecuenciaAcumulada
    
    def frecuenciaAcumuladaPorcentual(self):
        frecunciasAcumuladas = self.frecuencia_acumulada()
        n = len(self.nivelesDigitales)
        frecuenciasProcentualesAcumuladas = []
        for frecuencia in frecunciasAcumuladas:
            frecuenciasProcentualesAcumuladas.append(100*frecuencia/n)
        return frecuenciasProcentualesAcumuladas

    def tablaFrecuentas(self):
        nivelesDigitalBanda = sorted(self.nivelesDigitales)
        nivelesDigitalBanda = sorted(list(set(nivelesDigitalBanda)))
        dataframe = pd.DataFrame(np.array(nivelesDigitalBanda))
        tablaFrecuencias = dataframe.rename({0: 'ND'}, axis='columns')
        frecuencias = list(self.frecuencia().values())
        tablaFrecuencias['f'] = frecuencias
        tablaFrecuencias['f(%)'] = self.frecuencia_relativa()
        tablaFrecuencias['F'] = self.frecuencia_acumulada()
        tablaFrecuencias['F(%)'] = self.frecuenciaAcumuladaPorcentual()
        return tablaFrecuencias
    
    def intervalos(self, listadoIntervalos:list):
        resultados = {}
        tablaFrecuencias = self.tablaFrecuentas()
        tablaFrecuencias = tablaFrecuencias.rename({'F(%)': 'FP'}, axis='columns')
        for intervalo in listadoIntervalos:
            nivelesDiscriminados = tablaFrecuencias.query('FP >= @intervalo')
            nivelDigital = nivelesDiscriminados.iloc[0]['ND']
            resultados[str(intervalo)+'%'] = [int(nivelDigital)]
        tablaIntervalos = pd.DataFrame().from_dict(resultados)
        tablaIntervalos.index = ['ND']
        return tablaIntervalos
    
    def media(self):
        return sum(self.nivelesDigitales)/len(self.nivelesDigitales)
    
    def mediana(self):
        Niveles_digitales=sorted(self.nivelesDigitales)
        n = len(self.nivelesDigitales)
        if n %2 == 0:
            return (Niveles_digitales[int((n-1)/2)]+Niveles_digitales[int(n/2)])/2
        else:
            return Niveles_digitales[int(n/2)]
    
    def maximo(self):
        return max(self.nivelesDigitales)
    
    def minimo(self):
        return min(self.nivelesDigitales)
    
    def varianza(self):
        media=self.media()
        desviacion=0
        for nivel_digital in self.nivelesDigitales:
            desviacion+=(nivel_digital-media)*(nivel_digital-media)
        return desviacion/len(self.nivelesDigitales)
    
    def desviacionEstandar(self):
        return m.sqrt(self.varianza())
    
    def moda(self):
        frecunciaMaxina =  max(self.frecuencia().values())
        moda = [nivelDigital for nivelDigital, frecuencia in self.frecuencia().items() if frecuencia == frecunciaMaxina]
        return moda[0]
    
    

    def getNombre(self):
        return self.name
    
    
class EstadisticaMultiBanda:
    
    def __init__(self, listadoBandas):
        self.bandas = listadoBandas
    
    def MedidasTendeciaCentral(self, listadoBandas:list[Banda]):
        titulosTabla =  ['Banda', 'Mínimo', 'Máximo','Media','Mediana','Moda','Desv. Estandar']
        bandasMedidas = []
        for banda in listadoBandas:    
            datos = [banda.getNombre(), banda.minimo(), banda.maximo(), banda.media(), banda.mediana(), banda.moda(), banda.desviacionEstandar()]
            bandasMedidas.append(datos)
        estadisticas = pd.DataFrame(np.array(bandasMedidas))
        estadisticas.columns = titulosTabla
        return estadisticas


    def covarianza(self, bandaA:Banda, bandaB:Banda):
        pixelesBandaA = bandaA.nivelesDigitales
        pixelesBandaB = bandaB.nivelesDigitales
        numeroPixeles = len(bandaB.nivelesDigitales)
        mediaBandaA = bandaA.media()
        mediaBandaB = bandaB.media()
        covarianzaParcial = 0
        for indice in range(numeroPixeles):
            covarianzaParcial = covarianzaParcial + (pixelesBandaA[indice]-mediaBandaA)*(pixelesBandaB[indice]-mediaBandaB)/numeroPixeles
        return covarianzaParcial
            
    def matrizCovarianza(self):
        matrizCovarianza = []
        nombresBandas = []
        for bandaA in self.bandas:
            filaCovarianza = []
            nombresBandas.append(bandaA.getNombre())
            for bandaB in self.bandas:
                cov = self.covarianza(bandaA, bandaB)
                filaCovarianza.append(cov)
            matrizCovarianza.append(filaCovarianza)
        
        matrizCovarianza = pd.DataFrame(np.array(matrizCovarianza))
        matrizCovarianza.index = nombresBandas
        matrizCovarianza.columns = nombresBandas
        return matrizCovarianza
    
    def matrizCorrelacion(self):
        matrizCorrelacion = []
        nombresBandas = []
        for bandaA in self.bandas:
            filaCorrelacion = []
            nombresBandas.append(bandaA.getNombre())
            for bandaB in self.bandas:
                correlacion = self.covarianza(bandaA, bandaB)/(bandaA.desviacionEstandar()*bandaB.desviacionEstandar())
                filaCorrelacion.append(correlacion)
            matrizCorrelacion.append(filaCorrelacion)
        matrizColleracion = pd.DataFrame(np.array(matrizCorrelacion))
        matrizColleracion.index = nombresBandas
        matrizColleracion.columns = nombresBandas
        return matrizColleracion
    
class ExpansionLineal(Banda):
    banda:Banda = None
    porcentajeMinMax = []
    
    def __init__(self, banda:Banda, porcentajeMinMax):
        self.banda = banda
        self.porcentajeMinMax = porcentajeMinMax
    
    def calcularNivelVisualExpansionLineal(self, minimo, maximo, nd):
        if nd >= maximo:
            return 255
        elif nd <= minimo:
            return 0
        else:
            return m.trunc((255/(maximo-minimo))*(nd-minimo))

    def reclasificacionValores(self):
        nivelDigitalMin = self.banda.intervalos([self.porcentajeMinMax, 100-self.porcentajeMinMax])
        nivelesMinimosMaximos = nivelDigitalMin.to_dict()
        minimo = nivelesMinimosMaximos[str(self.porcentajeMinMax)+'%']['ND']
        maximo = nivelesMinimosMaximos[str(100-self.porcentajeMinMax)+'%']['ND']
        tablaFrecuencias = self.banda.tablaFrecuentas()
        tablaFrecuencias['NV'] = None
        for index, row in tablaFrecuencias.iterrows():
            tablaFrecuencias.loc[index,'NV'] = self.calcularNivelVisualExpansionLineal(minimo, maximo, row['ND'])
        return tablaFrecuencias
    
    def graficaNivelDigitalVisual(self):
        return sb.relplot(
            data=self.reclasificacionValores(),
            x="ND",
            y="NV",
            palette="ch:r=-.2,d=.3_r",
            kind="line"
        )

    def getStrechBanda(self, banda:Banda):
        bandaVisual  = banda
        matrizNivelVisual = bandaVisual.matriz 
        nivelDigitalMin = self.banda.intervalos([self.porcentajeMinMax, 100-self.porcentajeMinMax])
        nivelesMinimosMaximos = nivelDigitalMin.to_dict()
        minimo = nivelesMinimosMaximos[str(self.porcentajeMinMax)+'%']['ND']
        maximo = nivelesMinimosMaximos[str(100-self.porcentajeMinMax)+'%']['ND']
        
        for fila in range(len(matrizNivelVisual)):
            for columna in range(len(matrizNivelVisual[fila])):
                matrizNivelVisual[fila][columna] = self.calcularNivelVisualExpansionLineal(minimo, maximo, matrizNivelVisual[fila][columna])
        bandaVisual.matriz = matrizNivelVisual

        for nivelDigital in range(len(bandaVisual.nivelesDigitales)):
            bandaVisual.nivelesDigitales[nivelDigital] = self.calcularNivelVisualExpansionLineal(minimo, maximo, bandaVisual.nivelesDigitales[nivelDigital])
        return bandaVisual