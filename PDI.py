import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import math as m
from IPython.display import display, Math

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
            resultados[str(intervalo)+'%'] = [nivelDigital]
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
            palette="ch:r=-.2,d=.3_r"
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
    
    def getHistograma(self):
        nivelesVisuales = []
        for fila in range(len(self.banda.matriz)):
            for columna in range(len(self.banda.matriz[fila])):
                nivelesVisuales.append(self.banda.matriz[fila][columna])
        histograma = sb.displot(data=pd.DataFrame(np.array(nivelesVisuales)), binwidth=1, legend=False, facet_kws={'xlim':(0, 255)}, palette='mako')
        histograma.set(ylabel=None)
        histograma.set(title='Expansión Lineal Histograma')


class EcualizacionHistograma:
    banda:Banda = None
    bitsImagen: 8
    
    def __init__(self, banda:Banda, bitsImagen):
        self.banda = banda
        self.bitsImagen = bitsImagen

    def calculaNivelVisual(self, fAcumualada):
        nvCalculado = int(fAcumualada/(100/(m.pow(2,self.bitsImagen))))-1
        if nvCalculado < 0:
            return 0
        else:
            return nvCalculado
    
    def getTablaFrecuencias(self):
        tablaFrecuencias = self.banda.tablaFrecuentas()
        frecuenciaRelativa = tablaFrecuencias[['ND','f']]
        listaFrecuencias = frecuenciaRelativa.values.tolist()
        listadoNivelesDigitales = list(range(int(m.pow(2,self.bitsImagen))))
        nivelDigitales = []
        for fila in listaFrecuencias:
            nivelDigitales.append(fila[0])
        for nivel in listadoNivelesDigitales:
            if nivel not in nivelDigitales:
                listaFrecuencias.append([nivel,0])
        nuevoListadoFrecuencias = pd.DataFrame(np.array(listaFrecuencias))
        nuevoListadoFrecuencias.sort_values(by=[0],inplace=True)
        nuevoListadoFrecuencias = nuevoListadoFrecuencias.rename({0: 'ND',1: 'f'}, axis='columns')
        nuevoListadoFrecuencias = nuevoListadoFrecuencias.reset_index(drop=True)

        for index, row in nuevoListadoFrecuencias.iterrows():
            nuevoListadoFrecuencias.loc[index,'f(Porc)'] = (row['f']/len(listadoNivelesDigitales))*100
        
        nuevoListadoFrecuencias['F'] = 0
        nuevoListadoFrecuencias['F(Porc)'] = 0.0
        nuevoListadoFrecuencias['F(Obj)'] = 0.0
        nuevoListadoFrecuencias['NV'] = 0

        for index, row in nuevoListadoFrecuencias.iterrows():
            if index == 0:
                nuevoListadoFrecuencias.loc[index,'F(Obj)'] = 100*(index+1)/len(listadoNivelesDigitales)
                nuevoListadoFrecuencias.loc[index,'F'] =row['f']
            else:
                nuevoListadoFrecuencias.loc[index,'F(Obj)'] = 100*(index+1)/len(listadoNivelesDigitales)
                nuevoListadoFrecuencias.loc[index,'F'] = row['f'] + nuevoListadoFrecuencias.iloc[index-1]['F']
            
        for index, row in nuevoListadoFrecuencias.iterrows():
            nuevoListadoFrecuencias.loc[index,'F(Porc)'] = (row['F']/len(self.banda.nivelesDigitales))*100
        for index, row in nuevoListadoFrecuencias.iterrows():
            nuevoListadoFrecuencias.loc[index,'NV'] = self.calculaNivelVisual(row['F(Porc)'])
        return nuevoListadoFrecuencias
    
    def graficaNivelDigitalVisual(self):
        return sb.relplot(
            data=self.getTablaFrecuencias(),
            x="ND",
            y="NV",
            palette="ch:r=-.2,d=.3_r"
        )

    def getGraficaMatriz(self):
        matriz = self.banda.matriz
        tablaFrecuencias = self.getTablaFrecuencias().query('f > 0')
        nivelesVisuales = tablaFrecuencias[['ND','NV']].values.tolist()
        contador = 0
        for fila in range(len(matriz)):
            for columna in range(len(matriz[fila])):
                for niveles in nivelesVisuales:
                    if matriz[fila][columna] == niveles[0]:
                        matriz[fila][columna] = niveles[1]
                contador = contador +1
        self.arrayNP = np.array(matriz)
        self.dataframe =  pd.DataFrame(self.arrayNP)
        self.banda.matriz = matriz
        plt.figure(figsize=(10, 10))
        return sb.heatmap(self.dataframe, square=True, annot=True, xticklabels=[], yticklabels=[],fmt='g', vmin=0, vmax=255)

    def getHistograma(self):
        nivelesVisuales = []
        for fila in range(len(self.banda.matriz)):
            for columna in range(len(self.banda.matriz[fila])):
                nivelesVisuales.append(self.banda.matriz[fila][columna])
        histograma = sb.displot(data=pd.DataFrame(np.array(nivelesVisuales)), binwidth=1, legend=False, facet_kws={'xlim':(0, 255)}, palette='mako')
        histograma.set(ylabel=None)
        histograma.set(title='Histograma Ecualización Histograma')

class Filtros:
    bandaOriginal:Banda = Banda
    bandaExpandida:Banda = Banda
    bandaResultante:Banda = Banda
    
    def __init__(self, bandaOriginal):
        self.bandaOriginal = bandaOriginal
        self.bandaExpandida = bandaOriginal
        self.bandaResultante = bandaOriginal


    def expandirMatriz(self):
        banda = self.bandaOriginal.matriz
        bandaExpandida = []
        for fila in banda:
            fila.insert(0,fila[0])
            fila.append(fila[len(fila)-1])
            bandaExpandida.append(fila)
        bandaExpandida.insert(0,banda[0])
        bandaExpandida.append(banda[len(banda)-1])
        self.bandaExpandida.matriz = bandaExpandida
    
    def filtrarIntermedio(self, filtro):
        matrizExpandidaData = self.bandaExpandida.matriz.copy()
        matrizAuxiliar = []
        filtroSuma = 0

        for fila in range(len(filtro)):
            for columna in range(len(filtro[fila])):
                filtroSuma = filtroSuma + filtro[fila][columna]
        

        for fila in range(len(matrizExpandidaData)):
            filaData = []
            for columna in range(len(matrizExpandidaData[0])):
                if fila == 0 or fila == len(matrizExpandidaData)-1 or columna==0 or columna == len(matrizExpandidaData[0])-1:
                    pass
                else:
                    filaData.append(
                        round((
                        filtro[0][0]*matrizExpandidaData[fila-1][columna-1]+\
                        filtro[0][1]*matrizExpandidaData[fila-1][columna]+\
                        filtro[0][2]*matrizExpandidaData[fila-1][columna+1]+\
                        filtro[1][0]*matrizExpandidaData[fila][columna-1]+\
                        filtro[1][1]*matrizExpandidaData[fila][columna]+\
                        filtro[1][2]*matrizExpandidaData[fila][columna+1]+\
                        filtro[2][0]*matrizExpandidaData[fila+1][columna-1]+\
                        filtro[2][1]*matrizExpandidaData[fila+1][columna]+\
                        filtro[2][2]*matrizExpandidaData[fila+1][columna+1]
                        )/filtroSuma,2)
                    )
            matrizAuxiliar.append(filaData)
        matrizAuxiliar.remove(matrizAuxiliar[0])
        matrizAuxiliar.remove(matrizAuxiliar[len(matrizAuxiliar)-1])
        matrizIntermediaDF = pd.DataFrame(np.array(matrizAuxiliar))
        self.bandaResultante.matriz = matrizAuxiliar
        nivDigitalesResultante = []
        for fila in range(len(matrizAuxiliar)):
            for columna in range(len(matrizAuxiliar[0])):
                nivDigitalesResultante.append(matrizAuxiliar[fila][columna])
        self.bandaResultante.nivelesDigitales = nivDigitalesResultante
        return matrizIntermediaDF
    
class ComponentesPrincipales:
    listadoBandas:list[Banda] = []
    def __init__(self, listadoBandas:list[Banda]):
        self.listadoBandas = listadoBandas
    
    def getPromediosBandas(self):
        nombreBandas = []
        medias = []
        for banda in self.listadoBandas:
            nombreBandas.append(banda.getNombre())
            medias.append(banda.media())
        tablaBandasMedias = pd.DataFrame(np.array(medias))
        tablaBandasMedias.columns = ['Media']
        tablaBandasMedias.index = nombreBandas
        return tablaBandasMedias


    def impresionLatexCreacionMatrizVarianzaCovarianza(self):
        listadoBandas = self.listadoBandas
        cabecera = rf'''\documentclass{{article}}
                \usepackage[utf8]{{inputenc}}
                \usepackage{{amsmath}}
                \begin{{document}}
                '''
        latex = ''
        for fila in range(len(listadoBandas[0].matriz)):
            for columna in range(len(listadoBandas[0].matriz[fila])):
                filaColumna = '{Pixel :'+str(fila+1)+','+str(columna+1)+'}'
                a11 = round((listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media())*(listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media()),3)
                a12 = round((listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media())*(listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media()),3)
                a13 = round((listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media())*(listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media()),3)
                a14 = round((listadoBandas[3].matriz[fila][columna]-listadoBandas[3].media())*(listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media()),3)
                a21 = round((listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media())*(listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media()),3)
                a22 = round((listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media())*(listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media()),3)
                a23 = round((listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media())*(listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media()),3)
                a24 = round((listadoBandas[3].matriz[fila][columna]-listadoBandas[3].media())*(listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media()),3)
                a31 = round((listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media())*(listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media()),3)
                a32 = round((listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media())*(listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media()),3)
                a33 = round((listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media())*(listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media()),3)
                a34 = round((listadoBandas[3].matriz[fila][columna]-listadoBandas[3].media())*(listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media()),3)
                a41 = round((listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media())*(listadoBandas[3].matriz[fila][columna]-listadoBandas[0].media()),3)
                a42 = round((listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media())*(listadoBandas[3].matriz[fila][columna]-listadoBandas[0].media()),3)
                a43 = round((listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media())*(listadoBandas[3].matriz[fila][columna]-listadoBandas[0].media()),3)
                a44 = round((listadoBandas[3].matriz[fila][columna]-listadoBandas[3].media())*(listadoBandas[3].matriz[fila][columna]-listadoBandas[0].media()),3)
                latexMaultiplicacion = rf'''             
                \mathbf{filaColumna}
                \left ( \begin{{pmatrix}}
                {listadoBandas[0].matriz[fila][columna]}
                \\ {listadoBandas[1].matriz[fila][columna]}
                \\ {listadoBandas[2].matriz[fila][columna]}
                \\ {listadoBandas[3].matriz[fila][columna]}
                \end{{pmatrix}}
                -
                \begin{{pmatrix}}
                {round(listadoBandas[0].media(),3)}
                \\ {round(listadoBandas[1].media(),3)}
                \\ {round(listadoBandas[2].media(),3)}
                \\ {round(listadoBandas[3].media(),3)}
                \\ 
                \end{{pmatrix}}
                \right )

                \begin{{pmatrix}}
                 {round(listadoBandas[0].matriz[fila][columna]-listadoBandas[0].media(),3)}& {round(listadoBandas[1].matriz[fila][columna]-listadoBandas[1].media(),3)} & {round(listadoBandas[2].matriz[fila][columna]-listadoBandas[2].media(),3)} & {round(listadoBandas[3].matriz[fila][columna]-listadoBandas[3].media(),3)}
                \end{{pmatrix}}
                = \begin{{pmatrix}}
                {a11} & {a12} & {a13} & {a14}\\ 
                {a21} & {a22} & {a23} & {a24}\\ 
                {a31} & {a32} & {a33} & {a34}\\ 
                {a41} & {a42} & {a34} & {a44}
                \end{{pmatrix}}
                \\
                '''
                display(Math(latexMaultiplicacion))
    
