
import PDI

rutaBandaB17 = 'Bandas/B17.txt'
rutaBandaB27 = 'Bandas/B27.txt'
rutaBandaB37 = 'Bandas/B37.txt'
rutaBandaB47 = 'Bandas/B47.txt'

if __name__ == '__main__':
    bandaB17 = PDI.Banda(rutaArchivoBanda=rutaBandaB17)
    bandaB27 = PDI.Banda(rutaArchivoBanda=rutaBandaB27)
    bandaB37 = PDI.Banda(rutaArchivoBanda=rutaBandaB37)
    bandaB47 = PDI.Banda(rutaArchivoBanda=rutaBandaB47)

    listadobandas = [bandaB17, bandaB27, bandaB37, bandaB47]

    componentesPrincipales = PDI.ComponentesPrincipales(listadoBandas=listadobandas)
    print(componentesPrincipales.getPromediosBandas())
    componentesPrincipales.MatrizCovarianza()
    componentesPrincipales.PoligonomioCaracteristico()
