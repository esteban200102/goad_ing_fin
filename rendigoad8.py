import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from PIL import Image

# Función para descargar datos financieros
def descargar_datos(tickers, fecha_inicio, fecha_final):
    data = yf.download(tickers, start=fecha_inicio, end=fecha_final)['Adj Close']
    return data

# Función para mostrar gráficos
def mostrar_graficos(data):
    data = (data / data.iloc[0]) * 100  # Normalizar en base 100
    st.line_chart(data)

# Función para obtener la tasa libre de riesgo en tiempo real desde Yahoo Finance
def obtener_tasa_libre_riesgo():
    tasa_libre_riesgo_data = yf.Ticker("10Y=F").history(period="1d")
    tasa_libre_riesgo = tasa_libre_riesgo_data['Close'].iloc[-1] / 100  # Convertir a porcentaje
    return tasa_libre_riesgo

def main():
    image_path = "logoad.png"  # Ensure this path is correct
    
    # Abrir la imagen y mostrarla en Streamlit
    try:
        image = Image.open(image_path)
        st.image(image, caption='Adolfo Usuario | 0233486', use_column_width=True)
    except FileNotFoundError:
        st.error("Error: Imagen no encontrada. Por favor verifica la ruta del archivo de imagen.")

if __name__ == "__main__":
    main()

st.divider()

st.header("Investment Innovation")
st.write("""
Bienvenido a Investment Innovation, la app que optimizará tu camino hacia inversiones inteligentes y diversificadas. 
En esta plataforma, te ayudaremos a construir un portafolio con una adecuada relación riesgo-retorno basado en los mejores ETFs del mercado estadounidense administrados por BlackRock.
""")

st.write("### ¿Qué es la bolsa de valores? 📈")
st.write("""
En términos simples, la bolsa de valores es un mercado donde se compran y venden acciones de empresas. 
Las acciones son pequeños pedazos de propiedad de una empresa, y cuando compras una acción, te conviertes en parte dueño de esa empresa. 
El precio de una acción sube y baja según la oferta y la demanda, y el objetivo de los inversores es comprar acciones a bajo precio y venderlas a un precio más alto para obtener ganancias.
""")

st.write("### ¿Qué es el SPY? 📊")
st.write("""
El SPY, también conocido como el SPDR S&P 500 ETF, es un fondo cotizado en bolsa (ETF) que rastrea el índice S&P 500. 
El S&P 500 es un índice bursátil que incluye las 500 empresas más grandes de los Estados Unidos por capitalización de mercado. 
Esto significa que el SPY invierte en acciones de estas 500 empresas en la misma proporción en que se encuentran representadas en el índice.
""")

st.write("### Importancia del SPY 💡")
st.write("""
El SPY se considera un indicador general de la salud de la economía estadounidense. 
Cuando el SPY sube, generalmente significa que la economía está funcionando bien y que las empresas están obteniendo ganancias. 
Por el contrario, cuando el SPY baja, generalmente significa que la economía está teniendo dificultades y que las empresas están teniendo problemas.
""")

# Variables de los ETFs
etfs = ["SOXX", "IYW", "IGV", "OEF", "IYF", "IYC", "IYJ", "IYE", "IYH", "IYM", "IYR", "IYK", "IDU", "IYZ"]
mapa_tickers_nombres = {
    "SPY": "SPDR S&P 500 ETF",
    "SOXX": "iShares PHLX SOX Semiconductor Sector Index Fund",
    "IYW": "iShares U.S. Technology ETF",
    "IGV": "iShares Expanded Tech-Software Sector ETF",
    "OEF": "iShares S&P 100 ETF",
    "IYF": "iShares U.S. Financials ETF",
    "IYC": "iShares U.S. Consumer Services ETF",
    "IYJ": "iShares U.S. Industrials ETF",
    "IYE": "iShares U.S. Energy ETF",
    "IYH": "iShares U.S. Healthcare ETF",
    "IYM": "iShares U.S. Basic Materials ETF",
    "IYR": "iShares U.S. Real Estate ETF",
    "IYK": "iShares U.S. Consumer Goods ETF",
    "IDU": "iShares U.S. Utilities ETF",
    "IYZ": "iShares U.S. Telecommunications ETF"
}

# Crear lista desplegable con nombre y ticker
etfs_options = [f"{ticker} - {mapa_tickers_nombres[ticker]}" for ticker in etfs]

# Selección de ETFs por el usuario
st.write("### Selecciona los ETFs para el análisis")
etfs_seleccionados = st.multiselect("Selecciona uno o varios ETFs:", etfs_options, default=etfs_options)

# Verificar que se haya seleccionado al menos un ETF
if not etfs_seleccionados:
    st.warning("Debe seleccionar al menos un ETF para continuar.")
    st.stop()

# Extraer tickers seleccionados
etfs_seleccionados_tickers = [etf.split(" - ")[0] for etf in etfs_seleccionados]

# Fechas dinámicas
fecha_final = datetime.datetime.now()
fecha_inicio = fecha_final - datetime.timedelta(days=365*5)  # Aproximadamente 5 años

# Incluir siempre "SPY"
tickers = ["SPY"] + etfs_seleccionados_tickers
data = descargar_datos(tickers, fecha_inicio, fecha_final)

# Mostrar gráficos
mostrar_graficos(data)

# Verificar que todos los tickers seleccionados están en los datos descargados
etfs_seleccionados_tickers = [ticker for ticker in etfs_seleccionados_tickers if ticker in data.columns]

# Cálculo de rendimientos diarios
rendimientos = (data.pct_change().dropna())

# Calculando el retorno y el riesgo anual para cada Acción
retorno_anual = round((((rendimientos.mean() + 1) ** 252) - 1) * 100, 2)
riesgo_anual = round((rendimientos.std() * (252 ** 0.5)) * 100, 2)

# Creando un DataFrame inicial con tickers y nombres
df = pd.DataFrame({'Ticker': tickers, 'Name': [mapa_tickers_nombres.get(ticker, ticker) for ticker in tickers]})

# Asignando los rendimientos y riesgos al DataFrame usando el ticker como índice
df['Yield'] = df['Ticker'].map(retorno_anual)
df['Risk'] = df['Ticker'].map(riesgo_anual)

# Ordenando el DataFrame por rendimiento
df = df.sort_values(by='Yield', ascending=False)

# Mostrando la tabla unificada de "Riesgo y Rendimiento"
st.write("## Relación Riesgo y Rendimiento Anual")
st.write(df)

st.write("""
La tabla muestra la relación entre el rendimiento anual y el riesgo anual para cada ETF. 
Esto ayuda a identificar cuáles ETFs tienen un mejor rendimiento en relación con su riesgo.
""")

# Estableciendo el tamaño de la figura
plt.figure(figsize=(16, 10))  # Tamaño más grande para una visualización más llamativa

# Asignar variables
x = df['Risk']
y = df['Yield']
s = df['Risk'] * 10  # Escalamos el tamaño para mejor visualización
labels = df['Ticker']
names = df['Name']

# Dibujar las burbujas
scatter = plt.scatter(x, y, s=s, alpha=0.6, c=np.arange(len(x)))

# Agregar etiquetas para cada burbuja
for i, txt in enumerate(labels):
    plt.annotate(f"{txt} ({names.iloc[i]})", (x.iloc[i], y.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Personalización del gráfico
plt.title('Relación entre Riesgo Anual Y Retorno Anual')
plt.xlabel('Riesgo Anual (%)')
plt.ylabel('Retorno Anual (%)')
plt.grid(True, linestyle='--', linewidth=0.5)

# Crear leyenda con los nombres y tickers de cada ETF
legend_labels = [f"{label} ({name})" for label, name in zip(labels, names)]
plt.legend(handles=scatter.legend_elements()[0], title="ETFs", labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
st.pyplot(plt)

st.write("""
Este gráfico de burbujas muestra la relación entre el riesgo anual y el rendimiento anual para cada ETF. 
Cada burbuja representa un ETF, y su tamaño está relacionado con el riesgo del ETF. 
Esto permite identificar visualmente qué ETFs ofrecen un mejor rendimiento ajustado por riesgo.
""")

# Continuación del análisis
spy_rendimientos = rendimientos['SPY']

# Inicializar un diccionario para almacenar los resultados
resultados = {'Ticker': [], 'Alpha': [], 'Beta': [], 'Riesgo': []}

for ticker in rendimientos.columns:
    if ticker != 'SPY':  # Evitar la regresión de SPY consigo mismo
        y = rendimientos[ticker]
        X = sm.add_constant(spy_rendimientos)  # Añadir una constante para calcular el intercepto
        # Realizar la regresión lineal
        model = sm.OLS(y, X).fit()

        # Calcular los errores residuales
        errores = y - model.predict(X)

        # Extraer alpha y beta
        alpha_diario, beta = model.params

        betaRounded = round(beta, 4)

        # Anualizar el alpha
        alpha_anualizado = round(((1 + alpha_diario)**252 - 1)*100, 3)

        # Calcular la volatilidad diaria de los errores
        volatilidad_diaria_errores = errores.std()

        # Anualizar la volatilidad
        volatilidad_anualizada_errores = volatilidad_diaria_errores * (252 ** 0.5)

        # Almacenar la volatilidad anualizada en lugar del riesgo total acumulado
        riesgo_total_acumulado_anualizado = round(volatilidad_anualizada_errores * 100, 4)

        # Almacenar los resultados
        resultados['Ticker'].append(ticker)
        resultados['Alpha'].append(alpha_anualizado)
        resultados['Beta'].append(betaRounded)
        resultados['Riesgo'].append(riesgo_total_acumulado_anualizado)

# Convertir el diccionario a un DataFrame para una mejor visualización
resultados_df = pd.DataFrame(resultados)

resultados_df['Nombre_ETF'] = resultados_df['Ticker'].map(mapa_tickers_nombres)

# Reorganizar las columnas para insertar 'Nombre_ETF' como la segunda columna
columnas = resultados_df.columns.tolist()
columnas.remove('Nombre_ETF')
nueva_posicion = 1  # La posición donde queremos insertar 'Nombre_ETF', después de 'Ticker' que está en la posición 0
columnas.insert(nueva_posicion, 'Nombre_ETF')

# Reordenar el dataframe
resultados_df = resultados_df[columnas]

# Ordenar el dataframe por 'Alpha Anualizado'
resultados_df.sort_values(by='Alpha', ascending=False, inplace=True)

st.write("## Relación entre Alpha Anualizado y Riesgo Anual")
st.write(resultados_df)

st.write("""
La tabla muestra la relación entre el alpha anualizado y el riesgo anual para cada ETF. 
El alpha mide el rendimiento ajustado por riesgo de un ETF en comparación con el índice de referencia (SPY), 
mientras que el beta mide la sensibilidad del ETF respecto al mercado.
""")

# Crear el gráfico de burbujas para Alpha vs. Riesgo
plt.figure(figsize=(16, 10))  # Tamaño más grande para una visualización más llamativa

# Asignar variables
x = resultados_df['Riesgo']
y = resultados_df['Alpha']
s = resultados_df['Riesgo'] * 10  # Escalamos el tamaño para mejor visualización
labels = resultados_df['Ticker']
names = resultados_df['Nombre_ETF']

# Dibujar las burbujas
scatter = plt.scatter(x, y, s=s, alpha=0.6, c=np.arange(len(x)))

# Agregar etiquetas para cada burbuja
for i, txt in enumerate(labels):
    plt.annotate(f"{txt} ({names.iloc[i]})", (x.iloc[i], y.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Personalización del gráfico
plt.title('Relación entre Alpha Anualizado y Riesgo Anual')
plt.xlabel('Riesgo Anual (%)')
plt.ylabel('Alpha Anualizado (%)')
plt.grid(True, linestyle='--', linewidth=0.5)

# Crear leyenda con los nombres y tickers de cada ETF
legend_labels = [f"{label} ({name})" for label, name in zip(labels, names)]
plt.legend(handles.scatter.legend_elements()[0], title="ETFs", labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
st.pyplot(plt)

st.write("""
Este gráfico de burbujas muestra la relación entre el alpha anualizado y el riesgo anual para cada ETF. 
Cada burbuja representa un ETF, y su tamaño está relacionado con el riesgo del ETF. 
Esto permite identificar visualmente qué ETFs ofrecen un mejor rendimiento ajustado por riesgo (alpha) en comparación con su volatilidad.
""")

# Función para calcular el rendimiento y riesgo de un portafolio dado
def portafolio_performance(pesos, expected_returns, cov_matrix):
    rendimiento = np.dot(pesos, expected_returns)
    riesgo = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
    return rendimiento, riesgo

# Función para la optimización de la frontera eficiente
def frontera_eficiente(expected_returns, cov_matrix, num_portafolios=10000, riesgo_libre=0.02):
    num_assets = len(expected_returns)
    resultados = np.zeros((3, num_portafolios))
    pesos_registrados = []

    for i in range(num_portafolios):
        pesos = np.random.random(num_assets)
        pesos /= np.sum(pesos)
        rendimiento, riesgo = portafolio_performance(pesos, expected_returns, cov_matrix)
        resultados[0,i] = rendimiento
        resultados[1,i] = riesgo
        resultados[2,i] = (rendimiento - riesgo_libre) / riesgo
        pesos_registrados.append(pesos)

    return resultados, pesos_registrados

# Selección de ETFs para la simulación de portafolio
etfs_portafolio = etfs_seleccionados_tickers

# Cantidad a invertir
cantidad_inversion = st.number_input("Ingresa la cantidad a invertir ($):", min_value=1000, value=10000, step=500)

if len(etfs_portafolio) > 1:
    # Filtrar los datos de rendimientos para los ETFs seleccionados
    rendimientos_portafolio = rendimientos[etfs_portafolio]

    # Calcular las matrices de covarianza y los rendimientos esperados
    cov_matrix = rendimientos_portafolio.cov() * 252
    expected_returns = rendimientos_portafolio.mean() * 252

    # Obtener los resultados de la optimización de la frontera eficiente
    resultados, pesos_registrados = frontera_eficiente(expected_returns, cov_matrix)
    
    # Encontrar el portafolio con mayor relación Sharpe
    indice_sharpe_max = np.argmax(resultados[2])
    pesos_sharpe_max = pesos_registrados[indice_sharpe_max]
    rendimiento_sharpe_max, riesgo_sharpe_max = portafolio_performance(pesos_sharpe_max, expected_returns, cov_matrix)
    
    # Encontrar el portafolio de mínima varianza
    indice_min_var = np.argmin(resultados[1])
    pesos_min_var = pesos_registrados[indice_min_var]
    rendimiento_min_var, riesgo_min_var = portafolio_performance(pesos_min_var, expected_returns, cov_matrix)
    
    # Encontrar el portafolio moderado
    pesos_moderado = (pesos_sharpe_max + pesos_min_var) / 2
    rendimiento_moderado, riesgo_moderado = portafolio_performance(pesos_moderado, expected_returns, cov_matrix)

    # Calcular el VaR del portafolio con los pesos óptimos
    VaR_opt = np.percentile(rendimientos_portafolio.sum(axis=1), 5) * cantidad_inversion

    # Calcular los márgenes de error
    margen_error_rendimiento_sharpe = rendimiento_sharpe_max * 0.02  # ±2%
    margen_error_riesgo_sharpe = riesgo_sharpe_max * 0.02  # ±2%
    margen_error_rendimiento_min_var = rendimiento_min_var * 0.02  # ±2%
    margen_error_riesgo_min_var = riesgo_min_var * 0.02  # ±2%
    margen_error_rendimiento_moderado = rendimiento_moderado * 0.02  # ±2%
    margen_error_riesgo_moderado = riesgo_moderado * 0.02  # ±2%
    margen_error_VaR_opt = VaR_opt * 0.02  # ±2%

    # Mostrar los resultados en una tabla informativa
    st.write("### Resultados del Portafolio (Modelo Markowitz)")
    resultados_portafolio = {
        "Métrica": ["Rendimiento Esperado (Max Sharpe)", "Riesgo del Portafolio (Max Sharpe) (±2%)", 
                    "Rendimiento Esperado (Min Varianza)", "Riesgo del Portafolio (Min Varianza) (±2%)", 
                    "Rendimiento Esperado (Moderado)", "Riesgo del Portafolio (Moderado) (±2%)", 
                    "Value at Risk (VaR) (±2%)"],
        "Valor": [
            f"${round(rendimiento_sharpe_max * cantidad_inversion, 2)} ± ${round(margen_error_rendimiento_sharpe * cantidad_inversion, 2)} ({round(rendimiento_sharpe_max * 100, 2)}%)",
            f"{round(riesgo_sharpe_max * 100, 2)}% ± {round(margen_error_riesgo_sharpe * 100, 2)}%",
            f"${round(rendimiento_min_var * cantidad_inversion, 2)} ± ${round(margen_error_rendimiento_min_var * cantidad_inversion, 2)} ({round(rendimiento_min_var * 100, 2)}%)",
            f"{round(riesgo_min_var * 100, 2)}% ± {round(margen_error_riesgo_min_var * 100, 2)}%",
            f"${round(rendimiento_moderado * cantidad_inversion, 2)} ± ${round(margen_error_rendimiento_moderado * cantidad_inversion, 2)} ({round(rendimiento_moderado * 100, 2)}%)",
            f"{round(riesgo_moderado * 100, 2)}% ± {round(margen_error_riesgo_moderado * 100, 2)}%",
            f"${round(VaR_opt, 2)} ± ${round(margen_error_VaR_opt, 2)} ({round((VaR_opt / cantidad_inversion) * 100, 2)}%)"
        ]
    }
    resultados_df_portafolio = pd.DataFrame(resultados_portafolio)
    st.table(resultados_df_portafolio)

    st.write("""
    La tabla muestra los resultados del portafolio utilizando el modelo de Markowitz. 
    Se presentan los rendimientos esperados y los riesgos para el portafolio de máxima relación Sharpe, 
    el portafolio de mínima varianza y un portafolio moderado, así como el Valor en Riesgo (VaR) del portafolio.
    """)

    # Crear una gráfica de la ponderación del portafolio de máxima relación Sharpe
    st.write("### Ponderación del Portafolio (Max Sharpe)")
    plt.figure(figsize=(10, 6))
    plt.pie(pesos_sharpe_max, labels=etfs_portafolio, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
    plt.axis('equal')
    st.pyplot(plt)

    st.write("""
    La gráfica muestra la ponderación del portafolio de máxima relación Sharpe. 
    Este portafolio busca maximizar la relación Sharpe, que es la medida de rendimiento ajustado por riesgo.
    """)

    # Crear una gráfica de la ponderación del portafolio de mínima varianza
    st.write("### Ponderación del Portafolio (Min Varianza)")
    plt.figure(figsize=(10, 6))
    plt.pie(pesos_min_var, labels=etfs_portafolio, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
    plt.axis('equal')
    st.pyplot(plt)

    st.write("""
    La gráfica muestra la ponderación del portafolio de mínima varianza. 
    Este portafolio busca minimizar el riesgo total del portafolio.
    """)

    # Crear una gráfica de la ponderación del portafolio moderado
    st.write("### Ponderación del Portafolio (Moderado)")
    plt.figure(figsize=(10, 6))
    plt.pie(pesos_moderado, labels=etfs_portafolio, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
    plt.axis('equal')
    st.pyplot(plt)

    st.write("""
    La gráfica muestra la ponderación del portafolio moderado, que es una combinación de los portafolios de máxima relación Sharpe y mínima varianza.
    """)

    # Gráfica del VaR utilizando una distribución normal
    st.write("### VaR del Portafolio utilizando una Distribución Normal")
    plt.figure(figsize=(10, 6))
    mu, sigma = rendimiento_sharpe_max, riesgo_sharpe_max
    s = np.random.normal(mu, sigma, 10000)
    plt.hist(s, bins=50, alpha=0.75, color='blue')
    plt.axvline(x=np.percentile(s, 5), color='red', linestyle='--', label='VaR al 5%')
    plt.title('Distribución Normal del VaR del Portafolio')
    plt.xlabel('Rendimiento')
    plt.ylabel('Frecuencia')
    plt.legend()
    st.pyplot(plt)

    st.write("""
    La gráfica muestra la distribución normal del Valor en Riesgo (VaR) del portafolio. 
    La línea roja discontinua representa el VaR al 5%, indicando que hay un 5% de probabilidad de que las pérdidas del portafolio excedan este valor en un día determinado.
    """)

    # Cálculo del rendimiento y riesgo del portafolio utilizando CAPM
    st.write("### Simulación de Inversión con el Modelo CAPM")

    tasa_libre_riesgo = obtener_tasa_libre_riesgo()  # Obtener tasa libre de riesgo en tiempo real
    rendimiento_mercado = retorno_anual['SPY'] / 100  # Rendimiento esperado del mercado

    rendimiento_capm = []
    for ticker in etfs_portafolio:
        beta = resultados_df.loc[resultados_df['Ticker'] == ticker, 'Beta'].values[0]
        rendimiento = tasa_libre_riesgo + beta * (rendimiento_mercado - tasa_libre_riesgo)
        rendimiento_capm.append(rendimiento)

    rendimiento_portafolio_capm, riesgo_portafolio_capm = portafolio_performance(pesos_sharpe_max, rendimiento_capm, cov_matrix)

    # Calcular el rendimiento esperado en cantidad de dinero para CAPM
    rendimiento_esperado_dinero_capm = rendimiento_portafolio_capm * cantidad_inversion

    # Calcular los márgenes de error
    margen_error_rendimiento_capm = rendimiento_portafolio_capm * 0.02  # ±2%
    margen_error_riesgo_capm = riesgo_portafolio_capm * 0.02  # ±2%

    # Mostrar los resultados del CAPM en una tabla informativa
    st.write("### Resultados del Portafolio (Modelo CAPM)")
    resultados_portafolio_capm = {
        "Métrica": ["Rendimiento Esperado (CAPM)", "Riesgo del Portafolio (±2%)"],
        "Valor": [
            f"${round(rendimiento_esperado_dinero_capm, 2)} ± ${round(margen_error_rendimiento_capm * cantidad_inversion, 2)} ({round(rendimiento_portafolio_capm * 100, 2)}%)",
            f"{round(riesgo_portafolio_capm * 100, 2)}% ± {round(margen_error_riesgo_capm * 100, 2)}%"
        ]
    }
    resultados_df_portafolio_capm = pd.DataFrame(resultados_portafolio_capm)
    st.table(resultados_df_portafolio_capm)

    st.write("""
    La tabla muestra los resultados del portafolio utilizando el modelo CAPM. 
    Se presentan el rendimiento esperado y el riesgo del portafolio basado en el rendimiento esperado del mercado y la sensibilidad del portafolio al mercado (beta).
    """)

else:
    st.warning("Debe seleccionar al menos un ETF para generar un portafolio óptimo.")
    st.stop()

# Descripción de los modelos utilizados
st.write("### Anexos Informativos")
st.write("""
**Modelo de Markowitz**: 
El Modelo de Markowitz, también conocido como la Teoría de la Cartera Moderna, es un modelo de optimización de portafolios que busca maximizar el rendimiento esperado para un nivel dado de riesgo o minimizar el riesgo para un nivel dado de rendimiento esperado. Utiliza la varianza de los retornos de las inversiones como una medida del riesgo y considera la correlación entre los activos para construir un portafolio diversificado.

**Modelo CAPM (Capital Asset Pricing Model)**:
El Modelo de Valoración de Activos Financieros (CAPM) es un modelo que describe la relación entre el riesgo sistemático y el rendimiento esperado de los activos. Se utiliza para estimar el rendimiento esperado de una inversión dado su beta (medida de la sensibilidad del activo respecto al mercado), la tasa libre de riesgo y el rendimiento esperado del mercado. El CAPM ayuda a los inversores a evaluar el rendimiento de una inversión en comparación con su riesgo.

**Interpretación del riesgo y rendimiento del portafolio**:
Los valores de rendimiento y riesgo presentados en este análisis están sujetos a variabilidad y cambios en el mercado. Los márgenes de error proporcionados (±2%) permiten a los inversores tener una idea de los diferentes escenarios posibles y entender que las estimaciones están sujetas a cambios debido a la volatilidad del mercado.

**Explicación de la Gráfica del VaR**:
La gráfica del VaR muestra cómo se distribuye el Valor en Riesgo del portafolio utilizando una distribución normal. La línea roja discontinua representa el VaR al 5%, lo que significa que hay un 5% de probabilidad de que las pérdidas del portafolio excedan este valor en un día determinado. Esta gráfica ayuda a los inversores a visualizar el riesgo de pérdidas extremas y tomar decisiones informadas sobre su tolerancia al riesgo.
""")
