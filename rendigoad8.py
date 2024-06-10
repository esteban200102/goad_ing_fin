import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# Datos de ejemplo ampliados con múltiples tickers por sector
data = {
    "Sector": ["Tecnología"] * 15 + ["Financiero"] * 15 + ["Industrial"] * 15 + ["Salud"] * 15 + ["Consumo básico"] * 15,
    "Ticker": [
        # Tecnología
        "AAPL", "MSFT", "GOOGL", "FB", "TSLA", "NVDA", "AMD", "INTC", "ORCL", "SAP", 
        "IBM", "QCOM", "TXN", "ADBE", "CRM",
        # Financiero
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SPGI", "COF",
        "USB", "PNC", "TFC", "CB", "BK",
        # Industrial
        "GE", "MMM", "HON", "CAT", "LMT", "BA", "DE", "RTX", "DAL", "UPS",
        "GD", "PCAR", "CMI", "EMR", "ETN",
        # Salud
        "JNJ", "PFE", "UNH", "MRK", "ABT", "ABBV", "AMGN", "MDT", "DHR", "LLY",
        "BMY", "CI", "GILD", "SYK", "ISRG",
        # Consumo básico
        "PG", "KO", "PEP", "WMT", "COST", "MO", "PM", "MDLZ", "EL", "CL",
        "KMB", "GIS", "KHC", "KR", "SYY"
    ],
    "Company": [
        # Tecnología
        "Apple", "Microsoft", "Alphabet", "Facebook", "Tesla", "NVIDIA", "AMD", "Intel", "Oracle", "SAP",
        "IBM", "Qualcomm", "Texas Instruments", "Adobe", "Salesforce",
        # Financiero
        "JP Morgan", "Bank of America", "Wells Fargo", "Citigroup", "Goldman Sachs", "Morgan Stanley", "American Express", "BlackRock", "S&P Global", "Capital One",
        "US Bancorp", "PNC Financial Services", "Truist Financial", "Chubb", "Bank of New York Mellon",
        # Industrial
        "General Electric", "3M", "Honeywell", "Caterpillar", "Lockheed Martin", "Boeing", "Deere & Co", "Raytheon Technologies", "Delta Air Lines", "UPS",
        "General Dynamics", "PACCAR", "Cummins", "Emerson Electric", "Eaton",
        # Salud
        "Johnson & Johnson", "Pfizer", "UnitedHealth", "Merck", "Abbott Laboratories", "AbbVie", "Amgen", "Medtronic", "Danaher", "Eli Lilly",
        "Bristol Myers Squibb", "Cigna", "Gilead Sciences", "Stryker", "Intuitive Surgical",
        # Consumo básico
        "Procter & Gamble", "Coca-Cola", "PepsiCo", "Walmart", "Costco", "Altria", "Philip Morris", "Mondelez", "Estee Lauder", "Colgate-Palmolive",
        "Kimberly-Clark", "General Mills", "Kraft Heinz", "Kroger", "Sysco"
    ]
}

# Crear un DataFrame a partir de los datos
df_tickers = pd.DataFrame(data)
df_tickers['Display'] = df_tickers['Ticker'] + ' - ' + df_tickers['Company']

# Función para obtener la imagen de la noticia desde la URL
def obtener_imagen_desde_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tag = soup.find('img')
        if img_tag:
            img_url = img_tag['src']
            img_response = requests.get(img_url)
            img = Image.open(BytesIO(img_response.content))
            return img
    except Exception as e:
        st.warning("No se pudo obtener la imagen.")
    return None

# Configuración inicial de la página
st.set_page_config(page_title="GO Advisors", page_icon=":chart_with_upwards_trend:", layout="wide")

# Distribución de logo y menú en el encabezado
col1, col2 = st.columns([2, 7])
with col1:
    st.image("logoad.png", width=200)  # Ajusta el tamaño según sea necesario
with col2:
    menu_items = ["Inicio", "Proyecta tu Inversión", "Noticias de Mercado", "Mercado Financiero", "Cartera Eficiente"]
    menu_selection = st.radio("Menú", menu_items, index=0, help="Seleccione una opción para navegar por la página")

# Manejo de las selecciones del menú
if menu_selection == "Inicio":
    st.markdown("""
    # Bienvenido a GO ADVISORS
    ## Somos el camino que te ayudará a cumplir todas tus metas financieras
    """)
elif menu_selection == "Proyecta tu Inversión":
    st.markdown("# Proyecta tu Inversión")
    # Tu código para esta sección...
elif menu_selection == "Noticias de Mercado":
    st.markdown("# Noticias de Mercado")
    # Mostrar DataFrame con los tickers por sector
    st.write("Selecciona un sector y luego un ticker de la tabla para ver las noticias relevantes:")
    sector_seleccionado = st.selectbox("Selecciona un Sector", pd.unique(df_tickers['Sector']))
    
    # Filtrar tickers para el sector seleccionado
    df_sector = df_tickers[df_tickers['Sector'] == sector_seleccionado]
    ticker_display = st.selectbox("Selecciona un Ticker", df_sector['Display'].tolist())
    
    # Extraer el ticker seleccionado del texto display
    ticker_seleccionado = ticker_display.split(' - ')[0]

    # API Key
    api_key = "728e8a5efd29431ea15f0874e93b8a98"

    # Llamada al API para obtener noticias
    def obtener_noticias(ticker):
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error("No se pudieron obtener las noticias.")
            return []

    # Obtiene las noticias del mercado para el ticker seleccionado
    noticias = obtener_noticias(ticker_seleccionado)

    # Itera sobre cada elemento de noticias y extrae los detalles necesarios
    for item in noticias:
        # Extrae y muestra el título de la noticia
        st.write(f"### {item['title']}")
        # Extrae y muestra la fuente y la hora de publicación
        st.write(f"{item['source']['name']} - {item['publishedAt']}")
        # Muestra el link para leer más sobre la noticia
        st.write(f"[Leer más]({item['url']})")

        # Verifica si existe una imagen asociada y la muestra
        if item.get('urlToImage'):
            st.image(item['urlToImage'], caption="Imagen de la noticia", use_column_width=True)
        else:
            st.image("logoad.png", caption="Logo predeterminado", use_column_width=True)
        
        st.markdown("---")

elif menu_selection == "Mercado Financiero":
    st.markdown("# Mercado Financiero")
    ticker = st.text_input("Introduce el ticker de una compañía (ej. AAPL, GOOGL, MSFT):", value='AAPL')
    data_period = st.select_slider("Selecciona el rango de datos:", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
    load_data = st.button("Cargar Datos")

    if load_data:  # Asegurarse de que el botón ha sido presionado
        try:
            data = yf.download(ticker, period=data_period)
            if not data.empty:
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'])])
                fig.update_layout(title=f"Datos históricos de precios para {ticker}", xaxis_title='Fecha', yaxis_title='Precio')
                st.plotly_chart(fig)
            else:
                st.markdown("<span style='color:red;font-weight:bold'>LO LAMENTO BOLIVIANO, INGRESA UN TICKER VÁLIDO</span>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown("<span style='color:red;font-weight:bold'>LO LAMENTO BOLIVIANO, INGRESA UN TICKER VÁLIDO</span>", unsafe_allow_html=True)
elif menu_selection == "Cartera Eficiente":
    st.markdown("# Cartera Eficiente")
    # Sección de análisis de cartera
    st.sidebar.title("Selección de Activos")

    # Mostrar activos por sector
    for sector in pd.unique(df_tickers['Sector']):
        st.sidebar.markdown(f"## {sector}")
        activos_sector = df_tickers[df_tickers['Sector'] == sector]['Display'].tolist()
        activos_seleccionados_sector = st.sidebar.multiselect("Selecciona activos", activos_sector)
        if activos_seleccionados_sector:
            pesos, rendimiento, volatilidad = generar_cartera_optima(activos_seleccionados_sector)
            st.subheader(f"Cartera Óptima Generada para el Sector {sector}")
            st.write(f"Rendimiento Esperado: {rendimiento}")
            st.write(f"Volatilidad: {volatilidad}")
            st.write("Distribución de activos:")
            for i, activo in enumerate(activos_seleccionados_sector):
                st.write(f"{activo}: {pesos[i]*100:.2f}%")
