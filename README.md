# Gainsight

Gainsight es un dashboard integral de inteligencia de mercado para la industria de suplementos. Proporciona análisis en profundidad de tendencias de mercado, precios, marcas líderes e información competitiva para tomar decisiones informadas.

## Características

- Análisis y visualización de tendencias de mercado
- Monitoreo y comparación de precios
- Seguimiento de rendimiento de marcas
- Procesamiento de datos en tiempo real con integración Kafka
- Dashboards interactivos con filtros y análisis detallado
- Almacenamiento de datos en MongoDB para análisis escalable

## Stack Tecnológico

- **Frontend**: Streamlit
- **Procesamiento de Datos**: Pandas, NumPy
- **Visualización**: Plotly
- **Base de Datos**: MongoDB
- **Streaming**: Apache Kafka
- **Contenedorización**: Docker

## Instalación

1. Clonar el repositorio
2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Configurar la conexión MongoDB en `config.py`
4. Ejecutar la aplicación:
   ```
   streamlit run main_app.py
   ```

## Despliegue con Docker

Usar la configuración Docker proporcionada:
```
docker-compose up -d
```

## Estructura del Proyecto

- `main_app.py` - Punto de entrada principal de la aplicación
- `dashboard.py` - Funcionalidad principal del dashboard
- `config.py` - Configuraciones del sistema
- `utils.py` - Funciones utilitarias
- `Kafka_*.ipynb` - Notebooks de productor/consumidor Kafka para streaming de datos
- `docker-compose.yml` - Orquestación de contenedores

## Uso

Acceder al dashboard en `http://localhost:8501` después de iniciar la aplicación. Navegar por las diferentes secciones para explorar datos de mercado, tendencias y análisis específicos de la industria de suplementos.
