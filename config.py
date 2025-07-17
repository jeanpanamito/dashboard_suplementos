# config.py
import os
import logging
from pymongo import MongoClient
import pymongo.errors

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    # Configuración de MongoDB
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'fitness_supplements')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'products')
    
    # Configuración de timeouts y conexión
    SERVER_SELECTION_TIMEOUT_MS = int(os.getenv('SERVER_SELECTION_TIMEOUT_MS', 5000))
    CONNECT_TIMEOUT_MS = int(os.getenv('CONNECT_TIMEOUT_MS', 5000))
    SOCKET_TIMEOUT_MS = int(os.getenv('SOCKET_TIMEOUT_MS', 30000))
    MAX_POOL_SIZE = int(os.getenv('MAX_POOL_SIZE', 10))
    
    # Configuración de cache
    CACHE_TTL = int(os.getenv('CACHE_TTL', 300))  # 5 minutos
    
    # Configuración de la app (corregido el typo "appm")
    PAGE_TITLE = "Gainsight"
    PAGE_ICON = ""
    
    # Variables de entorno adicionales
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
    @classmethod
    def get_mongo_client(cls):
        """
        Crea y retorna un cliente MongoDB con configuración robusta
        """
        try:
            client = MongoClient(
                cls.MONGO_URI,
                serverSelectionTimeoutMS=cls.SERVER_SELECTION_TIMEOUT_MS,
                connectTimeoutMS=cls.CONNECT_TIMEOUT_MS,
                socketTimeoutMS=cls.SOCKET_TIMEOUT_MS,
                maxPoolSize=cls.MAX_POOL_SIZE,
                retryWrites=True,
                retryReads=True
            )
            
            # Probar la conexión
            client.admin.command('ping')
            logger.info(f"✅ Conexión exitosa a MongoDB: {cls.MONGO_URI}")
            
            return client
            
        except pymongo.errors.ServerSelectionTimeoutError as e:
            logger.error(f"❌ Timeout conectando a MongoDB: {e}")
            logger.error("Verifica que MongoDB esté ejecutándose y sea accesible")
            return None
            
        except pymongo.errors.ConnectionFailure as e:
            logger.error(f"❌ Fallo en la conexión a MongoDB: {e}")
            return None
            
        except pymongo.errors.ConfigurationError as e:
            logger.error(f"❌ Error de configuración MongoDB: {e}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error inesperado conectando a MongoDB: {e}")
            return None
    
    @classmethod
    def get_mongo_collection(cls):
        """
        Retorna la colección MongoDB configurada
        """
        try:
            client = cls.get_mongo_client()
            if client is None:
                return None
            
            db = client[cls.DATABASE_NAME]
            collection = db[cls.COLLECTION_NAME]
            
            # Verificar que la colección existe y tiene datos
            doc_count = collection.count_documents({})
            logger.info(f"📊 Colección '{cls.COLLECTION_NAME}' - Documentos: {doc_count}")
            
            return collection
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo colección: {e}")
            return None
    
    @classmethod
    def get_database(cls):
        """
        Retorna la base de datos MongoDB
        """
        try:
            client = cls.get_mongo_client()
            if client is None:
                return None
            
            return client[cls.DATABASE_NAME]
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo base de datos: {e}")
            return None
    
    @classmethod
    def test_connection(cls):
        """
        Prueba la conexión y retorna información del estado
        """
        result = {
            'connected': False,
            'database_exists': False,
            'collection_exists': False,
            'document_count': 0,
            'error': None
        }
        
        try:
            # Probar cliente
            client = cls.get_mongo_client()
            if client is None:
                result['error'] = "No se pudo conectar al cliente MongoDB"
                return result
            
            result['connected'] = True
            
            # Verificar base de datos
            db_names = client.list_database_names()
            result['database_exists'] = cls.DATABASE_NAME in db_names
            
            if result['database_exists']:
                # Verificar colección
                db = client[cls.DATABASE_NAME]
                collection_names = db.list_collection_names()
                result['collection_exists'] = cls.COLLECTION_NAME in collection_names
                
                if result['collection_exists']:
                    # Contar documentos
                    collection = db[cls.COLLECTION_NAME]
                    result['document_count'] = collection.count_documents({})
            
            client.close()
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error en test_connection: {e}")
        
        return result
    
    @classmethod
    def get_connection_info(cls):
        """
        Retorna información de la configuración de conexión
        """
        return {
            'MONGO_URI': cls.MONGO_URI,
            'DATABASE_NAME': cls.DATABASE_NAME,
            'COLLECTION_NAME': cls.COLLECTION_NAME,
            'SERVER_SELECTION_TIMEOUT_MS': cls.SERVER_SELECTION_TIMEOUT_MS,
            'CONNECT_TIMEOUT_MS': cls.CONNECT_TIMEOUT_MS,
            'SOCKET_TIMEOUT_MS': cls.SOCKET_TIMEOUT_MS,
            'MAX_POOL_SIZE': cls.MAX_POOL_SIZE,
            'CACHE_TTL': cls.CACHE_TTL,
            'DEBUG': cls.DEBUG,
            'ENVIRONMENT': cls.ENVIRONMENT
        }
    
    @classmethod
    def create_indexes(cls):
        """
        Crea índices recomendados para mejor performance
        """
        try:
            collection = cls.get_mongo_collection()
            if collection is None:
                return False
            
            # Índices recomendados para el dashboard
            indexes = [
                ('categoria', 1),
                ('precio_num', 1),
                ('rating_num', 1),
                ('fuente', 1),
                ('timestamp', -1),
                ([('categoria', 1), ('precio_num', 1)], 'categoria_precio'),
                ([('fuente', 1), ('rating_num', -1)], 'fuente_rating')
            ]
            
            for index in indexes:
                if isinstance(index, tuple) and len(index) == 2:
                    if isinstance(index[0], list):
                        # Índice compuesto
                        collection.create_index(index[0], name=index[1])
                        logger.info(f"✅ Índice creado: {index[1]}")
                    else:
                        # Índice simple
                        collection.create_index(index)
                        logger.info(f"✅ Índice creado: {index[0]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creando índices: {e}")
            return False

# Función de utilidad para uso directo
def get_collection():
    """Función de conveniencia para obtener la colección"""
    return Config.get_mongo_collection()

def test_mongo_connection():
    """Función de conveniencia para probar la conexión"""
    return Config.test_connection()

# Validación de configuración al importar
if __name__ == "__main__":
    print("🔧 Probando configuración MongoDB...")
    print("="*50)
    
    # Mostrar configuración
    info = Config.get_connection_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n🔍 Probando conexión...")
    result = Config.test_connection()
    
    for key, value in result.items():
        icon = "✅" if (key != 'error' and value) or (key == 'error' and not value) else "❌"
        print(f"{icon} {key}: {value}")
    
    if result['connected'] and result['collection_exists'] and result['document_count'] > 0:
        print("\n🎉 ¡Configuración exitosa!")
    else:
        print("\n⚠️ Hay problemas con la configuración. Revisa los errores arriba.")