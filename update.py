import os
import chardet
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
import boto3
import google.generativeai as genai
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

load_dotenv()

# MongoDB setup
client = MongoClient(os.environ['MONGO_URI'])
db = client['cheetah_research']

# AWS S3 setup
s3 = boto3.client('s3',region_name='us-east-1',
                  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

# Generative AI setup
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-pro', generation_config={"response_mime_type": "text/plain"})

def download_files_from_s3(study_id):
    key = f"surveys/{study_id}/"
    objects = s3.list_objects_v2(Bucket=os.environ['BUCKET_NAME'], Prefix=key)
    files = []
    if 'Contents' in objects:
        key_files = [item['Key'] for item in objects['Contents'] if item['Key'] != key]
        if not os.path.exists(f"./storage/{study_id}/"):
            os.makedirs(f"./storage/{study_id}/")
        for file_key in key_files:
            path = f"./storage/{study_id}/{file_key.split('/')[-1]}"
            file_obj = s3.get_object(Bucket=os.environ['BUCKET_NAME'], Key=file_key)
            if file_obj["ContentType"] == "application/pdf":
                s3.download_file(os.environ['BUCKET_NAME'], file_key, path)
                files.append(genai.upload_file(path))
            elif file_obj["ContentType"] == "text/csv":
                csv_body = file_obj["Body"].read()
                result_encoding = chardet.detect(csv_body)
                csv_content = csv_body.decode(result_encoding['encoding'])
                with open(path, 'wb') as f:
                    f.write(csv_content.encode('utf-8'))
                files.append(genai.upload_file(path))
            else:
                pass
    return files


def perform_analysis(study_id, title, objectives, target, filter, files, modules, study_promt):

    prompt_narrative = f"""
        Entrevistamos a personas sobre el siguiente tema: "{str(title)}"
        Objetivos de la encuesta: {str(objectives)}
        Mercado del estudio: {str(target)}
        Proposito de la encuesta: {str(study_promt)}
        Hemos recolectado datos a través de una encuesta cuyo archivo principal de preguntas es el siguiente: log_{study_id}.csv. Usa los demás archivos relacionados para alimentar tu análisis y reforzar las conclusiones.

        Por favor, realiza un análisis narrativo detallado de los datos que están en el archivo log_{study_id}.csv, CONSIDERA EL FILTRO DE: {str(filter)}. Proporciona un resumen bien estructurado, que incluya estadísticas exactas y ejemplos concretos de las respuestas de los encuestados. El análisis debe cubrir al menos los siguientes puntos:

        1. Tendencias generales importantes:
            - Proporciona una descripción detallada de las tendencias generales que encuentres en los datos.
            - Incluye porcentajes y otras estadísticas relevantes calculadas a partir de los datos.

        2. Diferencias significativas entre géneros:
            - Analiza y describe cualquier diferencia significativa entre las respuestas de los géneros.
            - Incluye estadísticas comparativas específicas para cada género.

        3. Ejemplos concretos de respuestas:
            - Incluye ejemplos textuales de respuestas de los encuestados que ilustren las tendencias y diferencias encontradas.
            - Proporciona al menos dos ejemplos contrastantes de respuestas de los estudiantes.

        4. Análisis de preguntas específicas:
            - Selecciona al menos dos preguntas específicas del archivo CSV.
            - Proporciona un análisis detallado de las respuestas a estas preguntas, incluyendo estadísticas relevantes.

        5. Conclusión general:
            - Resume los hallazgos principales del análisis.
            - No introduzcas nueva información en esta sección.

        Recuerda calcular y proporcionar todos los porcentajes y estadísticas exactas de acuerdo con los datos del archivo CSV. Utiliza los archivos adicionales para reforzar tus conclusiones y ejemplos y tambien usa un poco tu informacion de internet para agrandar mas tu analisis.

        Formato de salida:
        Devuelve un Markdown. El analisis análisis debe estar bien detallado y extenderse por un mínimo de tres párrafos grandes.
        

        EJEMPLO:
        "
        ## Análisis General de los Efectos de los Videojuegos en la Población Estudiantil de Tegucigalpa 
        **Tendencias Generales Importantes:**
        Un análisis de las respuestas obtenidas a través de la encuesta revela tendencias interesantes sobre los hábitos de consumo de videojuegos y sus efectos percibidos en la población estudiantil de Tegucigalpa. La mayoría de los encuestados (73%) afirma jugar videojuegos, lo cual indica una alta penetración de esta forma de entretenimiento entre los jóvenes. El tiempo dedicado a los videojuegos varía, siendo el rango más común de 1 a 3 horas diarias (41%). Los géneros de videojuegos preferidos son diversos, destacando los juegos de acción, RPG y casuales.  Un hallazgo importante es que la mayoría de los encuestados (60%) percibe efectos positivos en sus vidas gracias a los videojuegos. Entre los beneficios más mencionados se encuentran la mejora en las habilidades de concentración y resolución de problemas, así como un efecto relajante que ayuda a lidiar con el estrés.
        "

        **Diferencias Significativas Entre Géneros:**
        información sobre las diferencias entre los géneros.
        "
        IMPORTANTE: ANALIZA TODAS LAS PREGUNTAS POSIBLES y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n
        DEVUELVELO EN MARKDOWN.
        
        OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
        OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
        OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
        OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
    """

    prompt_factual = f"""
            Entrevistamos a personas sobre el siguiente tema: "{str(title)}"\n
            Objetivos de la encuesta: {str(objectives)}\n
            Mercado del estudio: {str(target)}\n
            Proposito de la encuesta: {str(study_promt)}
            El archivo principal de preguntas es el siguiente: log_{study_id}.csv, usa los demas archivos para alimentar tu data y las conclusiones de una mejor manera reforzando las respuesta con ella\n 
            Haz un ANALISIS FACTUAL de el SIGUIENTE FILTRO: {str(filter)} detallando bien cada porcentaje, asegurate que la suma de los porcentajes cuadre y de como resultado 100.
            EJEMPLO:
            "
            ## Análisis General de los Efectos de los Videojuegos en la Población Estudiantil de Tegucigalpa 
            **Tendencias Generales Importantes:**
            Un análisis de las respuestas obtenidas a través de la encuesta revela tendencias interesantes sobre los hábitos de consumo de videojuegos y sus efectos percibidos en la población estudiantil de Tegucigalpa. La mayoría de los encuestados (73%) afirma jugar videojuegos, lo cual indica una alta penetración de esta forma de entretenimiento entre los jóvenes. El tiempo dedicado a los videojuegos varía, siendo el rango más común de 1 a 3 horas diarias (41%). Los géneros de videojuegos preferidos son diversos, destacando los juegos de acción, RPG y casuales.  Un hallazgo importante es que la mayoría de los encuestados (60%) percibe efectos positivos en sus vidas gracias a los videojuegos. Entre los beneficios más mencionados se encuentran la mejora en las habilidades de concentración y resolución de problemas, así como un efecto relajante que ayuda a lidiar con el estrés.
            "
            IMPORTANTE: ANALIZA TODAS LAS PREGUNTAS POSIBLES y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n
            DEVUELVELO EN MARKDOWN.
            
            OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
            OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
            OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
            OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
        """

    prompt_individual_questions = f"""
    Entrevistamos a personas sobre el siguiente tema: "{str(title)}"

    Objetivos de la encuesta: {str(objectives)}
    Mercado del estudio: {str(target)}
    Propósito de la encuesta: {str(study_promt)}

    Con base en la información proporcionada en los archivos, hemos realizado entrevistas a profundidad con los encuestados.
    Ahora tenemos que llegar a conclusiones concisas y basadas en hechos que ayuden a las partes interesadas de nuestra empresa a obtener información valiosa.

    *Instrucciones:*
    1. *Extrae y analiza todas las preguntas del archivo log_{study_id}.csv sin omitir ninguna.* 
    2. *Enumera cada pregunta secuencialmente*, comenzando desde 1.
    3. Para cada pregunta, sigue el estilo del ejemplo proporcionado. Cada conclusión debe tener 2-3 pensamientos relevantes de los encuestados y estar estructurada de la siguiente manera:

    *Formato de análisis para cada pregunta:*
    1.  [Inserta la pregunta aquí]
    
    - Propósito de la pregunta: [Explica de manera detallada y larga  la pregunta de negocio relacionada con esta pregunta]

    - Analisis: [Proporciona de manera detallada y larga el análisis narrativo de las respuestas de los encuestados en esta pregunta]
    
    - Conclusiones e insights:
        - [Primer insight sobre las respuestas recolectadas]
        - [Segundo insight sobre las respuestas recolectadas]
        - [Tercer insight sobre las respuestas recolectadas]

    *Recomendaciones para las partes interesadas:*
    - [Recomendación 1]
    - [Recomendación 2]

    *Observación importante:*
    Si, al aplicar los filtros ({str(filter)}), no hay datos suficientes para responder una o más preguntas, debes indicar claramente que "no se encontraron datos suficientes para realizar un análisis de esta pregunta debido a las restricciones de filtrado". OBLIGATORIO INDICAR TEXTUALMENTE QUE NO PUEDES ENCONTRAR DATOS SUFICIENTES A continuación, sugiere posibles ajustes al filtro o menciona cómo la falta de datos podría afectar las conclusiones del estudio.

    Toma como referencia este formato para cada pregunta. Asegúrate de:
    - *Numerar y analizar todas las preguntas del archivo sin dejar ninguna.*
    - Seguir la estructura de conclusiones con ejemplos claros.
    - Evitar el conteo exacto de encuestados y no citar los IDs de las encuestas ni los nombres.
    - Usar la primera letra en mayúscula para nombres propios.
    - Devolver la información en formato Markdown.

    El archivo principal de preguntas es: log_{study_id}.csv. Usa los otros archivos para complementar las respuestas.

    Filtro aplicado: {str(filter)}
    
    
    SIGUE EL SIGUIENTE FORMATO MARKDOWN:
    # Título del Análisis - Título del Filtro
    Enfoque

    ---

    #### 1. Pregunta
    Resumen

    ---

    #### 2. Pregunta
    Resumen
    ---

    #### 3. Pregunta
    Resumen
    ---

    ## Análisis Final
    Contenido del análisis final
    """

    prompt_percentage_questions = f"""
    Entrevistamos a personas sobre el siguiente tema: "{str(title)}"

    Objetivos de la encuesta: {str(objectives)}
    Mercado del estudio: {str(target)}
    Propósito de la encuesta: {str(study_promt)}

    **Instrucciones:**
    1. **Haz un análisis porcentual por cada pregunta**, asegurándote de intentar el análisis aun cuando la cantidad de datos sea baja. Siempre que sea posible, intenta estimar los porcentajes.
    2. **Enumera todas las preguntas del archivo** sin omitir ninguna, incluso si no tienen suficientes datos para un análisis completo.
    3. Si realmente no se puede realizar el análisis porcentual de una pregunta debido al filtro aplicado, **indica claramente: "No se encontraron datos suficientes para realizar un análisis de esta pregunta debido a las restricciones de filtrado."**

    **Ejemplo del formato de análisis porcentual:**
    "
    1. ¿Con qué frecuencia visitas el restaurante de alitas?
        - Diariamente: 5%
        - Semanalmente: 25%
        - Mensualmente: 45%
        - Raramente: 25%

    2. ¿Con qué frecuencia haces ejercicio?
        No se encontraron datos suficientes para realizar un análisis de esta pregunta debido a las restricciones de filtrado.
    "

    **Formato que debes seguir para el análisis:**
    # Título del Análisis - Título del Filtro
    Enfoque

    ---

    #### *1. Pregunta*
    - Opción uno: porcentaje%

    ---

    #### *2. Pregunta*
    - Opción uno: porcentaje%

    ---

    #### *3. Pregunta*
    - Opción uno: porcentaje%

    ---

    ## Análisis Final
    Contenido del análisis final

    **Obligatorio:**
    - **Numerar y analizar todas las preguntas del archivo log_{study_id}.csv sin dejar ninguna**, aunque no haya suficientes datos.
    - **No mostrar conteo de encuestados**; solo mostrar el porcentaje en formato (porcentaje%).
    - La primera letra de los nombres propios debe ir en mayúscula.
    - Usa títulos, subtítulos y separaciones claras entre cada análisis.
    - El análisis debe devolverse en formato Markdown.

    El archivo principal de preguntas es: log_{study_id}.csv. Usa los demás archivos para complementar tu análisis.
    """


    prompt_nps_questions = f"""
        Entrevistamos a personas sobre el siguiente tema: "{str(title)}"\n
        Objetivos de la encuesta: {str(objectives)}\n
        Mercado del estudio: {str(target)}\n
        Proposito de la encuesta: {str(study_promt)}
        Haz un ANALISIS PORCENTUAL y NPS por pregunta sin detallar o explicar cosas, SACA LA INFORMACION DEL SIGUIENTE FILTRO: {str(filter)}\n
        En el mismo formato agregale la Clasificacion NPS a cada porcentaje
            Clasificación NPS:
                Promotores (P): Calificaciones de 9-10. Clientes extremadamente satisfechos y leales, que recomendarían activamente el restaurante.
                Indiferentes (I): Calificaciones de 7-8. Clientes satisfechos pero no entusiastas, que podrían cambiar a la competencia.
                Detractores (D): Calificaciones de 0-6. Clientes insatisfechos que podrían desaconsejar a otros de visitar el restaurante.
            
            Ejemplo:
                "
                1. ¿Con qué frecuencia visitas el restaurante de alitas?
                    Diariamente: 5% (P)
                    Semanalmente: 25% (P)
                    Mensualmente: 45% (I)
                    Raramente: 25% (D)
                "
        IMPORTANTE: ANALIZA TODAS LA INFORMACION y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n
        DEVUELVELO EN MARKDOWN.

        OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
        OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
        OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
        OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
        """

    prompt_ekman_questions = f"""
            Entrevistamos a personas sobre el siguiente tema: "{str(title)}"\n
            Objetivos de la encuesta: {str(objectives)}\n
            Mercado del estudio: {str(target)}\n
            SACA LA INFO DEL SIGUIENTE FILTRO: {str(filter)}\n
            Proposito de la encuesta: {str(study_promt)}
            Haz un Analisis de emosiones EKMAN dde cada pregunta, combina siguiendo el siguiente ejemplo:
            
            "¿Con qué frecuencia visitas el restaurante de alitas?
                Alegría (40%): Clientes que visitan semanalmente (25%) y diariamente (5%) experimentan alegría y satisfacción.
                Sorpresa (10%): Visitas raras (25%) que ocurren en ocasiones especiales pueden estar asociadas con sorpresa.
                Enojo/Tristeza (5%): Visitas raras (25%) debido a malas experiencias previas podrían generar enojo o tristeza.
                Miedo (0%): No se observaron respuestas asociadas con miedo.
                Disgusto (0%): No se observaron respuestas asociadas con disgusto.
            "
            
            IMPORTANTE: ANALIZA TODAS LAS PREGUNTAS POSIBLES y DEVUELVE EN MARKDOWN, SACA LA INFO DEL SIGUIENTE FILTRO: {str(filter)}\n
            
            El archivo principal de preguntas es el siguiente: log_{study_id}.csv, usa los demas archivos para alimentar tu data y las conclusiones de una mejor manera reforzando las respuesta con ella\n 
            
            DEVUELVELO EN MARKDOWN.
            
            OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
            OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
            OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
            OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
        """    

    prompt_personality_questions = f"""
            Entrevistamos a personas sobre el siguiente tema: "{str(title)}"\n
            Objetivos de la encuesta: {str(objectives)}\n
            Mercado del estudio: {str(target)}\n
            Proposito de la encuesta: {str(study_promt)}
            Haz un analisis de Rasgos de personalidad de cada pregunta, combina siguiendo el siguiente ejemplo:
            "¿Con qué frecuencia visitas el restaurante de alitas?
                Emocionales (30%):
                Semanalmente (25%) y Diariamente (5%): Estos clientes probablemente visitan el restaurante frecuentemente debido a una fuerte conexión emocional con la experiencia, disfrutando de la atmósfera, el sabor y las emociones positivas que asocian con el lugar.
                Racionales (70%):
                Mensualmente (45%) y Raramente (25%): Estos clientes planifican sus visitas con menos frecuencia, probablemente basando su decisión en factores racionales como presupuesto, conveniencia o eventos especiales."
            IMPORTANTE: ANALIZA TODAS LAS PREGUNTAS POSIBLES  y elimina los caracteres de escape raros o especiales como \\xa0 que pueden molestar el json
            
            El archivo principal de preguntas es el siguiente: log_{study_id}.csv, usa los demas archivos para alimentar tu data y las conclusiones de una mejor manera reforzando las respuesta con ella\n 
            
            IMPORTANTE: ANALIZA TODAS LAS PREGUNTAS POSIBLES y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n
            DEVUELVELO EN MARKDOWN.

            OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
            OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
            OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
            OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
        """
        
    prompt_segmentos_questions = f"""
            Entrevistamos a personas sobre el siguiente tema: "{str(title)}"\n
            Objetivos de la encuesta: {str(objectives)}\n
            Mercado del estudio: {str(target)}\n
            Proposito de la encuesta: {str(study_promt)}
            Con base en la información proporcionada en los archivos, hemos realizado entrevistas a profundidad con los encuestados.
            Ahora tenemos que llegar a conclusiones concisas y basadas en hechos que ayuden a las partes interesadas de nuestra empresa a obtener información valiosa.
            Haz un analisis de los segmentos psicograficos de los encuestados, combina siguiendo el siguiente ejemplo:
            "
            ¿Cuál es tu sabor de alitas favorito?
                Embajadores (30%): Preferencia por picante (30%) sugiere clientes que son apasionados y pueden recomendar el restaurante a otros.
                Leales (25%): Preferencia por agridulce (25%) indica una satisfacción constante y una fuerte preferencia por ciertos sabores.
                Indistintos (20%): Preferencia por barbacoa (20%) sugiere una satisfacción general sin una fuerte inclinación.
                Críticos (15%): Preferencia por búfalo (15%) puede indicar que disfrutan de la oferta pero no están totalmente satisfechos.
                En Riesgo (10%): Preferencia por limón y pimienta (10%) puede representar a aquellos que buscan opciones específicas y podrían cambiar si no las encuentran.
            "
            
            IMPORTANTE: ANALIZA TODAS LAS PREGUNTAS POSIBLES y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n
            
            El archivo principal de preguntas es el siguiente: log_{study_id}.csv, usa los demas archivos para alimentar tu data y las conclusiones de una mejor manera reforzando las respuesta con ella\n 
            
            DEVUELVE EL ANALISIS EN MARKDOWN.
            
            OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
            OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
            OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
            OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
        """        
            
    promptestilo_comunicacion_questions = f"""
            Entrevistamos a personas sobre el siguiente tema: "{str(title)}"\n
            Objetivos de la encuesta: {str(objectives)}\n
            Mercado del estudio: {str(target)}\n
            Proposito de la encuesta: {str(study_promt)}
            Con base en la información proporcionada en los archivos, hemos realizado entrevistas a profundidad con los encuestados.
            Ahora tenemos que llegar a conclusiones concisas y basadas en hechos que ayuden a las partes interesadas de nuestra empresa a obtener información valiosa.
            Haz un analisis de los Estilos de Comunicacion de los encuestados, combina siguiendo el siguiente ejemplo:
            
            Segmentación por Estilo de Comunicación:
            Para categorizar a los encuestados según su estilo de comunicación (Reveladores, Factuales, Informativos, Buscador de Acción), se utilizarán respuestas inventadas y segmentaciones basadas en como los encuestados expresan sus opiniones y preferencias.
            
            IMPORTANTE: ANALIZA TODAS LAS PREGUNTAS POSIBLES y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n

            EJEMPLO:
            "
            Análisis de los Resultados de la Encuesta:
                ¿Con qué frecuencia visitas el restaurante de alitas?
                    Reveladores: 15%
                    Factuales: 35%
                    Informativos: 30%
                    Buscador de Acción: 20%
            "
                    
            IMPORTANTE: ANALIZA TODAS LA INFORMACION y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n
            DEVUELVELO EN MARKDOWN.
            
            OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
            OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
            OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
            OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
        """
    
    prompt_user_personas = f"""
            Entrevistamos a personas sobre el siguiente tema: "{str(title)}"
            Objetivos de la encuesta: {str(objectives)}
            Mercado del estudio: {str(target)}
            Proposito de la encuesta: {str(study_promt)}
            Hemos recolectado datos a través de una encuesta cuyo archivo principal de preguntas es el siguiente: log_{study_id}.csv. Usa los demás archivos relacionados para alimentar tu análisis y reforzar las conclusiones.
            HAZ UN USUARIO Y DESCRIBE UN COMUN PARA ESTE ESTUDIO DE ACUERDO AL SIGUIENTE FILTRO: {filter}
            Formato de salida:
            Devuelvelo en MARKDOWN. Cada análisis debe estar bien detallado y extenderse por un mínimo de tres párrafos grandes.
            
            USA EL SIGUIENTE EJEMPLO PARA QUE TU ANALISIS SEA CORRECTO:
            "
                Nombre: Juan Pérez

                Demografía:
                Edad: 28 años
                Género: Masculino
                Nivel de Ingresos: $30,000 - $50,000 anuales
                Ocupación: Profesional de TI
                Estado Civil: Soltero
                
                Psicografía:
                Intereses y Aficiones: Deportes, videojuegos, salir con amigos
                Valores y Creencias: Valora la autenticidad y la calidad en los alimentos
                Estilo de Vida: Activo, social, disfruta de la vida nocturna
                Personalidad: Extrovertido, aventurero, le gusta probar cosas nuevas
                
                Comportamiento:
                Patrones de Compra: Visita el restaurante al menos una vez al mes, más frecuente durante eventos deportivos
                Lealtad a la Marca: Fiel al restaurante por sus sabores únicos y buen servicio
                Motivaciones de Compra: Busca una experiencia divertida y sabores intensos
                Canales de Compra Preferidos: Prefiere comer en el restaurante para disfrutar del ambiente
                
                Necesidades y Puntos de Dolor:
                Necesidades: Variedad de sabores, ambiente animado, opciones para ver deportes en vivo
                Puntos de Dolor: A veces, la espera es demasiado larga durante horas pico
                
                Preferencias y Hábitos de Consumo:
                Preferencias de Producto: Sabores picantes y agridulces
                Preferencias de Servicio: Valora un servicio rápido y amable
                Hábitos de Consumo: Visita más frecuentemente los fines de semana y durante eventos deportivos
                
                Competencia:
                Percepción de la Competencia: Considera que otros restaurantes no tienen tanta variedad de sabores            
            "
            
            USA SUBTITULOS Y TITULOS EN TU ANALISIS USANDO EL FORMATO MARKDOWN
            IMPORTANTE: ANALIZA TODAS LA INFORMACION y DEVUELVE EN MARKDOWN, SACA LA TODA INFORMCION CON RESPETO AL SIGUIENTE FILTRO: {str(filter)}\n
            DEVUELVELO EN MARKDOWN.
            
            OBLIGATORIO : NO MUESTRES EL CONTEO DE ENCUESTADOS EN NINGUN MOMENTO, SOLO ANALIZA Y PORCENTUALIZA
            OBLIGATORIO : LA PRIMERA LETRA DE LOS NOMBRES PROPIOS TIENE QUE IR EN MAYUSCULA
            OBLIGATORIO : NO CITAR LOS ID'S DE LAS ENCUESTAS NI LOS NOMBRES DE LOS ENCUESTADOS, SOLO INSERTAR LA CITA
            OBLIGATORIO : NO CITAR EL DOCUMENTO EN EL QUE SE BASA EL ANALISIS
        """
    prompts = {
        "narrative": prompt_narrative,
        "factual": prompt_factual,
        "individual_narrative": prompt_individual_questions,
        "percentage": prompt_percentage_questions,
        "user_personas": prompt_user_personas,
        "segmentos": prompt_segmentos_questions,
        "ekman": prompt_ekman_questions,
        "nps": prompt_nps_questions,
        "personality" : prompt_personality_questions,
        "estilo" : promptestilo_comunicacion_questions
    }
        
    def process_prompt(prompt, analysis):
        flag = True
        while flag:
            try:
                response = model.generate_content(files + [prompt])
                md_response = response.text
                path = f"analysis/{study_id}/"
                if(analysis == "factual" or analysis == "narrative"):
                    path += f"general/{analysis}/{filter}.md"
                elif(analysis == "individual_narrative" or analysis == "percentage"):
                    path += f"individual_questions/{analysis}/{filter}.md"
                elif(analysis == "user_personas"):
                    path += f"user_personas/{filter}.md"
                elif(analysis == "segmentos" or analysis == "ekman" or analysis == "nps" or analysis == "personality" or analysis == "estilo"):
                    path += f"psicographic_questions/{analysis}/{filter}.md"
                s3.put_object(Bucket=os.environ['BUCKET_NAME'], Key=path, Body=md_response)
                flag = False
            except Exception as e:
                flag = True
                print(f"Error processing prompt: {e}")

    with ThreadPoolExecutor() as executor:
        future_to_prompt = {executor.submit(process_prompt, prompt, key): key for key, prompt in prompts.items()}
        for future in as_completed(future_to_prompt):
            key = future_to_prompt[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error retrieving result for {key}: {e}")

    for file in files:
        file.delete()

def process_log(log):
    study_id = str(log['_id'])
    last_updated = log['last_update']
    # Get filters from Surveys collection
    if (datetime.now() - last_updated).seconds < 60:
        return
    filters = []
    modules = []
    try:
        survey = db['Surveys'].find_one({'_id': ObjectId(study_id)})
        study_promt = survey['prompt']
        filters = survey['filters']
    except Exception as e:
        pass
    filters.insert(0, 'General')

    # Get study details from Study collection
    study = db['Study'].find_one({'_id': ObjectId(study_id)})
    title = study['title']
    objectives = study['studyObjectives']
    target = study['marketTarget']
    files = download_files_from_s3(study_id)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(perform_analysis, study_id, title, objectives, target, filter, files, modules, study_promt) for filter in filters]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing analysis: {e}")
    with open("logs.txt", "a") as f:
        f.write(f"Study {study_id} processed at {datetime.now()}\n")
    shutil.rmtree(f"./storage/{study_id}/")
    db['survey_logs'].delete_one({'_id': ObjectId(study_id), 'last_update': last_updated})

def main():
    while True:
        collection = db['survey_logs'].find()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_log, log) for log in collection]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred during parallel processing: {e}")
                    
if __name__ == "__main__": 
    main()