# CREACION DE MODELO
# Importacion de librerias


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/content/heart.csv")

# Escalamiento
from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Modelos
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metricas
from sklearn.metrics import accuracy_score, classification_report, roc_curve

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print('Packages imported...')

# Creación de una copia del df
df1 = df

# Seleccion de las columnas a ser escaladas y codificadas
cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

# Codificando las columnas categoricas
df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

# Definiendo los atributos independientes y el atributo dependiente
X = df1.drop(['output'],axis=1)
y = df1[['output']]

# Instanciando el escalador
scaler = RobustScaler()

# Escalando los atributos continuos
X[con_cols] = scaler.fit_transform(X[con_cols])
print("Las primeras 5 filas de X")
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("El tamaño de X_train es      ", X_train.shape)
print("El tamaño de X_test es        ",X_test.shape)
print("El tamaño de y_train es       ",y_train.shape)
print("El tamaño de y_test es        ",y_test.shape)

# Instanciamiento del objeto
logreg = LogisticRegression()

# Ajustando el objeto
logreg.fit(X_train, y_train)

# Calculo de probabilidades
y_pred_proba = logreg.predict_proba(X_test)

# Encontrando los valores predichos
y_pred = np.argmax(y_pred_proba,axis=1)

# Impresión de la prueba de precisión
print("El puntaje de precisión en la prueba de Regresión Logística es ", accuracy_score(y_test, y_pred))


## GRADIO DE AQUÍ PARA ABAJO

# Instalamos nuestra librería para la respectiva interfaz gráfica para el usuario
#!pip install -q gradio

# Instalamos la libreria de hugging face spaces para el hosting en la nube
#!pip install -q transformers

#Importamos gradio y el metodo necesario para el hosting
import gradio as gr

# Variables globales
# Creamos una lista donde se almacenarán los nombres de todos los campo de la UI
nombres_campos = ['Edad', 'Sexo', 'Tipo de dolor toracico',
                  'Presión arterial en reposo', 'Colesterol',
                  'Azúcar en sangre en ayunas',
                  'Resultados electrocardiográficos en reposo',
                  'Ritmo cardíaco máximo alcanzado',
                  'Angina inducida por ejercicio',
                  'Depresión ST(Old Peak)', 'Pendiente del segmento ST', 
                  'Número de vasos principales',
                  'Resultado de la prueba de esfuerzo con talio']
texto_atributos = """-Diccionario de datos-\n
`Edad` - Edad del paciente en cuestión.\n
`Sexo` - Sexo del paciente.\n
`Tipo de dolor torácico` - Si se presenta alguno de los siguientes: angina típica, angina atípica, dolor no anginal, 3 = asintomático.\n
`Presión arterial en reposo` - Presión arterial en reposo (medido en mm Hg).\n
`Colesterol` - Es obtenido a través del sensor de IMC (Medido en  mg/dl).\n
`Azúcar en sangre en ayunas` - Cuando este valor es mayor a 120 mg/dl.\n
`Resultados electrocardiográficos en reposo` - Normal, Normalidad de la onda ST-T, Hipertrofia ventricular izquierda.\n
`Angina inducida por ejercicio` - Si se padece este tipo de angina.\n
`Ritmo cardíaco máximo alcanzado`  - Ritmo cardíaco máximo alcanzado en la prueba con talio.\n
`Depresión ST(Old Peak)` - Pico anterior: es inducida por el ejercicio en relación al reposo.\n
`Pendiente del segmento ST` - Inclinación del segmento ST (ascendente, descendente u horizontal ).\n
`Número de vasos principales` - Número de vasos principales obtenido en Fluoroscopia.\n
`Resultado de la prueba de esfuerzo con talio` - Resultado de la prueba de esfuerzo con talio.\n"""

# creación de tema personalizado
custom_theme = gr.themes.Soft(
  primary_hue="emerald",
  secondary_hue="teal",
  neutral_hue="stone",
  text_size="lg",
  spacing_size="sm",
  radius_size="lg"
  ).set(
    body_background_fill='*stat_background_fill',
    body_background_fill_dark='*neutral_800',
    body_text_color_subdued='*primary_100',
    body_text_weight='500',
    background_fill_primary_dark='*neutral_700',
    background_fill_secondary_dark='*background_fill_primary',
    border_color_accent='*primary_600',
    border_color_accent_dark='*neutral_950',
    color_accent='*neutral_800',
    color_accent_soft_dark='*body_text_color_subdued',
    link_text_color='*neutral_800',
    link_text_color_dark='*border_color_primary',
    prose_text_weight='500',
    prose_header_text_weight='400',
    block_background_fill='*neutral_700',
    block_border_color_dark='*neutral_950',
    block_border_width_dark='1 px',
    block_info_text_size='*text_md',
    block_label_background_fill_dark='*primary_800',
    checkbox_background_color='*neutral_200',
    checkbox_background_color_dark='*neutral_950',
    checkbox_background_color_focus_dark='*neutral_800',
    checkbox_border_color='*neutral_300',
    checkbox_border_color_dark='*neutral_800'
)

# Crear archivos temporales para guardar el grafico en una imagen
import tempfile

# cambiar la media de todo el dataset real por todos los registros reales donde el target fue 1
def crear_grafico(datos_paciente):
  nombres_campos_formateados = ['Edad', 'Sexo', 'Tipo de dolor\ntoracico',
                  'Presión arterial\nen reposo', 'Colesterol',
                  'Azúcar en sangre\nen ayunas',
                  'Resultados\nelectrocardiográficos\nen reposo',
                  'Ritmo cardíaco\nmáximo alcanzado',
                  'Angina inducida\nor ejercicio',
                  'Depresión ST\n(Old Peak)', 'Pendiente del\nsegmento ST', 
                  'Número de vasos\nprincipales',
                  'Resultado de la prueba\nde esfuerzo con talio']

  #features_mean_df= pd.DataFrame([df.mean(axis = 0)])
  features_mean_df = df.loc[df['output'] == 1, ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']]
  features_mean_df = features_mean_df.drop(['output'], axis = 1)
  
  # Valores para X y Y para la media
  y_plot_means = features_mean_df.iloc[0].values        # Medias

  # Valores para X y Y para los datos ingresados del paciente
  y_plot_paciente = pd.DataFrame(datos_paciente).iloc[0].values

  # Crear el gráfico de barras para la media del dataset
  fig, ax = plt.subplots(figsize=(16,9))    
  width = 0.35 # the width of the bars 
  ind = np.arange(len(y_plot_means))  # the x locations for the groups
  # Gráfico de barras para la media
  ax.barh(ind, y_plot_means, width, color="darkseagreen", label='Media')

  # Gráfico de barras para los datos del paciente
  ax.barh(ind + width, y_plot_paciente.astype(float), width, color="skyblue")

  ax.set_yticks(ind+width/2)
  ax.set_yticklabels(labels = nombres_campos_formateados, fontweight='bold')
  max_paciente = y_plot_paciente.astype(np.float64)
  x_labels = np.max(max_paciente)
  ax.set_xticklabels(labels = np.arange(0, x_labels+51, 50), fontweight='bold')

  plt.title('Comparación de los atributos del pacientes con la media de infartos reales.', fontweight='bold')
  plt.xlabel('Valores')
  plt.ylabel('Atributo')      

  plt.legend(labels=['Media de Infartos', 'Paciente'])
  fig.set_facecolor("mediumaquamarine")
  ax.set_facecolor("mintcream")

  # Agregar etiquetas con los valores del paciente en el gráfico
  for i, v in enumerate(y_plot_means):
    ax.annotate("{:.2f}".format(v), xy=(v + 4, i + .25), va='center', color='forestgreen', fontweight='bold')
      
  for i, v in enumerate(y_plot_paciente.astype(float)):
    ax.annotate("{:.2f}".format(v), xy=(v + 4, i + width + .33), va='center', color='steelblue', fontweight='bold')

  # Guardar la figura en un archivo temporal
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
    fig.savefig(temp_file, format='png')
    temp_file.close()
    temp_filename = temp_file.name

  return temp_filename

# Creamos la función con la que generaremos predicciones mediante el llenado de los
# valores de las variables y validaremos los valores que contienen los campos
def diagnosticar(age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall):
  # ---código para validación de campos---

  # creamos una lista donde se almacenan los valores de los campos
  inputs = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]

  if all(var is not None and (var!= "" )and (var!= '') for var in inputs):
    # verdadero si todas las variables tienen valores distintos a nulo
    not_nulls = True;
  else:
    # falso si alguna de las variables tiene un valor nulo
    not_nulls = False
    # creamos una lista la cual contendra una tupla, donde se tendrá el nombre del campo y su valor nulo 
    campos_con_nulos = [nombres_campos[i] for i in range(len(inputs)) if (inputs[i] == "") or (inputs[i] == '') or (inputs[i] is None)]
    # Dependiendo de la cantidad de campos vacíos mostraremos su error correspondiente
    if len(campos_con_nulos) == 1:
        raise gr.Error(message = f"Falta el campo: {campos_con_nulos} por llenar!")
    else:
      # creamos una cadena a partir de la lista campos_con_nulos
      # juntaremos todos los registros en una sola cadena y separaremos el nombre
      # de cada campo con una coma, seguido de un espacio y antes de que se
      # muestre el nombre del campo se escribe una viñeta, esto para facilitarle
      # al usuario la identificación de los campos vacíos
      campos_faltantes = ',  •'.join([campo for campo in campos_con_nulos])
      raise gr.Error(message = f"¡Faltan por llenar {len(campos_con_nulos)} campos!: •{campos_faltantes}")
    
    #---código para generar las predicciones---
  if(not_nulls):
    paciente_info = {
      'age'       :   [age],    
      'sex'       :   [sex],    
      'cp'        :   [cp],    
      'trtbps'    :   [trtbps],  
      'chol'      :   [chol],  
      'fbs'       :   [fbs],    
      'restecg'   :   [restecg],    
      'thalachh'  :   [thalachh],  
      'exng'      :   [exng],  
      'oldpeak'   :   [oldpeak],  
      'slp'       :   [slp],    
      'caa'       :   [caa],    
      'thall'     :   [thall]    
    }
    # creacion de DF para almacenar la información del paciente o usuario que
    # quiera generar una predicción
    paciente = pd.DataFrame(paciente_info)

    #crear graficas para comparar el paciente con la media o el resto del dataset
    graph = crear_grafico(paciente)

    # Codificando las columnas categoricas
    paciente_dummy = pd.get_dummies(paciente, columns = cat_cols, drop_first = True)

    # Definiendo los atributos independientes y el atributo dependiente
    all_cols = set(X_train.columns)

    missing_cols = all_cols - set(paciente_dummy.columns)
    for col in missing_cols:
      paciente_dummy[col] = 0

    paciente_dummy = paciente_dummy[X_train.columns]

    paciente[con_cols] = scaler.transform(paciente[con_cols])

    # Haciendo predicciones en nuevos datos
    prediccion = logreg.predict(paciente_dummy)


    if prediccion == 0:
      return ("No se presenta riesgo de un infarto", graph)
    else:
      return ("Existe riesgo de infarto\nPor favor visite a un medico", graph)
  else:
    raise gr.Error("¡ERROR CRÍTICO.- Consulte a los desarrolladores.")

# Creación de block
with gr.Blocks(title="Predicción de riesgo de un IAM", theme = custom_theme) as modelo:
  gr.Markdown(
        """
    # Predicción de riesgo de Infarto Agudo al Miocardio
    Modelo de Regresión Lineal para identificar la existencia de riesgo de un IAM.
    Al final se presentan ejemplos, para ejecutarlos seleccionar uno y dar click en el botón "Generar".
    """
    )
  with gr.Row():
    with gr.Column():
      age   = gr.Textbox(label="Edad",placeholder="Ingrese su edad en años.")
  with gr.Row():
      sex   = gr.Dropdown(label="Sexo",
            choices=["0","1"],
            info="Mujer (0)\nHombre (1)",
            placeholder="Seleccione la opción correspondiente.")
  with gr.Row():
      cp    = gr.Dropdown(label="Tipo de dolor toracico",
            info="0 = Angina típica\n1 = Angina atípica\n2 = Dolor no anginal\n3 = Asintomático",
            choices=["0","1","2","3"],
            placeholder="Seleccione la opción correspondiente.")
  with gr.Row():
      trtbps    = gr.Textbox(label="Presión arterial en reposo",
           info="(en mm Hg)",
           placeholder="Ingrese su presión arterial.")
  with gr.Row():
      chol    = gr.Textbox(label="Colesterol",
           info="(en mg/dl obtenido a través del sensor de IMC)",
           placeholder="Ingrese su nivel de colesterol.")
  with gr.Row():
      fbs   = gr.Dropdown(label="Azúcar en sangre en ayunas",
            info = "¿Es mayor a 120 mg/dl?\nSi (1) No (0)",
            choices =["0","1"],
            placeholder="Seleccione la opción correspondiente.")
  with gr.Row():
      restecg   = gr.Dropdown(label="Resultados electrocardiográficos en reposo",
            choices=["0","1","2"],
            info="0 = Normal\n1 = Normalidad de la onda ST-T\n2 = Hipertrofia ventricular izquierda",
            placeholder="Seleccione la opción correspondiente.")
  with gr.Row():
      thalachh    = gr.Textbox(label="Ritmo cardíaco máximo alcanzado",
           placeholder="Ingrese su ritmo cardíaco.")
  with gr.Row():
      exng    = gr.Dropdown(label="Angina inducida por ejercicio",
            choices=["0","1"],
            info= "Si (1)\nNo (0)",
            placeholder="Seleccione la opción correspondiente.")
  with gr.Row():
      oldpeak   = gr.Textbox(label ="Depresión ST(Old Peak)",
           info="Inducida por el ejercicio en relación al reposo.",
           placeholder="Ingrese el valor correspondiente.")
  with gr.Row():
      slp   = gr.Textbox(label="Pendiente del segmento ST",
           info="-- Valor 1: ascendente\n-- Valor 2: horizontal\n-- Valor 3: descendente",
           placeholder="Ingrese el valor en el pico del ejercicio.")
  with gr.Row():
      caa   = gr.Dropdown(label="Número de vasos principales",
            info="Valor obtenido en Fluoroscopia ",
            choices=["0","1","2","3"],
            placeholder="Seleccione la opción correspondiente.")
  with gr.Row():
      thall = gr.Dropdown(label="Resultado de la prueba de esfuerzo con talio",
            info="1 = normal.\n2 = defecto fijo.\n3 = defecto reversible.",
            choices=["1","2","3"],
            placeholder="Seleccione la opción correspondiente.")
  with gr.Row():
      prediction_btn = gr.Button(value = "Generar")
  with gr.Row():
    prediction = gr.Textbox(label=("Resultado"))
  with gr.Row():
    graphGR = gr.Image(label=("Gráfica"))

  prediction_btn.click(diagnosticar, 
                       inputs = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall], 
                       outputs = [prediction, graphGR],
                       api_name = "prediccion-riesgo-iam")
  data_dict = gr.Textbox(label = "Diccionario de datos", value = texto_atributos, interactive = False)

  examples = gr.Examples(label="Ejemplos",  examples=[
      [64, 1, 0, 120, 246, 0, 0, 96, 1, 2.2, 0, 1, 2], # riesgo inexistente
      [43,0,0,132,341,1,0,136,1,3,1,0,3,0], # riesgo inexistente
      [50,0,2,120,219,0,1,158,0,1.6,1,0,2,1], # hay riesgo
      [37,1,2,130,250,0,1,187,0,3.5,0,0,2]# hay riesgo
      ], inputs=[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall])

modelo.close()
modelo.launch(inline=False)